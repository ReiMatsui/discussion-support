#!/usr/bin/env python3
"""声紋帰属パイプラインのオフライン再生ハーネス（チューニング用）.

SonioxやpyannoteのAPIは一切叩かない。録音済みwavをGTの各発話区間（元セッション
turns.jsonl の ms〜end_ms）で切り出し、実運用（RecvLoop.flush）と同じ前処理・同じ
呼び出し（VoiceProfiles.classify: 登録者ゼロ・自動登録有効・max_human_speakers=3、
相槌は count=False、遡及リネーム反映）で時系列に再生し、数秒で帰属精度を測る。

【位置づけ】pyannoteクラスタ帰属（cluster_namer / SpeakerResolver / stt_fallback /
constrain_human_speaker_key）はオフラインでは再現できないため対象外。ここで測るのは
「声紋照合＋自動登録だけでどこまで当たるか」＝声紋のみの上限性能である。
STTラベルは元セッションの diag.jsonl の label（Sonioxの生ラベル）を再利用する。

使い方（ユーザーのMac、リポジトリ直下で）:
    uv run python eval/replay_attribution.py
    uv run python eval/replay_attribution.py --thresh 0.46 --margin 0.03
    uv run python eval/replay_attribution.py --sweep thresh=0.38,0.42,0.46 \\
        --sweep short_bonus=0.03,0.05,0.08
    uv run python eval/replay_attribution.py --dump   # 発話ごとの判定を表示
    # ライブランの実セグメンテーション＋実Sonioxラベル＋timeline形式GTで再生:
    uv run python eval/replay_attribution.py --from-session 2026-07-15_1306 \\
        --gt data/callhome/0856.gt.json --wav data/callhome/0856.wav

指標は eval/eval_speaker_gt.py と同じ流儀（単独話者のみ・最適1:1対応での帰属精度・
混同行列・未確定/誤帰属の内訳）に、判定機構（kind）別の正解率を加えたもの。
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import tempfile
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # eval/（_gtlib 用）

# _gtlib = GT読み込み・突合・最適対応の共通実装
import _gtlib  # noqa: E402

from das.asr.live._constants import _BACKCHANNEL_RE, SR, UNSURE_SPEAKER  # noqa: E402
from das.asr.live._voice_profiles import VoiceProfiles  # noqa: E402

# CLI/sweep で上書きできる VoiceProfiles のパラメータ。
# ctor: __init__ 引数、attr: 生成後に setattr する運用チューニング値。
_CTOR_PARAMS = {"thresh", "margin", "min_sec", "consist", "dedupe"}
_ATTR_PARAMS = {"short_floor", "short_bonus", "strict_sec",
                "enroll_min_total_chars", "enroll_win_sec", "enroll_consist_bonus",
                "label_purity_window", "person_th_offset"}
_ALL_PARAMS = sorted(_CTOR_PARAMS | _ATTR_PARAMS)


# ------------------------------------------------------------------
# 入力の読み込み
# ------------------------------------------------------------------

def _read_audio(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path)) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, "16bitモノラル前提"
        assert w.getframerate() == SR, f"サンプルレート{w.getframerate()} != {SR}"
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0   # RecvLoop.flush と同じ正規化


def load_inputs(gt_path: Path, wav_path: Path | None):
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    session = gt["session"]
    root = ROOT / "transcripts"
    turns = _gtlib.read_jsonl(root / f"{session}.turns.jsonl")
    diag_path = root / f"{session}.diag.jsonl"
    diag = _gtlib.read_jsonl(diag_path) if diag_path.exists() else []
    # 元セッションの diag から、発話開始ms→Soniox生ラベルの対応を引く
    label_by_ms = {d["ms"]: str(d["label"]) for d in diag if "label" in d}
    audio = _read_audio(wav_path or (root / f"{session}.wav"))
    items = []
    for t in sorted(turns, key=lambda t: (t["ms"], t["turn_id"])):
        code = gt["labels"].get(str(t["turn_id"])) or gt["labels"].get(t["turn_id"])
        items.append({
            "turn_id": t["turn_id"],
            "ms": t["ms"], "end_ms": t["end_ms"],
            "text": t.get("text", ""),
            "label": label_by_ms.get(t["ms"], "1"),   # diagが無ければ単一ラベル
            "gt": code,
        })
    return gt, items, audio


# タイムライン突合（支配80%/カバレッジ30%）は eval_speaker_gt と共通の
# _gtlib.gt_code_by_timeline を使う（採点系の互換維持。従来ここにあった
# 同名の複製実装は 2026-07-17 に _gtlib へ一本化）。
_gt_code_by_timeline = _gtlib.gt_code_by_timeline


def load_session_inputs(session: str, gt_path: Path, wav_path: Path):
    """ライブランの実セグメンテーション＋実Sonioxラベルで再生する入力を作る.

    turns.jsonl ベースの load_inputs は GTアノテータ形式（labels: turn_id→S1..）
    専用。こちらは transcripts/<session>.diag.jsonl の各行（label, ms, end）を
    そのまま発話系列として使い、ライブで実際に起きたセグメント区切り・STTラベルの
    揺れを忠実に再現する（CallHome実測 docs/design/
    handoff_2026-07-14_unregistered_speakers.md §13.1 の失敗ランをオフラインで
    再現・修正検証するためのモード）。テキストは同セッションの turns.jsonl から
    発話開始msで突合（相槌判定・chars用）。GTは timeline形式
    （eval/fetch_callhome_jpn.py が生成）を時間重なり（支配80%）で突合する。
    前提: 再生wavとライブ録音が同一クロック（ライブはこのwavのスピーカー再生）。
    """
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    if gt.get("kind") != "timeline":
        sys.exit("--from-session は timeline形式GT（eval/fetch_callhome_jpn.py 生成）専用です")
    root = ROOT / "transcripts"
    with open(root / f"{session}.diag.jsonl", encoding="utf-8") as f:
        diag = [json.loads(line) for line in f]
    text_by_ms: dict[int, str] = {}
    with open(root / f"{session}.turns.jsonl", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            text_by_ms.setdefault(t["ms"], t.get("text", ""))
    tl: dict[str, list[tuple[int, int]]] = {}
    for seg in gt["timeline"]:
        tl.setdefault(seg["speaker"], []).append((seg["start_ms"], seg["end_ms"]))
    audio = _read_audio(wav_path)
    items = []
    utts = sorted((d for d in diag if "label" in d and "ms" in d and "end" in d),
                  key=lambda d: d["ms"])
    for i, d in enumerate(utts):
        items.append({
            "turn_id": i + 1,
            "ms": d["ms"], "end_ms": d["end"],
            "text": text_by_ms.get(d["ms"], ""),
            "label": str(d["label"]),
            "gt": _gt_code_by_timeline(d["ms"], d["end"], tl),
        })
    return gt, items, audio, list(gt["speakers"])


# ------------------------------------------------------------------
# トラッカー生成（sweep用にモデルは1回だけロードし、状態をリセットして使い回す）
# ------------------------------------------------------------------

def build_tracker(model: str) -> VoiceProfiles:
    # 実運用の voices.json を誤って読み込まないよう、存在しないパスを渡す
    # （enroll() を呼ばないので書き込みも発生しない）。
    path = os.path.join(tempfile.mkdtemp(prefix="replay_vp_"), "voices.json")
    vp = VoiceProfiles(path=path, model=model)
    # sweep の再実行を速くする: 同一波形の埋め込みは決定的なのでメモ化する。
    orig_embed = vp._embed
    cache: dict[tuple, np.ndarray | None] = {}

    def cached_embed(wav: np.ndarray):
        key = (wav.size, hash(wav.tobytes()))
        if key not in cache:
            cache[key] = orig_embed(wav)
        return cache[key]

    vp._embed = cached_embed  # type: ignore[method-assign]
    return vp


def reset_tracker(vp: VoiceProfiles, params: dict, *,
                  max_speakers: int | None, hybrid: bool, auto: bool) -> None:
    """セッション状態を登録者ゼロの初期状態に戻し、パラメータを適用する."""
    d_th, d_dd, d_cs = VoiceProfiles.DEFAULTS[vp.model]
    vp.thresh, vp.dedupe, vp.consist = d_th, d_dd, d_cs
    vp.margin, vp.min_sec = 0.05, 1.0
    vp.short_floor, vp.short_bonus = 0.45, 0.08
    vp.strict_sec = 3.0
    vp.enroll_min_total_chars = 45
    vp.enroll_win_sec = 1.5
    vp.enroll_consist_bonus = 0.08
    vp.person_th_offset = 0.12
    vp.profiles = {}
    vp.sp_map = {}
    vp.label_hist = {}
    vp.label_purity_window = 4
    vp.label_embs = {}
    vp.pool = []
    vp.n_anon = 0
    vp.same_sims, vp.diff_sims = [], []
    vp.own_sims = {}
    vp.own_embs = {}
    vp._own_updates = {}
    vp.counts = {}
    vp.last = None
    vp._active_keys = set()
    vp.auto = auto
    vp.set_max_human_speakers(max_speakers)
    vp.set_hybrid(hybrid)
    for name, value in params.items():
        if not hasattr(vp, name):
            sys.exit(f"未知のパラメータ: {name}（使用可能: {', '.join(_ALL_PARAMS)}）")
        setattr(vp, name, value)


# ------------------------------------------------------------------
# 再生
# ------------------------------------------------------------------

def replay(vp: VoiceProfiles, items: list[dict], audio: np.ndarray) -> list[dict]:
    """発話を時系列に classify へ流し、[{pred, kind, ...}] を返す（遡及リネーム反映）."""
    results = []
    recent: list[tuple[int, int, str]] = []   # RecvLoop.recent_segs 相当
    _t0 = time.monotonic()
    for _idx, it in enumerate(items, 1):
        if _idx % 20 == 0 or _idx == len(items):
            _el = time.monotonic() - _t0
            _eta = _el / _idx * (len(items) - _idx)
            print(f"  ...{_idx}/{len(items)}発話 処理済み (経過{_el:.0f}s / 残り目安{_eta:.0f}s)",
                  file=sys.stderr, flush=True)
        s, e = int(it["ms"] * SR / 1000), int(it["end_ms"] * SR / 1000)
        wav = audio[s:e]
        text = it["text"].strip()
        is_bc = bool(_BACKCHANNEL_RE.match(text))
        overlapped = any(lbl != it["label"]
                         and min(e0, it["end_ms"]) - max(s0, it["ms"]) > 0
                         for s0, e0, lbl in recent)
        pred = vp.classify(wav, it["label"], overlapped=overlapped,
                           count=not is_bc, chars=len(text))
        d = vp.last or {}
        # RecvLoop.flush と同じ遡及リネーム（自動登録/合流の #ラベル→人物N）
        rename = d.get("rename")
        if rename:
            old, new = rename
            for r in results:
                if r["pred"] == old:
                    r["pred"] = new
                if r["pred_raw"] == old:
                    r["pred_raw"] = new
        # RecvLoop.flush と同じ相槌の最終規則（相槌レコードは未確定に落とす）。
        # 上書き前の予測は pred_raw に残す（相槌未確定ルールの妥当性採点用）。
        pred_raw = pred
        if is_bc:
            pred = UNSURE_SPEAKER
        recent.append((it["ms"], it["end_ms"], it["label"]))
        del recent[:-12]
        results.append({**it, "pred": pred, "pred_raw": pred_raw,
                        "kind": d.get("kind"), "bc": is_bc,
                        "sim": d.get("sim"), "second": d.get("second"),
                        "cand": d.get("name")})
    return results


# ------------------------------------------------------------------
# 指標（eval/eval_speaker_gt.py と同じ流儀）
# ------------------------------------------------------------------

def best_mapping(single: list[dict],
                 gts: tuple[str, ...] = ("S1", "S2", "S3")) -> tuple[float, dict]:
    """最適1:1対応（未確定は常に不正解扱い）での帰属精度と対応表を返す.

    _gtlib の共通実装に委譲（番兵は classify の生キー UNSURE_SPEAKER="?"。
    eval_speaker_gt 側は表示ラベル "未確定" と値が違うため、共通化では
    番兵を引数で渡す設計にしている。_gtlib docstring 参照）。
    """
    return _gtlib.best_mapping([(r["pred"], r["gt"]) for r in single],
                               gts, unsure=UNSURE_SPEAKER)


def summarize(results: list[dict], *, verbose: bool = True,
              gts: tuple[str, ...] = ("S1", "S2", "S3")) -> float:
    single = [r for r in results if r["gt"] in gts]
    if not single:
        sys.exit("GTの単独話者発話がありません")
    acc, mapping = best_mapping(single, gts)
    if not verbose:
        return acc
    n_multi = sum(1 for r in results if r["gt"] == "MULTI")
    n_unk = sum(1 for r in results if r["gt"] == "UNK")
    n_none = sum(1 for r in results if r["gt"] is None)
    print(f"= 全 {len(results)} 発話（単独話者 {len(single)}、"
          f"複数人 {n_multi}、不明 {n_unk}、正解範囲外 {n_none}）\n")

    # 混同行列
    sys_labels = sorted({r["pred"] for r in single})
    conf = {s: Counter() for s in sys_labels}
    for r in single:
        conf[r["pred"]][r["gt"]] += 1
    w = max(len(s) for s in sys_labels) + 2
    print("== 混同行列（行=システム, 列=正解） ==")
    print(" " * w + "  ".join(f"{g:>6}" for g in gts) + "   純度")
    for s in sys_labels:
        total = sum(conf[s].values())
        purity = max(conf[s].values()) / total if total else 0
        print(f"{s:<{w}}" + "  ".join(f"{conf[s][g]:>6}" for g in gts)
              + f"   {purity:.0%}" + ("  ←未確定" if s == UNSURE_SPEAKER else ""))

    print(f"\n== 最適1:1対応での帰属精度: {acc:.0%} ==")
    for s, g in mapping.items():
        print(f"  {s} = {g}")
    # 相槌内訳（ヘッドラインは従来どおり全発話）。相槌判定は本体と同じ
    # _BACKCHANNEL_RE（replay で bc として記録済み）。現行ルールでは相槌は
    # 常に未確定（＝常に不正解）なので、相槌除き精度は「相槌未確定ルールの
    # 妥当性」を数字で見るための参考値。対応表はヘッドラインと同一
    # （相槌はどの対応でも正解に寄与しないため最適対応は変わらない）。
    non_bc = [r for r in single if not r["bc"]]
    if non_bc:
        acc_nb = sum(1 for r in non_bc if mapping.get(r["pred"]) == r["gt"]) / len(non_bc)
        n_bc = len(single) - len(non_bc)
        bc_ok = sum(1 for r in single
                    if r["bc"] and mapping.get(r["pred_raw"]) == r["gt"])
        print(f"  相槌内訳: 全発話 {acc:.1%} (n={len(single)})"
              f" ／ 相槌除き {acc_nb:.1%} (n={len(non_bc)})"
              f" ／ 相槌 {n_bc}件（現行ルールで全件未確定。"
              f"未確定に落とさなければ正解 {bc_ok}/{n_bc}）")
    unsure = sum(1 for r in single if r["pred"] == UNSURE_SPEAKER)
    wrong = sum(1 for r in single
                if r["pred"] != UNSURE_SPEAKER and r["pred"] in mapping
                and mapping[r["pred"]] != r["gt"])
    unmapped = sum(1 for r in single
                   if r["pred"] != UNSURE_SPEAKER and r["pred"] not in mapping)
    n = len(single)
    print(f"  内訳: 未確定 {unsure/n:.0%} ／ 誤帰属 {wrong/n:.0%}"
          f" ／ 対応外ラベル {unmapped/n:.0%}")

    # 機構（kind）別の正解率
    print("\n== 判定機構（kind）別の正解率（単独話者のみ） ==")
    by_kind = defaultdict(list)
    for r in single:
        by_kind[r["kind"]].append(r)
    for kind, rows in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
        ok = sum(1 for r in rows if mapping.get(r["pred"]) == r["gt"])
        uns = sum(1 for r in rows if r["pred"] == UNSURE_SPEAKER)
        print(f"  {kind or '(なし)'}: n={len(rows)} 正解 {ok}/{len(rows)}"
              f" ({ok/len(rows):.0%}) 未確定 {uns}")
    return acc


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_sweep(specs: list[str]) -> list[dict]:
    """--sweep "thresh=0.38,0.42" 群をパラメータ組み合わせのリストに展開する."""
    axes: list[tuple[str, list[float]]] = []
    for spec in specs:
        name, _, vals = spec.partition("=")
        name = name.strip()
        if name not in _CTOR_PARAMS | _ATTR_PARAMS:
            sys.exit(f"--sweep: 未知のパラメータ {name}（使用可能: {', '.join(_ALL_PARAMS)}）")
        try:
            values = [float(v) for v in vals.split(",") if v.strip()]
        except ValueError:
            sys.exit(f"--sweep: 数値を解釈できません: {spec}")
        if not values:
            sys.exit(f"--sweep: 値がありません: {spec}")
        axes.append((name, values))
    combos = []
    for values in itertools.product(*(vs for _, vs in axes)):
        combos.append({name: v for (name, _), v in zip(axes, values, strict=True)})
    return combos


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="録音＋GTで声紋帰属パイプラインをオフライン再生し精度を測る"
                    "（pyannoteクラスタは対象外＝声紋のみの上限性能）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--gt", default=str(ROOT / "eval" / "gt_2026-07-14_142016.json"),
                   help="GT JSON（gt_annotator 形式 or timeline形式）")
    p.add_argument("--wav", default=None,
                   help="wavパス（省略時は transcripts/<session>.wav）")
    p.add_argument("--from-session", default=None, metavar="SESSION",
                   help="ライブランの diag.jsonl（実セグメンテーション＋実Sonioxラベル）"
                        "で再生する。--gt に timeline形式GT、--wav に音源wavが必須")
    p.add_argument("--model", default="redimnet",
                   choices=["redimnet", "ecapa", "resemblyzer"], help="声紋モデル")
    p.add_argument("--max-speakers", type=int, default=3,
                   help="max_human_speakers（実運用と同じ既定3）")
    p.add_argument("--hybrid", action=argparse.BooleanOptionalAction, default=True,
                   help="ハイブリッド相当（短発話の声紋照合を既知1人でも試みる）")
    p.add_argument("--auto", action=argparse.BooleanOptionalAction, default=True,
                   help="未知の声の自動登録（実運用と同じ既定ON）")
    # VoiceProfiles の主要しきい値（省略時はモデル別既定値）
    p.add_argument("--thresh", type=float, default=None, help="即時判定しきい値")
    p.add_argument("--margin", type=float, default=None, help="2位との差の下限")
    p.add_argument("--min-sec", type=float, default=None, help="通常照合の下限秒数")
    p.add_argument("--consist", type=float, default=None, help="蓄積の一貫性しきい値")
    p.add_argument("--dedupe", type=float, default=None, help="既存人物への合流しきい値")
    p.add_argument("--short-floor", type=float, default=None,
                   help="短発話厳格照合の下限秒数")
    p.add_argument("--short-bonus", type=float, default=None,
                   help="短発話照合のしきい値上乗せ")
    p.add_argument("--strict-sec", type=float, default=None,
                   help="厳格照合を要求する発話長の上限秒数")
    p.add_argument("--enroll-min-total-chars", type=float, default=None,
                   help="自動登録に必要な累積文字数")
    p.add_argument("--enroll-win-sec", type=float, default=None,
                   help="登録サンプルの分割窓長")
    p.add_argument("--enroll-consist-bonus", type=float, default=None,
                   help="登録時の一貫性しきい値上乗せ")
    p.add_argument("--label-purity-window", type=float, default=None,
                   help="ラベル継続の健全性窓（直近N回の照合成功が単一人物で"
                        "あることを要求。0で無効=旧挙動）")
    p.add_argument("--person-th-offset", type=float, default=None,
                   help="人物別しきい値のオフセット（p35 - この値。既定0.12）")
    p.add_argument("--sweep", action="append", default=[], metavar="NAME=V1,V2,...",
                   help=f"グリッド探索（複数指定で直積）。対象: {', '.join(_ALL_PARAMS)}")
    p.add_argument("--dump", action="store_true", help="発話ごとの判定を表示")
    return p


def cli_params(args) -> dict:
    params = {}
    for name in _ALL_PARAMS:
        v = getattr(args, name)
        if v is not None:
            params[name] = v
    return params


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    gt_codes: tuple[str, ...] = ("S1", "S2", "S3")
    if args.from_session:
        if not args.wav:
            sys.exit("--from-session には --wav が必須です")
        gt, items, audio, codes = load_session_inputs(
            args.from_session, Path(args.gt), Path(args.wav))
        gt_codes = tuple(codes)
    else:
        gt, items, audio = load_inputs(Path(args.gt),
                                       Path(args.wav) if args.wav else None)
    print(f"# セッション {gt['session']}: {len(items)}発話 / "
          f"音声 {audio.size/SR:.0f}s / モデル {args.model}"
          f" / max_speakers={args.max_speakers} hybrid={args.hybrid} auto={args.auto}")
    print("# 注: pyannoteクラスタ帰属は再現不可のため対象外（声紋のみの上限性能）\n")
    vp = build_tracker(args.model)
    base = cli_params(args)

    if args.sweep:
        combos = parse_sweep(args.sweep)
        print(f"# sweep: {len(combos)}通り（CLI指定値をベースに上書き）")
        scored = []
        for combo in combos:
            params = {**base, **combo}
            reset_tracker(vp, params, max_speakers=args.max_speakers,
                          hybrid=args.hybrid, auto=args.auto)
            acc = summarize(replay(vp, items, audio), verbose=False, gts=gt_codes)
            scored.append((acc, combo))
            print(f"  acc={acc:.1%}  "
                  + "  ".join(f"{k}={v}" for k, v in combo.items()))
        best_acc, best_combo = max(scored, key=lambda x: x[0])
        print(f"\n# 最良: acc={best_acc:.1%}  "
              + "  ".join(f"{k}={v}" for k, v in best_combo.items()))
        return

    reset_tracker(vp, base, max_speakers=args.max_speakers,
                  hybrid=args.hybrid, auto=args.auto)
    results = replay(vp, items, audio)
    if args.dump:
        for r in results:
            dur = (r["end_ms"] - r["ms"]) / 1000
            mark = "bc " if r["bc"] else ""
            sim = "" if r["sim"] is None else (
                f" sim={r['sim']:.2f}/"
                + ("--" if r["second"] is None else f"{r['second']:.2f}")
                + f"→{r['cand']}")
            print(f"[{r['ms']/1000:6.1f}s {dur:4.1f}s] gt={r['gt'] or '--'} "
                  f"pred={r['pred']:<8} kind={r['kind']}{sim} {mark}"
                  f"{r['text'][:30]}")
        print()
    summarize(results, gts=gt_codes)


if __name__ == "__main__":
    main()

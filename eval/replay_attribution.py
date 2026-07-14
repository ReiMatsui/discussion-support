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
import wave
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from das.asr.live._constants import _BACKCHANNEL_RE, SR, UNSURE_SPEAKER  # noqa: E402
from das.asr.live._voice_profiles import VoiceProfiles  # noqa: E402

# CLI/sweep で上書きできる VoiceProfiles のパラメータ。
# ctor: __init__ 引数、attr: 生成後に setattr する運用チューニング値。
_CTOR_PARAMS = {"thresh", "margin", "min_sec", "consist", "dedupe"}
_ATTR_PARAMS = {"short_floor", "short_bonus", "short_margin_mult",
                "enroll_min_total_chars", "enroll_win_sec", "enroll_consist_bonus"}
_ALL_PARAMS = sorted(_CTOR_PARAMS | _ATTR_PARAMS)


# ------------------------------------------------------------------
# 入力の読み込み
# ------------------------------------------------------------------

def load_inputs(gt_path: Path, wav_path: Path | None):
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    session = gt["session"]
    root = ROOT / "transcripts"
    turns_path = root / f"{session}.turns.jsonl"
    with open(turns_path, encoding="utf-8") as f:
        turns = [json.loads(line) for line in f]
    diag_path = root / f"{session}.diag.jsonl"
    diag = []
    if diag_path.exists():
        with open(diag_path, encoding="utf-8") as f:
            diag = [json.loads(line) for line in f]
    # 元セッションの diag から、発話開始ms→Soniox生ラベルの対応を引く
    label_by_ms = {d["ms"]: str(d["label"]) for d in diag if "label" in d}
    wav_path = wav_path or (root / f"{session}.wav")
    with wave.open(str(wav_path)) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2, "16bitモノラル前提"
        assert w.getframerate() == SR, f"サンプルレート{w.getframerate()} != {SR}"
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    audio = pcm.astype(np.float32) / 32768.0   # RecvLoop.flush と同じ正規化
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
    vp.short_floor, vp.short_bonus, vp.short_margin_mult = 0.45, 0.05, 2.0
    vp.enroll_min_total_chars = 45
    vp.enroll_win_sec = 1.5
    vp.enroll_consist_bonus = 0.08
    vp.profiles = {}
    vp.sp_map = {}
    vp.label_embs = {}
    vp.pool = []
    vp.n_anon = 0
    vp.same_sims, vp.diff_sims = [], []
    vp.own_sims = {}
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
    for it in items:
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
        # RecvLoop.flush と同じ相槌の最終規則（相槌レコードは未確定に落とす）
        if is_bc:
            pred = UNSURE_SPEAKER
        recent.append((it["ms"], it["end_ms"], it["label"]))
        del recent[:-12]
        results.append({**it, "pred": pred, "kind": d.get("kind"), "bc": is_bc})
    return results


# ------------------------------------------------------------------
# 指標（eval/eval_speaker_gt.py と同じ流儀）
# ------------------------------------------------------------------

def best_mapping(single: list[dict]) -> tuple[float, dict]:
    """最適1:1対応（未確定は常に不正解扱い）での帰属精度と対応表を返す."""
    gts = ["S1", "S2", "S3"]
    real = sorted({r["pred"] for r in single if r["pred"] != UNSURE_SPEAKER})
    best_acc, best_map = 0.0, {}
    for k in range(min(3, len(real)) + 1):
        for perm in permutations(real, k):
            for gsel in permutations(gts, k):
                m = dict(zip(perm, gsel, strict=False))
                acc = sum(1 for r in single if m.get(r["pred"]) == r["gt"]) / len(single)
                if acc > best_acc:
                    best_acc, best_map = acc, m
    return best_acc, best_map


def summarize(results: list[dict], *, verbose: bool = True) -> float:
    single = [r for r in results if r["gt"] in ("S1", "S2", "S3")]
    if not single:
        sys.exit("GTの単独話者発話がありません")
    acc, mapping = best_mapping(single)
    if not verbose:
        return acc
    n_multi = sum(1 for r in results if r["gt"] == "MULTI")
    n_unk = sum(1 for r in results if r["gt"] == "UNK")
    print(f"= GT付き {len(results)} 発話（単独話者 {len(single)}、"
          f"複数人 {n_multi}、不明 {n_unk}）\n")

    # 混同行列
    gts = ["S1", "S2", "S3"]
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
                   help="GT JSON（gt_annotator で作成したもの）")
    p.add_argument("--wav", default=None,
                   help="wavパス（省略時は transcripts/<session>.wav）")
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
    p.add_argument("--short-margin-mult", type=float, default=None,
                   help="短発話照合のmargin倍率")
    p.add_argument("--enroll-min-total-chars", type=float, default=None,
                   help="自動登録に必要な累積文字数")
    p.add_argument("--enroll-win-sec", type=float, default=None,
                   help="登録サンプルの分割窓長")
    p.add_argument("--enroll-consist-bonus", type=float, default=None,
                   help="登録時の一貫性しきい値上乗せ")
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
    gt, items, audio = load_inputs(Path(args.gt), Path(args.wav) if args.wav else None)
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
            acc = summarize(replay(vp, items, audio), verbose=False)
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
            print(f"[{r['ms']/1000:6.1f}s {dur:4.1f}s] gt={r['gt'] or '--'} "
                  f"pred={r['pred']:<8} kind={r['kind']} {mark}"
                  f"{r['text'][:30]}")
        print()
    summarize(results)


if __name__ == "__main__":
    main()

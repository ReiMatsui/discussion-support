#!/usr/bin/env python3
"""記録済みランを**本番コードのまま**再生する（帰属の全層）.

eval/replay_attribution.py との違いは、再生できる範囲:

  - replay_attribution.py : 声紋層（VoiceProfiles.classify）だけ。クラスタ層
    （ClusterVoiceNamer / SpeakerResolver / 匿名キー / constrain）は「再現不可」
    として対象外だった
  - このスクリプト        : ``_attribution.decide_speaker`` と ``SessionState`` の
    実物を通す＝**ライブと同じ判定経路**

再生できるようになった理由は、判定の入力を diag に残すようにしたから
（2026-07-25。handoff §23）。従来 diarization provider の話者区間はメモリ上に
しか無く実行後に失われていたため、クラスタ層はオフラインで動かしようがなかった。
いまは1発話ごとに:

  - ``diar``  : その判定が実際に見た話者区間の窓 [[source, speaker, start, end], …]
  - ``ov``    : classify に渡した overlapped
  - ``enr``   : classify に渡した enroll（エコー窓依存。記録からは再現できない）
  - ``chars`` : 文字数（自動登録の累積判定に効く）

が残り、先頭行の ``session_config`` に構成（想定話者数・声紋モデル・ハイブリッド
可否など）が入る。これらと wav があれば、判定は決定的に再生できる。

**何のためか**: 帰属の設計変更（門番・閾値・統合規則）を、実会話を録り直さずに
評価するため。ライブ1ランの比較では Soniox の揺れ（実質 ±12pt）が支配的で
差が読めない（§15.13）。同じ記録に対して旧コードと新コードを流せば、
差はコードの差だけになる。

使い方:
    uv run python eval/replay_live_attribution.py --session 2026-07-25_1723
    uv run python eval/replay_live_attribution.py --session 2026-07-25_1723 --gt eval/gt_...json
    uv run python eval/replay_live_attribution.py --session X --set vp_mint_cluster_link=1
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import tempfile
import wave
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _gtlib  # noqa: E402

from das.asr.live._attribution import decide_speaker  # noqa: E402
from das.asr.live._cluster_naming import ClusterVoiceNamer  # noqa: E402
from das.asr.live._constants import (  # noqa: E402
    _BACKCHANNEL_RE,
    SR,
    UNSURE_SPEAKER,
)
from das.asr.live._diarization import DiarizationEvent, SpeakerResolver  # noqa: E402
from das.asr.live._session_state import SessionState  # noqa: E402
from das.asr.live._voice_profiles import VoiceProfiles  # noqa: E402


class _RecordedProvider:
    """記録された窓をそのまま返す provider スタブ.

    ヒステリシス（SessionState._uses_pyannote_hysteresis）は provider の name しか
    見ないため、name さえ合っていればライブと同じ経路になる。区間の供給は
    diarization_window の差し替えで行う（下記 _install_window 参照）——記録に
    残っているのは「判定が実際に見た窓」であって provider の内部状態ではないため、
    確定済み/進行中の区別を復元する必要がない。
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def drain_events(self) -> list[DiarizationEvent]:
        return []

    def active_events(self) -> list[DiarizationEvent]:
        return []


def _install_window(state: SessionState, windows: dict[int, list[DiarizationEvent]]):
    """発話開始msをキーに、記録された窓を返すよう diarization_window を差し替える."""
    def _window(start_ms, end_ms, _w=windows):
        return list(_w.get(start_ms, []))
    state.diarization_window = _window   # type: ignore[method-assign]


# ------------------------------------------------------------------
# 記録の読み込み
# ------------------------------------------------------------------

def load_session(session: str, root: Path | None = None) -> dict:
    root = root or ROOT / "transcripts"
    cfg: dict = {}
    utts: list[dict] = []
    for line in _gtlib.read_jsonl(root / f"{session}.diag.jsonl"):
        t = line.get("type")
        if t == "session_config":
            cfg = line
        elif t is None and "label" in line and "key" in line:
            utts.append(line)
    text_by_ms: dict[int, str] = {}
    for r in _gtlib.read_jsonl(root / f"{session}.turns.jsonl"):
        text_by_ms.setdefault(r["ms"], r.get("text", ""))
    with wave.open(str(root / f"{session}.wav")) as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        assert w.getframerate() == SR, f"サンプルレート{w.getframerate()} != {SR}"
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    audio = pcm.astype(np.float32) / 32768.0
    return {"config": cfg, "utts": utts, "text": text_by_ms, "audio": audio,
            "session": session}


def has_replayable_inputs(utts: list[dict]) -> bool:
    """判定の入力（diar 窓・classify フラグ）が記録されているか."""
    return any("diar" in u or "enr" in u for u in utts)


# ------------------------------------------------------------------
# 再生
# ------------------------------------------------------------------

def build_state(cfg: dict, overrides: dict, tmp: Path, tracker=None):
    """記録された構成で SessionState / VoiceProfiles / ClusterVoiceNamer を組む.

    ``tracker`` を渡すと声紋モデルの読み込みを省略できる（テスト用の注入口）。
    """
    tmp.mkdir(parents=True, exist_ok=True)
    diarization = overrides.get("diarization", cfg.get("diarization") or "pyannote")
    max_speakers = overrides.get("diarization_max_speakers",
                                 cfg.get("diarization_max_speakers"))
    vp_model = overrides.get("vp_model", cfg.get("vp_model") or "redimnet")
    vp_auto = bool(overrides.get("vp_auto", cfg.get("vp_auto", True)))
    naming = bool(overrides.get("vp_cluster_naming",
                                cfg.get("vp_cluster_naming", True)))
    mint_link = bool(overrides.get("vp_mint_cluster_link",
                                   cfg.get("vp_mint_cluster_link", False)))
    if tracker is None:
        # 実運用の voices.json を読まないよう存在しないパスを渡す（登録者ゼロ再現）
        tracker = VoiceProfiles(path=str(tmp / "voices.json"), model=vp_model,
                                auto=vp_auto)
    tracker.set_max_human_speakers(max_speakers)
    tracker.set_hybrid(naming)
    namer = ClusterVoiceNamer(tracker) if naming else None
    args = SimpleNamespace(diarization_max_speakers=max_speakers,
                           vp_mint_cluster_link=mint_link, vp_debug=False,
                           lang="ja")
    state = SessionState(
        args=args, started=datetime.datetime(2026, 1, 1),
        out_path=str(tmp / "o.md"), html_path=str(tmp / "o.html"),
        diag_path=str(tmp / "o.diag"), turns_path=str(tmp / "o.turns"),
        wav_path=str(tmp / "o.wav"), tracker=tracker, serve=False,
        diarization_provider=_RecordedProvider(diarization),
        speaker_resolver=SpeakerResolver(), cluster_namer=namer)
    state.save = lambda *a, **k: None       # type: ignore[method-assign]
    return state, tracker, args


def replay(rec: dict, overrides: dict, tmp: Path, tracker=None) -> list[dict]:
    """記録を本番の判定経路に流し、発話ごとの最終キーを返す."""
    from das.asr.live._recv_loop import RecvLoop

    utts, audio = rec["utts"], rec["audio"]
    windows = {
        u["ms"]: [DiarizationEvent(start_ms=e[2], end_ms=e[3],
                                   speaker=e[1], source=e[0])
                  for e in u.get("diar", [])]
        for u in utts
    }
    state, tracker, args = build_state(rec["config"], overrides, tmp, tracker)
    _install_window(state, windows)
    loop = RecvLoop(state, args, backend=None)   # flush は使わず部品だけ借りる
    out: list[dict] = []
    for u in utts:
        ms, end = int(u["ms"]), int(u["end"])
        wav = audio[int(ms * SR / 1000):int(end * SR / 1000)]
        text = rec["text"].get(ms, "")
        is_bc = bool(_BACKCHANNEL_RE.match(text.strip()))
        sp_id = tracker.classify(
            wav, u["label"],
            overlapped=bool(u.get("ov", False)),
            count=not is_bc,
            enroll=bool(u.get("enr", not is_bc)),
            chars=int(u.get("chars", len(text.strip()))))
        d = tracker.last
        rec_extra: dict = {}
        # ライブ（flush）と同じ順序: 鋳造/合流の遡及リネーム → 鋳造リンク → 帰属
        if d and d.get("kind") in ("自動登録", "合流"):
            if d.get("rename"):
                state.rekey(*d["rename"])
            if d["kind"] == "自動登録":
                loop.cur_ms, loop.cur_end = ms, end
                loop._link_mint_to_cluster(d["name"])
        sp_id = decide_speaker(state, sp_id=sp_id, d=d, wav=wav,
                               start_ms=ms, end_ms=end,
                               rec_extra=rec_extra, vp_debug=False)
        final = state.constrain_human_speaker_key(
            UNSURE_SPEAKER if is_bc else sp_id)
        state.records.append({"ms": ms, "end_ms": end, "speaker": final,
                              "text": text})
        state.disp_name(final)
        out.append({"ms": ms, "end_ms": end, "text": text, "bc": is_bc,
                    "kind": d.get("kind") if d else None,
                    "recorded": u.get("final_key")})
    # rekey は records を遡及的に書き換えるので、最終状態を採用する
    for row, r in zip(out, [r for r in state.records if "speaker" in r], strict=True):
        row["pred"] = str(r["speaker"])
    return out


# ------------------------------------------------------------------
# 採点
# ------------------------------------------------------------------

def fidelity(rows: list[dict]) -> tuple[float, int]:
    """記録された final_key をどれだけ再現したか（同一構成での自己一致率）."""
    pairs = [(r["pred"], str(r["recorded"])) for r in rows
             if r.get("recorded") is not None]
    if not pairs:
        return 0.0, 0
    return sum(1 for a, b in pairs if a == b) / len(pairs), len(pairs)


def score(rows: list[dict], gt_path: Path | None, session: str,
          root: Path | None = None) -> dict | None:
    if gt_path is None or not gt_path.exists():
        return None
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    turns = _gtlib.read_jsonl((root or ROOT / "transcripts")
                              / f"{session}.turns.jsonl")
    code_by_ms = {t["ms"]: gt["labels"].get(str(t["turn_id"])) for t in turns}
    pairs = [(r["pred"], code_by_ms.get(r["ms"]), r["bc"]) for r in rows]
    single = [(p, g, bc) for p, g, bc in pairs if g in ("S1", "S2", "S3")]
    if not single:
        return None
    acc, mapping = _gtlib.best_mapping([(p, g) for p, g, _ in single],
                                       ("S1", "S2", "S3"), unsure=UNSURE_SPEAKER)
    non_bc = [(p, g) for p, g, bc in single if not bc]
    n = len(non_bc) or 1
    return {
        "n": len(single), "n_nonbc": len(non_bc), "acc_all": acc,
        "acc": sum(1 for p, g in non_bc if mapping.get(p) == g) / n,
        "unsure": sum(1 for p, _ in non_bc if p == UNSURE_SPEAKER) / n,
        "wrong": sum(1 for p, g in non_bc
                     if p != UNSURE_SPEAKER and mapping.get(p, "__x__") != g) / n,
    }


def parse_overrides(specs: list[str]) -> dict:
    out: dict = {}
    for spec in specs:
        name, _, raw = spec.partition("=")
        name = name.strip()
        raw = raw.strip()
        if raw.lower() in ("1", "true", "yes", "on"):
            out[name] = True
        elif raw.lower() in ("0", "false", "no", "off"):
            out[name] = False
        elif raw.lower() in ("none", "null", ""):
            out[name] = None
        else:
            try:
                out[name] = int(raw)
            except ValueError:
                out[name] = raw
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--session", required=True, help="transcripts/<SESSION>.* を再生")
    p.add_argument("--gt", default=None, help="GT JSON（あれば精度も出す）")
    p.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                   help="構成の上書き（例: vp_mint_cluster_link=1 / "
                        "diarization_max_speakers=2）")
    p.add_argument("--dump", action="store_true", help="発話ごとの判定を表示")
    args = p.parse_args(argv)

    rec = load_session(args.session)
    if not has_replayable_inputs(rec["utts"]):
        sys.exit(
            f"# {args.session} には判定の入力（diar 窓 / classify フラグ）が"
            "記録されていません。\n"
            "# この再生には 2026-07-25 以降のコードで録ったランが必要です"
            "（それ以前のランは eval/replay_attribution.py で声紋層のみ再生できます）")
    overrides = parse_overrides(args.set)
    tmp = Path(tempfile.mkdtemp(prefix="replay_live_"))
    print(f"# セッション {args.session}: {len(rec['utts'])}発話 / "
          f"音声 {rec['audio'].size / SR:.0f}s")
    print(f"# 構成: {json.dumps(rec['config'], ensure_ascii=False)}")
    if overrides:
        print(f"# 上書き: {overrides}")
    rows = replay(rec, overrides, tmp)

    fid, n_fid = fidelity(rows)
    print(f"\n== 記録との自己一致: {fid:.1%} ({n_fid}発話) ==")
    if not overrides and fid < 1.0:
        print("#   ※ 上書き無しで 100% にならない場合、再生に必要な入力が"
              "まだ足りていない（差分を --dump で確認）")
    if args.dump:
        for r in rows:
            mark = "" if r["pred"] == str(r["recorded"]) else "  ← 記録と相違"
            print(f"[{r['ms'] / 1000:7.1f}s] {r['pred']:<10}"
                  f"（記録 {r['recorded']}）{r['kind'] or '':<8}"
                  f"{'bc' if r['bc'] else '  '} {r['text'][:26]}{mark}")
    print("\n== 最終キーの分布 ==")
    for k, n in Counter(r["pred"] for r in rows).most_common():
        print(f"  {k}: {n}")
    sc = score(rows, Path(args.gt) if args.gt else None, args.session)
    if sc:
        print(f"\n== 精度（単独話者 {sc['n']}発話 / 相槌除き {sc['n_nonbc']}） ==")
        print(f"  実質 {sc['acc']:.1%} ／ 誤帰属 {sc['wrong']:.1%}"
              f" ／ 未確定 {sc['unsure']:.1%}（全発話 {sc['acc_all']:.1%}）")


if __name__ == "__main__":
    main()

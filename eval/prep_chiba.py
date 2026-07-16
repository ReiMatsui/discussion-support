#!/usr/bin/env python3
"""千葉大3人会話コーパス(Chiba3Party)を話者帰属評価用に変換する.

対面・日本語・自然会話（高速の掛け合い・相槌）での帰属性能を測るための
前処理。話者別ヘッドセット録音（Wav1/<conv>-{A,B,C}.wav, 16kHz mono）と
トークン単位の時刻付き形態論情報（Morph/<conv>.csv, cp932）から:

1. 話者タイムラインGTを自動生成（手作業アノテーション不要）:
   - transcripts/<セッション名>.turns.jsonl  … GT定義セッション（turn_id→区間）
   - eval/gt_<セッション名>.json             … labels形式（A→S1, B→S2, C→S3）
2. 3チャンネルを1本のモノラル16kHz wavにミックス（実走の入力）

測定経路は handoff §14 の CallHome と同一（replay_attribution はSonioxラベル
前提のためコーパスには使わない。§14「測定手順の知見」参照）:

    uv run python eval/prep_chiba.py --conv chiba0132 --minutes 5
    uv run das listen-soniox --hybrid --max-speakers 3 \
        --wav data/chiba/chiba0132_mix_5min.wav --soniox-args "--no-agent"
    uv run python eval/eval_speaker_gt.py eval/gt_chiba0132m5.json <新セッション名>

注意:
- ヘッドセット録音のミックスは実際の卓上マイクより音響的にクリーン。
  これで測れるのは「自然な掛け合いに区切りと判定が耐えるか」であり、
  部屋の残響への耐性ではない（docs/design/handoff §14 の位置づけと同じ）。
- 個人名はコーパス側でビープ音に置換されている。該当区間はSTT・声紋の
  両方でノイズになるが、件数は少なく採点はタイムライン方式で吸収される。
"""
from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "chiba" / "Chiba3Party"
SR = 16000
SPEAKERS = ("A", "B", "C")
GT_CODE = {"A": "S1", "B": "S2", "C": "S3"}


def load_tokens(conv: str) -> list[tuple[float, float, str, str]]:
    """Morph CSV を (start, end, who, text) のリストに読む（cp932）."""
    rows = []
    with open(CORPUS / "Morph" / f"{conv}.csv", encoding="cp932") as f:
        for r in csv.DictReader(f):
            who = (r.get("who") or "").strip()
            if who not in SPEAKERS:
                continue
            try:
                s, e = float(r["startTime"]), float(r["endTime"])
            except (KeyError, ValueError):
                continue
            if e > s:
                rows.append((s, e, who, r.get("text") or ""))
    rows.sort(key=lambda x: (x[0], x[1]))
    return rows


def merge_utterances(tokens, gap_sec: float):
    """話者ごとにトークンを発話区間へ併合する（間隙 gap_sec 以下は連結）."""
    segs: list[dict] = []
    cur: dict[str, dict] = {}
    for s, e, who, text in tokens:
        c = cur.get(who)
        if c is not None and s - c["end"] <= gap_sec:
            c["end"] = max(c["end"], e)
            c["text"] += text
        else:
            if c is not None:
                segs.append(c)
            cur[who] = {"start": s, "end": e, "who": who, "text": text}
    segs.extend(cur.values())
    segs.sort(key=lambda x: (x["start"], x["end"]))
    return segs


def clip_minutes(segs, minutes: float | None):
    if minutes is None:
        return segs
    limit = minutes * 60.0
    out = []
    for g in segs:
        if g["start"] >= limit:
            continue
        h = dict(g)
        h["end"] = min(h["end"], limit)
        if h["end"] > h["start"]:
            out.append(h)
    return out


def overlap_stats(segs) -> tuple[float, float]:
    """総発話時間と、複数人が同時に話している時間の比率（10ms格子で近似）."""
    if not segs:
        return 0.0, 0.0
    end = max(g["end"] for g in segs)
    grid = np.zeros(int(end * 100) + 1, dtype=np.int8)
    for g in segs:
        grid[int(g["start"] * 100):int(g["end"] * 100)] += 1
    speech = float((grid > 0).sum()) / 100.0
    overlap = float((grid > 1).sum()) / 100.0
    return speech, (overlap / speech if speech else 0.0)


def mix_wavs(conv: str, minutes: float | None, out_path: Path) -> float:
    """話者別3chを加算ミックスし、ピーク正規化してモノラル16kで書き出す."""
    waves = []
    for sp in SPEAKERS:
        with wave.open(str(CORPUS / "Wav1" / f"{conv}-{sp}.wav")) as w:
            assert w.getframerate() == SR and w.getnchannels() == 1, \
                f"{conv}-{sp}: 期待形式は16kHz mono"
            waves.append(np.frombuffer(
                w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32))
    n = max(len(x) for x in waves)
    if minutes is not None:
        n = min(n, int(minutes * 60 * SR))
    mixed = np.zeros(n, dtype=np.float32)
    for x in waves:
        mixed[:min(n, len(x))] += x[:n]
    peak = float(np.abs(mixed).max()) or 1.0
    if peak > 32000.0:          # クリップ回避時のみ縮める（無用な音量低下を避ける）
        mixed *= 32000.0 / peak
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(mixed.astype(np.int16).tobytes())
    return n / SR


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--conv", default="chiba0132", help="会話ID（例: chiba0132）")
    p.add_argument("--minutes", type=float, default=None,
                   help="先頭から切り出す分数（未指定なら全体≒9.5分）")
    p.add_argument("--gap", type=float, default=0.5,
                   help="トークンをひとつの発話に併合する最大間隙秒")
    a = p.parse_args()

    session = a.conv + (f"m{a.minutes:g}" if a.minutes else "")
    tokens = load_tokens(a.conv)
    segs = clip_minutes(merge_utterances(tokens, a.gap), a.minutes)
    speech, ov = overlap_stats(segs)

    turns_path = ROOT / "transcripts" / f"{session}.turns.jsonl"
    with open(turns_path, "w", encoding="utf-8") as f:
        for i, g in enumerate(segs, 1):
            f.write(json.dumps({
                "turn_id": i, "speaker": g["who"], "text": g["text"],
                "ms": int(g["start"] * 1000), "end_ms": int(g["end"] * 1000),
            }, ensure_ascii=False) + "\n")

    gt_path = ROOT / "eval" / f"gt_{session}.json"
    gt = {"session": session,
          "labels": {str(i): GT_CODE[g["who"]] for i, g in enumerate(segs, 1)},
          "speaker_names": {GT_CODE[s]: f"話者{s}" for s in SPEAKERS}}
    gt_path.write_text(json.dumps(gt, ensure_ascii=False, indent=1),
                       encoding="utf-8")

    mix_name = f"{a.conv}_mix" + (f"_{a.minutes:g}min" if a.minutes else "") + ".wav"
    mix_path = ROOT / "data" / "chiba" / mix_name
    dur = mix_wavs(a.conv, a.minutes, mix_path)

    per_sp = {s: sum(g["end"] - g["start"] for g in segs if g["who"] == s)
              for s in SPEAKERS}
    print(f"= {a.conv}: 発話 {len(segs)} 件, 音声 {dur:.1f}s, "
          f"発話総時間 {speech:.1f}s, 重なり比率 {ov:.0%}")
    print("  話者別発話時間: " + ", ".join(f"{s}={per_sp[s]:.1f}s" for s in SPEAKERS))
    print(f"  GT: {gt_path.relative_to(ROOT)} / {turns_path.relative_to(ROOT)}")
    print(f"  音声: {mix_path.relative_to(ROOT)}")
    print("\n次の手順（実機）:")
    print(f"  uv run das listen-soniox --hybrid --max-speakers 3 \\")
    print(f"      --wav {mix_path.relative_to(ROOT)} --soniox-args \"--no-agent\"")
    print(f"  uv run python eval/eval_speaker_gt.py {gt_path.relative_to(ROOT)} <新セッション名>")


if __name__ == "__main__":
    main()

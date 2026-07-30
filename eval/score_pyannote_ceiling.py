#!/usr/bin/env python3
"""pyannoteバッチ分離の「天井」採点: 正解タイムラインに対するクラスタ一貫性.

目的: scripts/benchmark_pyannote.py が保存した transcripts/<session>.pyannote_bench.json
（precision-2 バッチの話者区間）を、ユーザー作成の正解（eval/gt_*.json）で採点し、
「区切り・クラスタを最高品質にした場合に話者帰属が何点出るか」の上限を測る。
上限が高ければ『pyannote境界で切り直す構造変更』に投資する根拠になる
（docs/design/handoff_2026-07-14_unregistered_speakers.md §11 の選択肢(b)の事前検証）。

使い方:
    uv run python scripts/benchmark_pyannote.py --wav transcripts/2026-07-14_142016.wav --num-speakers 3
    uv run python eval/score_pyannote_ceiling.py eval/gt_2026-07-14_142016.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(gt_path: str) -> None:
    gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    session = gt["session"]
    with open(ROOT / "transcripts" / f"{session}.turns.jsonl",
              encoding="utf-8") as f:
        turns = [json.loads(line) for line in f]
    bench_path = ROOT / "transcripts" / f"{session}.pyannote_bench.json"
    if not bench_path.exists():
        sys.exit(f"{bench_path} がありません。先に benchmark_pyannote.py を実行してください")
    bench = json.loads(bench_path.read_text(encoding="utf-8"))
    # 保存形式からpyannote区間を取り出す（segments / diarization どちらのキーでも）
    segs = bench.get("pyannote_segments") or bench.get("segments") or \
        (bench.get("output") or {}).get("diarization") or bench.get("diarization")
    if segs is None:
        sys.exit(f"pyannote区間が読めません。キー一覧: {list(bench.keys())}")
    norm = []
    for s in segs:
        start = s.get("start_ms", s.get("start", 0) * 1000 if "start" in s else 0)
        end = s.get("end_ms", s.get("end", 0) * 1000 if "end" in s else 0)
        norm.append((float(start), float(end), s["speaker"]))

    # GT発話（=GT区切りそのもの）ごとに、pyannoteの支配クラスタを求める
    rows = []  # (gt_code, cluster or None)
    for t in turns:
        code = gt["labels"].get(str(t["turn_id"]))
        if code not in ("S1", "S2", "S3"):
            continue
        ovs: dict[str, float] = defaultdict(float)
        for a, b, sp in norm:
            ov = min(t["end_ms"], b) - max(t["ms"], a)
            if ov > 0:
                ovs[sp] += ov
        cluster = max(ovs, key=ovs.get) if ovs else None
        rows.append((code, cluster))

    clusters = sorted({c for _, c in rows if c})
    print(f"= {session}: GT単独話者 {len(rows)} 発話 / pyannoteクラスタ {len(clusters)} 個")
    conf = {c: Counter() for c in clusters}
    for g, c in rows:
        if c:
            conf[c][g] += 1
    for c in clusters:
        tot = sum(conf[c].values())
        pur = max(conf[c].values()) / tot
        print(f"  {c}: n={tot} 純度{pur:.0%} 内訳{dict(conf[c])}")

    # 最適1:1対応（クラスタに完璧な名前付けができた場合の帰属精度＝天井）
    best, bm = 0.0, {}
    for k in range(min(3, len(clusters)) + 1):
        for perm in permutations(clusters, k):
            for gsel in permutations(["S1", "S2", "S3"], k):
                m = dict(zip(perm, gsel, strict=False))
                a = sum(1 for g, c in rows if m.get(c) == g) / len(rows)
                if a > best:
                    best, bm = a, m
    uncov = sum(1 for _, c in rows if c is None)
    print(f"\n== 天井（完璧な名前付けを仮定した帰属精度）: {best:.0%} ==")
    print(f"   対応: {bm} ／ pyannote区間が無い発話: {uncov}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])

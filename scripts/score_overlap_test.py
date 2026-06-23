#!/usr/bin/env python3
"""重なりテストの採点: answer.json vs システム出力(turns.jsonl).

指標:
  - 話者帰属: 正解話者⇔システム話者(人物N等)を最適1対1対応づけした上での発話正解率。
    重なりあり/なしの内訳も出す。
  - テキスト: 正解文と認識文の文字類似度(difflib)。発話の取りこぼし率。

使い方:
  uv run python scripts/score_overlap_test.py data/overlap_test/A_clean.answer.json \
      transcripts/<日時>.turns.jsonl
"""
from __future__ import annotations

import argparse
import difflib
import json
from collections import defaultdict
from pathlib import Path


def best_mapping(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """(システム話者, 正解話者) ペアの頻度から貪欲に1対1対応を作る."""
    count: dict[tuple[str, str], int] = defaultdict(int)
    for s, g in pairs:
        count[(s, g)] += 1
    mapping: dict[str, str] = {}
    used_g: set[str] = set()
    for (s, g), _ in sorted(count.items(), key=lambda kv: -kv[1]):
        if s not in mapping and g not in used_g:
            mapping[s] = g
            used_g.add(g)
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("answer", type=Path)
    ap.add_argument("turns", type=Path)
    args = ap.parse_args()

    ans = json.loads(args.answer.read_text(encoding="utf-8"))["turns"]
    sys_turns = [json.loads(line) for line in args.turns.read_text(encoding="utf-8").splitlines() if line.strip()]
    for t in sys_turns:
        t["s"] = (t.get("ms") or 0) / 1000.0

    # 正解発話ごとに、時間窓に最も重なるシステム発話群を対応づけ
    matches = []   # (正解turn, [システムturn,...])
    for a in ans:
        hits = [t for t in sys_turns
                if a["start_s"] - 1.0 <= t["s"] <= a["end_s"] + 1.0
                and difflib.SequenceMatcher(None, a["text"], t["text"]).ratio() > 0.25]
        if not hits:   # テキストが薄くても時間で拾う
            hits = [t for t in sys_turns if a["start_s"] - 0.5 <= t["s"] <= a["end_s"]]
        matches.append((a, hits))

    pairs = []
    for a, hits in matches:
        for h in hits:
            pairs.append((h["speaker"], a["speaker"]))
    mapping = best_mapping(pairs)

    stats = {True: [0, 0], False: [0, 0]}   # overlapped -> [正解数, 総数]
    text_scores = []
    missed = 0
    for a, hits in matches:
        if not hits:
            missed += 1
            stats[a["overlapped"]][1] += 1
            continue
        # 多数決の話者
        votes: dict[str, int] = defaultdict(int)
        joined = ""
        for h in hits:
            votes[mapping.get(h["speaker"], h["speaker"])] += len(h["text"])
            joined += h["text"]
        pred = max(votes.items(), key=lambda kv: kv[1])[0]
        ok = pred == a["speaker"]
        stats[a["overlapped"]][0] += int(ok)
        stats[a["overlapped"]][1] += 1
        text_scores.append(difflib.SequenceMatcher(None, a["text"], joined).ratio())

    total_ok = stats[True][0] + stats[False][0]
    total_n = stats[True][1] + stats[False][1]
    print(f"対応づけ: {mapping}")
    print(f"発話の取りこぼし: {missed}/{len(ans)}")
    print(f"話者帰属の正解率: 全体 {total_ok}/{total_n} = {total_ok/max(total_n,1)*100:.0f}%")
    for ov, label in ((False, "重なりなし"), (True, "重なりあり")):
        ok, n = stats[ov]
        if n:
            print(f"  {label}: {ok}/{n} = {ok/n*100:.0f}%")
    if text_scores:
        import statistics
        print(f"テキスト類似度(文字): 平均 {statistics.mean(text_scores)*100:.0f}% "
              f"最低 {min(text_scores)*100:.0f}%")


if __name__ == "__main__":
    main()

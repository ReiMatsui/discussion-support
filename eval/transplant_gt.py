#!/usr/bin/env python3
"""話者タイムラインGTを別セッションの発話区切りへ移植する.

コーパス評価（eval/prep_chiba.py）のGTは合成セッション（例: chiba0532）の
区切りに紐づく。ライブ実走ラン（例: 2026-07-16_1723）を replay_attribution で
オフライン反復するには、そのランの turn_id にGTラベルを付けた gt json が
必要になる。本スクリプトは eval_speaker_gt.py と同じタイムライン方式
（支配的話者80%/カバレッジ30%）で移植する。

使い方:
    uv run python eval/transplant_gt.py eval/gt_chiba0532.json 2026-07-16_1723
    → eval/gt_2026-07-16_1723.json を生成
    → uv run python eval/replay_attribution.py --gt eval/gt_2026-07-16_1723.json \
          で API 不要のオフライン評価が回る（handoff §15.7）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(gt_path: str, session: str) -> None:
    gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
    root = ROOT / "transcripts"
    gt_turns = [json.loads(l)
                for l in open(root / f"{gt['session']}.turns.jsonl", encoding="utf-8")]
    turns = [json.loads(l)
             for l in open(root / f"{session}.turns.jsonl", encoding="utf-8")]

    tl: dict[str, list[tuple[int, int]]] = {}
    for g in gt_turns:
        c = gt["labels"].get(str(g["turn_id"])) or gt["labels"].get(g["turn_id"])
        if c in ("S1", "S2", "S3"):
            tl.setdefault(c, []).append((g["ms"], g["end_ms"]))

    labels: dict[str, str] = {}
    n_multi = n_uncov = 0
    for t in turns:
        s, e = t["ms"], t["end_ms"]
        dur = max(1, e - s)
        ovs = {c: sum(max(0, min(e, b) - max(s, a)) for a, b in ivs)
               for c, ivs in tl.items()}
        total = sum(ovs.values())
        if total < dur * 0.3:
            n_uncov += 1
            continue
        c, top = max(ovs.items(), key=lambda x: x[1])
        if top >= total * 0.8:
            labels[str(t["turn_id"])] = c
        else:
            labels[str(t["turn_id"])] = "MULTI"
            n_multi += 1

    out = {"session": session, "labels": labels,
           "speaker_names": gt.get("speaker_names", {}),
           "transplanted_from": gt["session"]}
    out_path = ROOT / "eval" / f"gt_{session}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    n_single = sum(1 for v in labels.values() if v in ("S1", "S2", "S3"))
    print(f"{out_path.relative_to(ROOT)}: 単独 {n_single} / 混在 {n_multi} / "
          f"範囲外 {n_uncov}（全 {len(turns)} 発話、GT元: {gt['session']}）")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])

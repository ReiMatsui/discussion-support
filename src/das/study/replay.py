"""人工の段階別入力で検査する。人の反応や介入効果は再現しない。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import Review, StudyConfig, StudyEngine, Turn, revision_of


def replay(data: dict) -> dict:
    if data.get("data_kind") != "synthetic":
        raise ValueError("This fixture runner accepts explicitly synthetic data only")
    engine = StudyEngine(StudyConfig.model_validate(data["config"]))
    results = []
    for step in data["steps"]:
        turns = [Turn.model_validate(t) for t in step["turns"]]
        candidate, reason = engine.evaluate(
            turns,
            Review.model_validate(step["review"]),
            revision=revision_of(turns),
            now=step["at"],
            new_turn_ids=set(step["new_turn_ids"]),
        )
        committed = bool(
            candidate and engine.commit(candidate, revision=revision_of(turns), now=step["at"])
        )
        results.append(
            {
                "reason": reason,
                "text": candidate.text if candidate else None,
                "committed": committed,
            }
        )
    return {
        "data_kind": "synthetic",
        "warning": "人工入力の処理検査。支援効果の結果ではない。",
        "steps": results,
        "request_counts": engine.counts,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("fixture")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = replay(json.loads(Path(args.fixture).read_text(encoding="utf-8")))
    Path(args.out).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "data_kind": result["data_kind"],
                "steps": len(result["steps"]),
                "request_counts": result["request_counts"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

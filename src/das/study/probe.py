"""人工会話1件で実LLMの接続を点検する。マイク・音声再生・参加者データは使わない。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from .engine import EXTRACTION_PROMPT, StudyConfig, StudyEngine, Turn, revision_of
from .live import request_review


def main():
    from das.asr.live._bootstrap import load_env

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", default="study/synthetic_replay.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    load_env()
    key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL_FAST", "gpt-5.4-mini")
    data = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    if data.get("data_kind") != "synthetic":
        raise SystemExit("人工入力と明示された確認用ファイルを指定してください")
    report = {
        "data_kind": "synthetic",
        "model": model,
        "purpose": "single transport probe, not accuracy or efficacy evaluation",
        "prompt_sha256": hashlib.sha256(EXTRACTION_PROMPT.encode()).hexdigest(),
    }
    start = time.monotonic()
    try:
        if not key or key.startswith("test-"):
            raise RuntimeError("OPENAI_API_KEYをローカルに設定してください")
        config = StudyConfig.model_validate(data["config"])
        step = data["steps"][1]
        turns = [Turn.model_validate(t) for t in step["turns"]]
        review = request_review(config, turns, set(step["new_turn_ids"]), key, model)
        candidate, reason = StudyEngine(config).evaluate(
            turns,
            review,
            revision=revision_of(turns),
            now=time.monotonic(),
            new_turn_ids=set(step["new_turn_ids"]),
        )
        report.update(
            status="completed",
            review=review.model_dump(),
            decision=reason,
            prepared_text=candidate.text if candidate else None,
        )
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        if getattr(exc, "code", None) is not None:
            report["http_status"] = exc.code
    report["elapsed_seconds"] = round(time.monotonic() - start, 3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "model": model,
                "elapsed_seconds": report["elapsed_seconds"],
            },
            ensure_ascii=False,
        )
    )
    if report["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

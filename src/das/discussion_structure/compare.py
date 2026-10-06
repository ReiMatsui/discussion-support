"""Replay a fixture suite with one shared USD 2 budget and aggregate calibration."""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .config import Config
from .judgement import OpenAITransport
from .metrics import comparison, quantiles
from .replay import run


def run_suite(fixtures: Path, out: Path, *, backend="openai_logprobs", config=None, transport=None):
    config = config or Config()
    if backend == "openai_logprobs":
        transport = transport or OpenAITransport(config)
    all_events, all_truth, completed = [], {}, []
    out.mkdir(parents=True, exist_ok=True)
    failure = None
    try:
        for path in sorted(fixtures.glob("*.json")):
            scenario = json.loads(path.read_text())
            tracker = run(
                path.with_suffix(".turns.jsonl"),
                scenario["agenda"],
                backend,
                scenario_path=path,
                out=out / path.stem,
                config=config,
                transport=transport,
            )
            completed.append(path.stem)
            for event in tracker.events:
                all_events.append(
                    event.model_copy(update={"turn_id": f"{path.stem}:{event.turn_id}"})
                )
            all_truth.update(
                {f"{path.stem}:{tid}": data for tid, data in scenario["judgements"].items()}
            )
    except (RuntimeError, ValueError, OSError) as exc:
        failure = str(exc)
    summary = {
        "backend": backend,
        "completed": completed,
        "failure": failure,
        "agreement_by_item": comparison(all_events, all_truth),
        "judgement_latency_ms": quantiles(
            [e.data["latency_ms"] for e in all_events if e.type == "judgement"]
        ),
        "api_cost_usd": transport.spent_usd if transport else 0,
        "api_reserved_usd": transport.reserved_usd if transport else 0,
        "budget_usd": config.budget_usd,
    }
    (out / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures", type=Path, default=Path("tests/fixtures/discussion_structure")
    )
    parser.add_argument("--out", type=Path, default=Path("data/discussion_structure/comparison"))
    parser.add_argument(
        "--backend", choices=["openai_logprobs", "scripted"], default="openai_logprobs"
    )
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    load_dotenv(override=False)
    try:
        summary = run_suite(
            args.fixtures, args.out, backend=args.backend, config=Config.load(args.config)
        )
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f"comparison: {exc}\n")
    if summary["failure"]:
        parser.exit(1, f"comparison: {summary['failure']}\n")


if __name__ == "__main__":
    main()

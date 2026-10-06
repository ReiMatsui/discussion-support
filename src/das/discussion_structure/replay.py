"""python -m das.discussion_structure.replay turns.jsonl --agenda ..."""

import argparse
import json
from pathlib import Path

from .config import Config
from .display import timeline
from .judgement import JevBackend, OpenAILogprobsBackend, OpenAITransport, ScriptedBackend
from .labels import OpenAILabels, ScriptedLabels
from .metrics import comparison, compute, projection, scenario_check
from .models import Turn
from .tracker import Tracker


def read_turns(path):
    """The legacy writer includes AI rows with null transcript timestamps."""
    turns, notices = [], []
    last_end = 0
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        speaker = str(row.get("speaker", ""))
        if "role" not in row and speaker in {"ファシリテーター", "パートナー", "__AGENT__"}:
            row["role"] = "ai"
        row["unsure"] = bool(row.get("unsure")) or speaker in {"未確定", "?"}
        row["backchannel"] = bool(row.get("backchannel", row.get("bc", False)))
        if row.get("ms") is None or row.get("end_ms") is None:
            if row.get("role", "human") == "human":
                raise ValueError(f"line {number}: human turn requires ms and end_ms")
            row["ms"] = row["end_ms"] = last_end
            notices.append(
                {
                    "turn_id": str(row.get("turn_id")),
                    "reason": "AI null timestamps; anchored to preceding end_ms and excluded from judgement",
                }
            )
        turn = Turn.model_validate(row)
        turns.append(turn)
        last_end = max(last_end, turn.end_ms)
    return sorted(turns, key=lambda t: t.end_ms), notices


def run(
    input_path,
    agenda,
    backend="scripted",
    scenario_path=None,
    out=None,
    config=None,
    transport=None,
):
    config = config or Config()
    scenario = json.loads(Path(scenario_path).read_text()) if scenario_path else None
    if backend == "scripted":
        if scenario is None:
            raise ValueError("scripted requires --scenario")
        judge, labels = ScriptedBackend(scenario["judgements"]), ScriptedLabels(scenario["labels"])
    else:
        # Check Jev key before trying the label-provider key.
        judge = JevBackend(config) if backend == "jev" else None
        transport = transport or OpenAITransport(config)
        judge = judge or OpenAILogprobsBackend(config, transport)
        labels = OpenAILabels(config, transport)
    starting_cost = transport.spent_usd if transport else 0
    turns, notices = read_turns(input_path)
    if len({t.turn_id for t in turns}) != len(turns):
        raise ValueError("turn_id must be unique")
    tracker = Tracker(agenda, judge, labels, config)
    for notice in notices:
        tracker.emit("input_normalization", **notice)
    checkpoints = {}
    destination = Path(out) if out else Path("data/discussion_structure") / Path(input_path).stem
    destination.mkdir(parents=True, exist_ok=True)
    try:
        for turn in turns:
            tracker.process(turn)
            checkpoints[turn.turn_id] = projection(tracker.snapshot())
        tracker.finish()
    finally:
        # Preserve completed evidence even when a backend fails mid-replay.
        snapshot = tracker.snapshot()
        metrics = compute(
            tracker.events,
            cost_usd=transport.spent_usd - starting_cost if transport else 0,
            reserved_usd=transport.reserved_usd if transport else 0,
            issue_chars=config.issue_chars,
            position_chars=config.position_chars,
        )
        if scenario:
            metrics["backend_comparison"] = comparison(tracker.events, scenario["judgements"])
        (destination / "events.jsonl").write_text(
            "".join(e.model_dump_json() + "\n" for e in tracker.events)
        )
        (destination / "snapshot.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2)
        )
        (destination / "timeline.txt").write_text(timeline(tracker.events))
        (destination / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        if scenario:
            (destination / "scenario_check.json").write_text(
                json.dumps(
                    scenario_check(scenario, snapshot, checkpoints), ensure_ascii=False, indent=2
                )
            )
    return tracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("turns", type=Path)
    parser.add_argument("--agenda", required=True)
    parser.add_argument(
        "--backend", choices=["scripted", "openai_logprobs", "jev"], default="scripted"
    )
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(override=False)
    try:
        run(
            args.turns, args.agenda, args.backend, args.scenario, args.out, Config.load(args.config)
        )
    except (RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f"replay: {exc}\n")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json

from click.testing import CliRunner

from das.asr.live import replay
from das.asr.live.replay import ReplayOptions, load_turns, replay_snapshot, run_replay


def _write_turns(path, rows):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_load_turns_excludes_agent_by_default(tmp_path):
    p = tmp_path / "sample.turns.jsonl"
    _write_turns(p, [
        {"turn_id": 1, "speaker": "参加者A", "text": "発話", "ms": 0, "end_ms": 1000},
        {"turn_id": 2, "speaker": "ファシリテーター", "text": "介入", "ms": None, "end_ms": None},
    ])

    turns = load_turns(p)

    assert [t["speaker"] for t in turns] == ["参加者A"]


def test_run_replay_fact_candidate_without_api():
    turns = [
        {"turn_id": 1, "speaker": "A", "text": "指標Xの計算式は分母を分子で割る感じです", "ms": 0, "end_ms": 1000},
    ]

    events = run_replay(turns, ReplayOptions(no_api=True, checks={"fact"}))

    assert events[0]["type"] == "fact_candidate"
    assert events[0]["turn_id"] == 1


def test_run_replay_fact_with_mock_checker():
    turns = [
        {"turn_id": 1, "speaker": "A", "text": "進め方の話です", "ms": 0, "end_ms": 1000},
        {"turn_id": 2, "speaker": "B", "text": "指標Xの計算式は分母を分子で割る感じです", "ms": 1000, "end_ms": 2000},
    ]

    def fake_fact(utts, _key, _model):
        assert utts[-1]["speaker"] == "B"
        return {
            "should_correct": True,
            "claim": "指標Xの計算式は分母を分子で割る",
            "correction": "指標Xは分子を分母で割ります。",
            "reason": "式が逆",
        }

    events = run_replay(
        turns,
        ReplayOptions(api_key="key", checks={"fact"}),
        check_fact=fake_fact,
    )

    assert events == [{
        "turn_id": 2,
        "ms": 1000,
        "type": "fact",
        "speaker": "B",
        "text": "指標Xの計算式は分母を分子で割る感じです",
        "detail": "指標Xは分子を分母で割ります。",
        "claim": "指標Xの計算式は分母を分子で割る",
        "reason": "式が逆",
    }]


def test_run_replay_drift_with_mock_checker():
    turns = [
        {"turn_id": i, "speaker": "A", "text": f"発話{i}", "ms": i * 1000, "end_ms": i * 1000 + 500}
        for i in range(1, 4)
    ]

    def fake_drift(_utts, topics, _key, _model):
        assert topics[0]["topic"] == "AI導入"
        return {"drift": True, "reason": "雑談"}

    events = run_replay(
        turns,
        ReplayOptions(api_key="key", topic="AI導入", checks={"drift"}),
        check_drift=fake_drift,
    )

    assert events[0]["type"] == "drift"
    assert events[0]["detail"] == "雑談"


def test_cli_no_api_outputs_fact_candidate(tmp_path):
    p = tmp_path / "sample.turns.jsonl"
    _write_turns(p, [
        {"turn_id": 1, "speaker": "A", "text": "指標Xの計算式は分母を分子で割る感じです", "ms": 0, "end_ms": 1000},
    ])

    result = CliRunner().invoke(replay.main, [str(p), "--no-api", "--checks", "fact"])

    assert result.exit_code == 0
    assert '"type": "fact_candidate"' in result.output


def test_replay_snapshot_for_ui():
    turns = [{"turn_id": 1, "speaker": "A", "text": "指標Xの計算式は分母を分子で割る", "ms": 0}]
    events = [{"turn_id": 1, "type": "fact_candidate", "detail": "候補"}]

    snap = replay_snapshot("x.turns.jsonl", turns, events,
                           ReplayOptions(no_api=True, checks={"fact"}))

    assert snap["source"] == "x.turns.jsonl"
    assert snap["turns"] == turns
    assert snap["events"] == events
    assert snap["checks"] == ["fact"]


def test_cli_help_has_serve_option():
    result = CliRunner().invoke(replay.main, ["--help"])

    assert result.exit_code == 0
    assert "--serve" in result.output
    assert "--port" in result.output

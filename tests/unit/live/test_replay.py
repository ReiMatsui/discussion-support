from __future__ import annotations

import json

from click.testing import CliRunner

from das.asr.live import replay
from das.asr.live._constants import _INVITE_WARMUP
from das.asr.live.replay import (
    ReplayOptions,
    default_interventions_path,
    intervention_review_items,
    load_interventions,
    load_turns,
    replay_snapshot,
    run_replay,
)


def _write_turns(path, rows):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path, rows):
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


def test_default_interventions_path_for_turns_file():
    assert str(default_interventions_path("sample.turns.jsonl")).endswith(
        "sample.interventions.jsonl"
    )


def test_load_interventions_returns_empty_for_missing_file(tmp_path):
    assert load_interventions(tmp_path / "missing.interventions.jsonl") == []


def test_intervention_review_items_pair_trigger_and_delivery(tmp_path):
    path = tmp_path / "sample.interventions.jsonl"
    _write_jsonl(path, [
        {
            "event_id": "int-0001",
            "type": "trigger",
            "reason": "drift",
            "detail": "雑談",
            "metadata": {
                "turn_count": 3,
                "topics": [{"topic": "AI導入", "speaker": "議題"}],
                "recent_utterances": [{"speaker": "A", "text": "話"}],
            },
        },
        {
            "type": "delivery",
            "trigger_event_id": "int-0001",
            "text": "本題に戻しましょう",
        },
    ])

    items = intervention_review_items(load_interventions(path))

    assert items == [{
        "event_id": "int-0001",
        "status": "delivered",
        "reason": "drift",
        "detail": "雑談",
        "created_at": None,
        "turn_count": 3,
        "recent_utterances": [{"speaker": "A", "text": "話"}],
        "topics": [{"topic": "AI導入", "speaker": "議題"}],
        "delivery_text": "本題に戻しましょう",
        "quality_flags": [],
        "trigger": load_interventions(path)[0],
        "delivery": load_interventions(path)[1],
    }]


def test_intervention_review_items_marks_missing_delivery():
    items = intervention_review_items([{
        "event_id": "int-0001",
        "type": "trigger",
        "reason": "invite",
        "detail": "参加者Bさんに声かけ",
        "metadata": {},
    }])

    assert items[0]["status"] == "missing_delivery"
    assert items[0]["delivery_text"] == ""
    assert items[0]["quality_flags"] == ["missing_delivery", "no_recent_context"]


def test_intervention_review_items_marks_orphan_delivery():
    items = intervention_review_items([{
        "type": "delivery",
        "trigger_event_id": None,
        "text": "本題に戻しましょう",
    }])

    assert items[0]["status"] == "orphan_delivery"
    assert items[0]["delivery_text"] == "本題に戻しましょう"
    assert items[0]["quality_flags"] == ["orphan_delivery", "no_recent_context"]


def test_intervention_review_items_flags_long_delivery_and_drift_without_topic():
    items = intervention_review_items([
        {
            "event_id": "int-0001",
            "type": "trigger",
            "reason": "drift",
            "detail": "雑談",
            "metadata": {"recent_utterances": [{"speaker": "A", "text": "話"}]},
        },
        {
            "type": "delivery",
            "trigger_event_id": "int-0001",
            "text": "長い介入です。" * 20,
        },
    ])

    assert items[0]["quality_flags"] == ["drift_without_topic", "long_delivery"]


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


def test_run_replay_fact_checks_target_with_recent_context():
    turns = [
        {"turn_id": 1, "speaker": "A", "text": "事物Aの高さは200メートルです", "ms": 0, "end_ms": 1000},
        {"turn_id": 2, "speaker": "B", "text": "国Bの首都は都市Aです", "ms": 1000, "end_ms": 2000},
    ]
    calls = []

    def fake_fact(utts, _key, _model):
        calls.append(utts)
        return {"should_correct": False}

    run_replay(
        turns,
        ReplayOptions(api_key="key", checks={"fact"}),
        check_fact=fake_fact,
    )

    assert calls == [
        [{"speaker": "A", "text": "事物Aの高さは200メートルです"}],
        [
            {"speaker": "A", "text": "事物Aの高さは200メートルです"},
            {"speaker": "B", "text": "国Bの首都は都市Aです"},
        ],
    ]


def test_run_replay_fact_retryable_error_is_visible():
    turns = [
        {"turn_id": 1, "speaker": "A", "text": "国Bの首都は都市Aです", "ms": 0, "end_ms": 1000},
    ]

    def fake_fact(_utts, _key, _model):
        return {"should_correct": False, "retryable_error": True}

    events = run_replay(
        turns,
        ReplayOptions(api_key="key", checks={"fact"}),
        check_fact=fake_fact,
    )

    assert events == [{
        "turn_id": 1,
        "ms": 0,
        "type": "fact_retryable_error",
        "speaker": "A",
        "text": "国Bの首都は都市Aです",
        "detail": "LLM事実判定の一時失敗",
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


def test_run_replay_invite_rejects_unknown_target():
    turns = []
    for i in range(_INVITE_WARMUP):
        speaker = "A" if i < _INVITE_WARMUP - 1 else "B"
        turns.append({
            "turn_id": i + 1,
            "speaker": speaker,
            "text": f"発話{i}",
            "ms": i * 1000,
            "end_ms": i * 1000 + 500,
        })

    def fake_invite(_participation, _utts, _key, _model):
        return {"invite": True, "speaker": "C", "reason": "静か"}

    events = run_replay(
        turns,
        ReplayOptions(api_key="key", checks={"invite"}),
        check_participation=fake_invite,
    )

    assert events[-1]["type"] == "invite_rejected"
    assert events[-1]["detail"] == "C"


def test_run_replay_invite_skips_balanced_text_without_timestamps():
    turns = [
        {"turn_id": i + 1, "speaker": "A" if i % 2 == 0 else "B", "text": "同じ長さ"}
        for i in range(_INVITE_WARMUP)
    ]
    calls = []

    def fake_invite(_participation, _utts, _key, _model):
        calls.append(1)
        return {"invite": True, "speaker": "B", "reason": "静か"}

    events = run_replay(
        turns,
        ReplayOptions(api_key="key", checks={"invite"}),
        check_participation=fake_invite,
    )

    assert calls == []
    assert events == []


def test_run_replay_invite_uses_char_share_without_timestamps():
    turns = [
        {
            "turn_id": i + 1,
            "speaker": "A" if i < _INVITE_WARMUP - 1 else "B",
            "text": "長い発言です" if i < _INVITE_WARMUP - 1 else "短",
        }
        for i in range(_INVITE_WARMUP)
    ]
    seen = []

    def fake_invite(participation, _utts, _key, _model):
        seen.extend(participation)
        return {"invite": True, "speaker": "B", "reason": "静か"}

    events = run_replay(
        turns,
        ReplayOptions(api_key="key", checks={"invite"}),
        check_participation=fake_invite,
    )

    target = next(p for p in seen if p["speaker"] == "B")
    assert target["time_share"] == 0.0
    assert target["participation_share_label"] == "発話文字数"
    assert target["participation_share"] < 0.25
    assert events[-1]["type"] == "invite"


def test_cli_no_api_outputs_fact_candidate(tmp_path):
    p = tmp_path / "sample.turns.jsonl"
    _write_turns(p, [
        {"turn_id": 1, "speaker": "A", "text": "指標Xの計算式は分母を分子で割る感じです", "ms": 0, "end_ms": 1000},
    ])

    result = CliRunner().invoke(replay.main, [str(p), "--no-api", "--checks", "fact"])

    assert result.exit_code == 0
    assert '"type": "fact_candidate"' in result.output


def test_cli_serve_auto_loads_sibling_interventions(tmp_path, monkeypatch):
    p = tmp_path / "sample.turns.jsonl"
    _write_turns(p, [{"turn_id": 1, "speaker": "A", "text": "進め方の話です"}])
    _write_jsonl(tmp_path / "sample.interventions.jsonl", [
        {"event_id": "int-0001", "type": "trigger", "reason": "count"},
    ])
    seen = {}

    def fake_serve(snapshot, *, port, open_browser):
        seen["snapshot"] = snapshot

    monkeypatch.setattr(replay, "serve_replay", fake_serve)

    result = CliRunner().invoke(replay.main, [str(p), "--no-api", "--serve"])

    assert result.exit_code == 0
    assert seen["snapshot"]["intervention_review"][0]["event_id"] == "int-0001"


def test_replay_snapshot_for_ui():
    turns = [{"turn_id": 1, "speaker": "A", "text": "指標Xの計算式は分母を分子で割る", "ms": 0}]
    events = [{"turn_id": 1, "type": "fact_candidate", "detail": "候補"}]
    interventions = [{"event_id": "int-0001", "type": "trigger", "reason": "fact"}]

    snap = replay_snapshot("x.turns.jsonl", turns, events,
                           ReplayOptions(no_api=True, checks={"fact"}), interventions)

    assert snap["source"] == "x.turns.jsonl"
    assert snap["turns"] == turns
    assert snap["events"] == events
    assert snap["checks"] == ["fact"]
    assert snap["interventions"] == interventions
    assert snap["intervention_review"][0]["event_id"] == "int-0001"


def test_cli_help_has_serve_option():
    result = CliRunner().invoke(replay.main, ["--help"])

    assert result.exit_code == 0
    assert "--serve" in result.output
    assert "--port" in result.output
    assert "--interventions" in result.output

from __future__ import annotations

import json

from click.testing import CliRunner

from das.asr.live import replay
from das.asr.live._constants import _INVITE_WARMUP
from das.asr.live.replay import (
    ReplayOptions,
    default_interventions_path,
    intervention_review_items,
    intervention_review_run_summary,
    intervention_review_summary,
    load_interventions,
    load_turns,
    replay_snapshot,
    run_replay,
    voice_call_diag_items,
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
            "created_at": "2026-01-01T09:00:00",
        },
        {
            "type": "delivery",
            "trigger_event_id": "int-0001",
            "created_at": "2026-01-01T09:00:03",
            "text": "本題に戻しましょう",
        },
    ])

    items = intervention_review_items(load_interventions(path))

    assert items == [{
        "event_id": "int-0001",
        "status": "delivered",
        "reason": "drift",
        "detail": "雑談",
        "created_at": "2026-01-01T09:00:00",
        "turn_count": 3,
        "recent_utterances": [{"speaker": "A", "text": "話"}],
        "topics": [{"topic": "AI導入", "speaker": "議題"}],
        "timing": {},
        "trigger_to_delivery_sec": 3.0,
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
    assert items[0]["timing"] == {}
    assert items[0]["trigger_to_delivery_sec"] is None


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


def test_intervention_review_summary_counts_status_reasons_and_flags():
    items = [
        {
            "status": "delivered",
            "reason": "drift",
            "quality_flags": ["long_delivery"],
            "timing": {"candidate_wait_sec": 1.0},
            "trigger_to_delivery_sec": 2.0,
        },
        {
            "status": "missing_delivery",
            "reason": "invite",
            "quality_flags": ["missing_delivery", "no_recent_context"],
            "timing": {"candidate_wait_sec": 3.0},
        },
        {
            "status": "delivered",
            "reason": "drift",
            "quality_flags": [],
            "trigger_to_delivery_sec": 4.0,
        },
    ]

    assert intervention_review_summary(items) == {
        "total": 3,
        "flagged_count": 2,
        "status_counts": {"delivered": 2, "missing_delivery": 1},
        "reason_counts": {"drift": 2, "invite": 1},
        "flag_counts": {
            "long_delivery": 1,
            "missing_delivery": 1,
            "no_recent_context": 1,
        },
        "manual_call_total": 0,
        "manual_call_delivered": 0,
        "manual_call_expired": 0,
        "avg_candidate_wait_sec": 2.0,
        "max_candidate_wait_sec": 3.0,
        "avg_trigger_to_delivery_sec": 3.0,
        "max_trigger_to_delivery_sec": 4.0,
    }


def _manual_call_trigger(event_id="int-0001", *, reason="manual_call",
                         source="ui", request="ここまで整理して", wait=2.5,
                         outcome="selected"):
    return {
        "event_id": event_id,
        "type": "trigger",
        "reason": reason,
        "detail": request or "直近の議論整理",
        "metadata": {
            "recent_utterances": [{"speaker": "A", "text": "話"}],
            "timing": {"kind": "manual", "source": source, "request": request,
                       "candidate_wait_sec": wait, "outcome": outcome},
        },
        "created_at": "2026-01-01T09:00:00",
    }


def test_intervention_review_items_manual_call_delivered():
    """manual_call は通常の trigger と同様に delivery とペアリングされる."""
    items = intervention_review_items([
        _manual_call_trigger(source="voice"),
        {"type": "delivery", "trigger_event_id": "int-0001",
         "created_at": "2026-01-01T09:00:02", "text": "論点は2つです。"},
    ])

    assert len(items) == 1
    assert items[0]["status"] == "delivered"
    assert items[0]["reason"] == "manual_call"
    assert items[0]["manual"] == {
        "source": "voice",
        "request": "ここまで整理して",
        "outcome": "selected",
        "candidate_wait_sec": 2.5,
    }
    assert items[0]["quality_flags"] == []
    assert items[0]["trigger_to_delivery_sec"] == 2.0


def test_intervention_review_items_manual_call_expired():
    """manual_call_expired は delivery なしの不発として専用 status になる."""
    items = intervention_review_items([
        _manual_call_trigger(reason="manual_call_expired",
                             wait=31.0, outcome="expired"),
    ])

    assert len(items) == 1
    assert items[0]["status"] == "expired_manual_call"
    assert items[0]["manual"]["outcome"] == "expired"
    assert items[0]["manual"]["candidate_wait_sec"] == 31.0
    # missing_delivery ではなく「呼び出し不発」フラグで要確認に入る
    assert "manual_call_expired" in items[0]["quality_flags"]
    assert "missing_delivery" not in items[0]["quality_flags"]


def test_intervention_review_items_non_manual_have_no_manual_field():
    """他 kind の item 形式は不変（manual フィールドを持たない）."""
    items = intervention_review_items([{
        "event_id": "int-0001", "type": "trigger", "reason": "drift",
        "detail": "雑談", "metadata": {},
    }])

    assert "manual" not in items[0]


def test_intervention_review_summary_counts_manual_calls():
    """summary に manual call の成功/不発の指標が追加される."""
    items = intervention_review_items([
        _manual_call_trigger("int-0001"),
        {"type": "delivery", "trigger_event_id": "int-0001",
         "created_at": "2026-01-01T09:00:02", "text": "論点は2つです。"},
        _manual_call_trigger("int-0002", reason="manual_call_expired",
                             wait=31.0, outcome="expired"),
        {"event_id": "int-0003", "type": "trigger", "reason": "drift",
         "detail": "雑談", "metadata": {}},
    ])

    summary = intervention_review_summary(items)

    assert summary["manual_call_total"] == 2
    assert summary["manual_call_delivered"] == 1
    assert summary["manual_call_expired"] == 1
    # 既存指標は不変に動く
    assert summary["status_counts"]["delivered"] == 1
    assert summary["reason_counts"]["drift"] == 1


def test_voice_call_diag_items_extracts_detected_and_ignored():
    diags = voice_call_diag_items([
        {"type": "voice_call_diag", "time": "09:00:01", "detected": True,
         "text": "ファシリテーター、ここまで整理して", "request": "ここまで整理して",
         "ignored_reason": None},
        {"type": "voice_call_diag", "time": "09:00:05", "detected": False,
         "text": "AIさんですね", "request": None, "ignored_reason": "no_request"},
        {"event_id": "int-0001", "type": "trigger", "reason": "drift"},
    ])

    assert [d["detected"] for d in diags] == [True, False]
    assert diags[0]["request"] == "ここまで整理して"
    assert diags[1]["ignored_reason"] == "no_request"


def test_replay_snapshot_includes_voice_diag_and_manual_metrics():
    """snapshot に voice_call_diag と summary の追加指標が入り、review を汚さない."""
    interventions = [
        _manual_call_trigger("int-0001"),
        {"type": "delivery", "trigger_event_id": "int-0001",
         "created_at": "2026-01-01T09:00:02", "text": "論点は2つです。"},
        _manual_call_trigger("int-0002", reason="manual_call_expired",
                             wait=31.0, outcome="expired"),
        {"type": "voice_call_diag", "time": "09:00:01", "detected": True,
         "text": "ファシリテーター、ここまで整理して",
         "request": "ここまで整理して", "ignored_reason": None},
        {"type": "voice_call_diag", "time": "09:00:05", "detected": False,
         "text": "AIについて話しましょう", "request": None,
         "ignored_reason": "meta_topic"},
    ]
    turns = [{"turn_id": 1, "speaker": "A", "text": "話", "ms": 0}]

    snap = replay_snapshot("x.turns.jsonl", turns, [],
                           ReplayOptions(no_api=True), interventions)

    assert len(snap["voice_call_diag"]) == 2
    summary = snap["intervention_review_summary"]
    assert summary["manual_call_total"] == 2
    assert summary["manual_call_delivered"] == 1
    assert summary["manual_call_expired"] == 1
    assert summary["voice_call_detected"] == 1
    assert summary["voice_call_ignored"] == 1
    # diag イベントは trigger/delivery のペアリングを汚さない
    statuses = {i["status"] for i in snap["intervention_review"]}
    assert statuses == {"delivered", "expired_manual_call"}


def test_intervention_review_run_summary_adds_normalized_metrics():
    items = [
        {"status": "delivered", "reason": "drift", "quality_flags": []},
        {"status": "missing_delivery", "reason": "invite", "quality_flags": ["missing_delivery"]},
    ]

    summary = intervention_review_run_summary(items, turn_count=20)

    assert summary["turn_count"] == 20
    assert summary["interventions_per_10_turns"] == 1.0
    assert summary["delivered_per_10_turns"] == 0.5
    assert summary["flagged_per_10_turns"] == 0.5


def _classify_all_facts(_utts, _key, _model):
    return {"factual_claim": True, "facilitator_request": ""}


def test_run_replay_fact_candidate_without_api():
    turns = [
        {"turn_id": 1, "speaker": "A", "text": "指標Xの計算式は分母を分子で割る感じです", "ms": 0, "end_ms": 1000},
    ]

    events = run_replay(turns, ReplayOptions(no_api=True, checks={"fact"}))

    assert events[0]["type"] == "fact_candidate"
    assert events[0]["turn_id"] == 1


def test_run_replay_fact_skips_when_classify_rejects():
    """LLM分類が fact候補でないと判定した発話（創作表現・評価など）は事実判定に回さない."""
    turns = [
        {"turn_id": 1, "speaker": "A", "text": "その弾丸は奴をめがけて襲いかかる。", "ms": 0},
    ]
    fact_calls = []

    def fake_fact(utts, _key, _model):
        fact_calls.append(utts)
        return {"should_correct": True, "confidence": "high",
                "claim": "x", "correction": "y", "reason": "z"}

    def fake_classify(_utts, _key, _model):
        return {"factual_claim": False, "facilitator_request": ""}

    events = run_replay(
        turns,
        ReplayOptions(api_key="key", checks={"fact"}),
        check_fact=fake_fact,
        classify=fake_classify,
    )

    assert events == []
    assert fact_calls == []


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
        classify=_classify_all_facts,
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
        classify=_classify_all_facts,
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
        classify=_classify_all_facts,
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


def test_cli_writes_intervention_review_jsonl(tmp_path):
    p = tmp_path / "sample.turns.jsonl"
    review_out = tmp_path / "review.jsonl"
    _write_turns(p, [{"turn_id": 1, "speaker": "A", "text": "進め方の話です"}])
    _write_jsonl(tmp_path / "sample.interventions.jsonl", [
        {"event_id": "int-0001", "type": "trigger", "reason": "count"},
    ])

    result = CliRunner().invoke(
        replay.main,
        [str(p), "--no-api", "--review-out", str(review_out)],
    )

    assert result.exit_code == 0
    rows = [json.loads(line) for line in review_out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["event_id"] == "int-0001"
    assert rows[0]["quality_flags"] == ["missing_delivery", "no_recent_context"]


def test_cli_writes_intervention_review_summary_json(tmp_path):
    p = tmp_path / "sample.turns.jsonl"
    summary_out = tmp_path / "summary.json"
    _write_turns(p, [{"turn_id": 1, "speaker": "A", "text": "進め方の話です"}])
    _write_jsonl(tmp_path / "sample.interventions.jsonl", [
        {"event_id": "int-0001", "type": "trigger", "reason": "count"},
    ])

    result = CliRunner().invoke(
        replay.main,
        [str(p), "--no-api", "--review-summary-out", str(summary_out)],
    )

    assert result.exit_code == 0
    data = json.loads(summary_out.read_text(encoding="utf-8"))
    assert data["total"] == 1
    assert data["flag_counts"] == {"missing_delivery": 1, "no_recent_context": 1}


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
    assert snap["intervention_review_summary"]["total"] == 1
    assert snap["intervention_review_summary"]["turn_count"] == 1
    assert snap["intervention_review_summary"]["interventions_per_10_turns"] == 10.0


def test_cli_help_has_serve_option():
    result = CliRunner().invoke(replay.main, ["--help"])

    assert result.exit_code == 0
    assert "--serve" in result.output
    assert "--port" in result.output
    assert "--interventions" in result.output
    assert "--review-out" in result.output
    assert "--review-summary-out" in result.output

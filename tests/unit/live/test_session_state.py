"""SessionState の脱線検出シード（Fix 8）のユニットテスト."""
from __future__ import annotations

import datetime

from das.asr.live._session_state import SessionState


def _make_state() -> SessionState:
    return SessionState(
        args=object(),
        started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md",
        html_path="/tmp/o.html",
        diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns",
        wav_path="/tmp/o.wav",
    )


def test_seed_topic_adds_when_empty():
    s = _make_state()
    s.seed_topic("AIツール導入の是非")
    assert [t["topic"] for t in s.topics] == ["AIツール導入の是非"]
    assert s.topics[0]["speaker"] == "議題"


def test_delivery_event_includes_timing(tmp_path):
    """delivery イベントに speak_start_latency_ms などの timing を残せる（Phase4観測）."""
    import json
    s = _make_state()
    s.interventions_path = str(tmp_path / "o.interventions.jsonl")
    s.add_facilitator_delivery_event("本題に戻しましょう",
                                     timing={"speak_start_latency_ms": 420.0})
    with open(s.interventions_path, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f]
    assert lines[-1]["type"] == "delivery"
    assert lines[-1]["timing"]["speak_start_latency_ms"] == 420.0


def test_delivery_event_omits_timing_when_absent(tmp_path):
    """timing 未指定なら delivery イベントに timing キーを付けない（既存互換）."""
    import json
    s = _make_state()
    s.interventions_path = str(tmp_path / "o.interventions.jsonl")
    s.add_facilitator_delivery_event("本題に戻しましょう")
    with open(s.interventions_path, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f]
    assert "timing" not in lines[-1]


def test_seed_topic_noop_for_empty_input():
    s = _make_state()
    s.seed_topic("")
    s.seed_topic(None)
    assert s.topics == []


def test_seed_topic_does_not_override_existing():
    s = _make_state()
    s.topics.append({"topic": "既存論点", "speaker": "話者1"})
    s.seed_topic("議題テーマ")
    assert [t["topic"] for t in s.topics] == ["既存論点"]

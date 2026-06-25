"""介入イベントモデルのテスト."""
from __future__ import annotations

from das.asr.live.facilitation import (
    FacilitationEvent,
    FacilitationEventType,
    new_intervention_id,
)


def test_event_serializes_to_json_compatible_dict():
    intervention_id = new_intervention_id()
    event = FacilitationEvent.create(
        FacilitationEventType.RESPONSE_REQUESTED,
        intervention_id,
        trigger_kind="invite",
        trigger_reason="田中さんに声かけ",
        input_utterances=({"speaker": "佐藤", "text": "どうでしょう"},),
        details={"invite_target": "田中"},
    )

    data = event.to_dict()

    assert intervention_id.startswith("int_")
    assert data["schema_version"] == 1
    assert data["event_type"] == "response_requested"
    assert data["input_utterances"] == [
        {"speaker": "佐藤", "text": "どうでしょう"},
    ]


def test_event_ids_are_unique():
    first = FacilitationEvent.create(
        FacilitationEventType.TRIGGER_REQUESTED,
        new_intervention_id(),
    )
    second = FacilitationEvent.create(
        FacilitationEventType.TRIGGER_REQUESTED,
        new_intervention_id(),
    )

    assert first.event_id != second.event_id
    assert first.intervention_id != second.intervention_id

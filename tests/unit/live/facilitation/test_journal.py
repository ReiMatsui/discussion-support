"""介入イベントJSONLジャーナルのテスト."""
from __future__ import annotations

import json

from das.asr.live.facilitation import (
    FacilitationEvent,
    FacilitationEventType,
    FacilitationJournal,
)


def test_journal_appends_one_json_object_per_line(tmp_path):
    path = tmp_path / "meeting.interventions.jsonl"
    journal = FacilitationJournal(path)
    event = FacilitationEvent.create(
        FacilitationEventType.NOOP,
        "int_test",
        trigger_kind="count",
        playback=False,
    )

    journal.append(event)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["event_type"] == "noop"


def test_journal_can_switch_path_for_new_meeting(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    journal = FacilitationJournal(first)
    journal.append(FacilitationEvent.create(
        FacilitationEventType.TRIGGER_REQUESTED,
        "int_first",
    ))

    journal.set_path(second)
    journal.append(FacilitationEvent.create(
        FacilitationEventType.TRIGGER_REQUESTED,
        "int_second",
    ))

    assert "int_first" in first.read_text(encoding="utf-8")
    assert "int_second" in second.read_text(encoding="utf-8")


def test_journal_can_append_to_captured_meeting_path(tmp_path):
    old_path = tmp_path / "old.jsonl"
    new_path = tmp_path / "new.jsonl"
    journal = FacilitationJournal(new_path)

    journal.append(FacilitationEvent.create(
        FacilitationEventType.RESPONSE_COMPLETED,
        "int_old",
    ), path=old_path)

    assert "int_old" in old_path.read_text(encoding="utf-8")
    assert not new_path.exists()

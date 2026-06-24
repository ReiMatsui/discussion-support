from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings


def current_timestamp_ms() -> int:
    return time.time_ns() // 1_000_000


@dataclass(slots=True)
class SpeakerStatsEntry:
    speaker_key: str
    participant_id: str | None
    display_name: str | None
    speaker_label: str | None
    utterance_count: int = 0
    text_chars: int = 0
    estimated_speech_ms: int = 0
    last_spoke_at_ms: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "speaker_key": self.speaker_key,
            "participant_id": self.participant_id,
            "display_name": self.display_name,
            "speaker_label": self.speaker_label,
            "utterance_count": self.utterance_count,
            "text_chars": self.text_chars,
            "estimated_speech_ms": self.estimated_speech_ms,
            "last_spoke_at_ms": self.last_spoke_at_ms,
        }


class SpeakerStats:
    def __init__(self, *, settings: Settings) -> None:
        self._settings = settings
        self._entries: dict[str, SpeakerStatsEntry] = {}

    def update_from_transcript(
        self,
        transcript: dict[str, Any],
    ) -> dict[str, Any] | None:
        if transcript.get("type") != "transcript.final":
            return None
        if transcript.get("is_final") is not True:
            return None

        text = transcript.get("text")
        if not isinstance(text, str) or not text.strip():
            return None

        participant_id = _optional_string(transcript.get("participant_id"))
        display_name = _optional_string(transcript.get("display_name"))
        speaker_label = _optional_string(transcript.get("speaker_label"))
        speaker_key = _speaker_key(
            participant_id=participant_id,
            speaker_label=speaker_label,
        )
        entry = self._entries.get(speaker_key)
        if entry is None:
            entry = SpeakerStatsEntry(
                speaker_key=speaker_key,
                participant_id=participant_id,
                display_name=display_name,
                speaker_label=speaker_label,
            )
            self._entries[speaker_key] = entry
        else:
            entry.participant_id = participant_id or entry.participant_id
            entry.display_name = display_name or entry.display_name
            entry.speaker_label = speaker_label or entry.speaker_label

        entry.utterance_count += 1
        entry.text_chars += len(text.strip())
        entry.estimated_speech_ms += _duration_ms(
            start_ms=transcript.get("start_ms"),
            end_ms=transcript.get("end_ms"),
        )
        timestamp = transcript.get("server_timestamp_ms")
        entry.last_spoke_at_ms = timestamp if isinstance(timestamp, int) else None

        return self.as_event()

    def as_event(self) -> dict[str, Any]:
        return {
            "type": "speaker.stats.updated",
            "meeting_id": self._settings.meeting_id,
            "stats": [
                entry.as_dict()
                for entry in sorted(
                    self._entries.values(),
                    key=lambda item: (
                        item.display_name or "",
                        item.speaker_label or "",
                        item.speaker_key,
                    ),
                )
            ],
            "server_timestamp_ms": current_timestamp_ms(),
        }


def _speaker_key(*, participant_id: str | None, speaker_label: str | None) -> str:
    if participant_id is not None:
        return f"participant:{participant_id}"
    if speaker_label is not None:
        return f"speaker:{speaker_label}"
    return "unknown"


def _duration_ms(*, start_ms: object, end_ms: object) -> int:
    if not isinstance(start_ms, int) or not isinstance(end_ms, int):
        return 0
    return max(0, end_ms - start_ms)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None

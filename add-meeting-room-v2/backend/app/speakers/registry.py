from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from app.audio.output import Pcm16AudioPlayer
from app.core.config import Settings
from app.soniox.client import TranscriptEvent
from app.voice_agent.realtime2_client import Realtime2VoiceAgentClient

ParticipantRole = Literal["human", "agent"]
SpeakerStatus = Literal["mapped", "unassigned"]


def current_timestamp_ms() -> int:
    return time.time_ns() // 1_000_000


@dataclass(frozen=True, slots=True)
class Participant:
    participant_id: str
    display_name: str
    role: ParticipantRole

    def as_dict(self) -> dict[str, str]:
        return {
            "participant_id": self.participant_id,
            "display_name": self.display_name,
            "role": self.role,
        }


@dataclass(frozen=True, slots=True)
class SpeakerAssignment:
    participant_id: str
    display_name: str
    role: ParticipantRole
    source: str

    def as_dict(self) -> dict[str, str]:
        return {
            "participant_id": self.participant_id,
            "display_name": self.display_name,
            "role": self.role,
            "source": self.source,
        }


@dataclass(slots=True)
class IntroSession:
    participant_id: str
    known_labels: set[str]
    expires_at_ms: int
    candidates: set[str] = field(default_factory=set)
    expired_notified: bool = False


class SpeakerRegistry:
    def __init__(self, *, settings: Settings) -> None:
        self._settings = settings
        self._participants: dict[str, Participant] = {}
        self._speaker_map: dict[str, SpeakerAssignment] = {}
        self._observed_labels: set[str] = set()
        self._unassigned_seen: set[str] = set()
        self._active_intro: IntroSession | None = None
        self._lock = asyncio.Lock()
        self._realtime_client = Realtime2VoiceAgentClient(settings=settings)

    async def handle_command(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        command_type = payload.get("type")
        if command_type == "participant.list.update":
            return await self._update_participants(payload)
        if command_type == "speaker.intro.start":
            return await self._start_intro(payload)
        if command_type == "speaker.intro.cancel":
            return await self._cancel_intro()
        if command_type == "speaker.bind":
            return await self._bind_from_command(payload)
        return []

    async def enrich_transcript(
        self,
        event: TranscriptEvent,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        extra_events: list[dict[str, Any]] = []
        async with self._lock:
            self._clear_expired_intro(extra_events)
            if event.speaker_label is not None:
                self._observed_labels.add(event.speaker_label)

            assignment = self._assignment_for(event.speaker_label)
            if assignment is None:
                candidate = self._maybe_collect_intro_candidate(event.speaker_label)
                if candidate is not None:
                    extra_events.append(candidate)
                elif (
                    event.speaker_label is not None
                    and event.speaker_label not in self._unassigned_seen
                ):
                    self._unassigned_seen.add(event.speaker_label)
                    extra_events.append(self._unassigned_event(event.speaker_label))
                assignment = self._assignment_for(event.speaker_label)

            speaker_status: SpeakerStatus = "mapped" if assignment else "unassigned"
            enriched = {
                **event.as_dict(),
                "participant_id": assignment.participant_id if assignment else None,
                "display_name": assignment.display_name if assignment else None,
                "speaker_status": speaker_status,
            }
        return enriched, extra_events

    async def state_events(self) -> list[dict[str, Any]]:
        async with self._lock:
            events = [
                self._participant_list_event(),
                self._speaker_map_event(),
            ]
            events.extend(
                self._unassigned_event(speaker_label)
                for speaker_label in sorted(
                    self._observed_labels.difference(self._speaker_map)
                )
            )
            return events

    async def is_intro_active(self) -> bool:
        async with self._lock:
            extra_events: list[dict[str, Any]] = []
            self._clear_expired_intro(extra_events)
            return self._active_intro is not None

    async def is_agent_speaker(self, speaker_label: str | None) -> bool:
        if speaker_label is None:
            return False
        async with self._lock:
            assignment = self._speaker_map.get(speaker_label)
            return assignment is not None and assignment.role == "agent"

    async def is_human_speaker(self, speaker_label: str | None) -> bool:
        if speaker_label is None:
            return False
        async with self._lock:
            assignment = self._speaker_map.get(speaker_label)
            return assignment is not None and assignment.role == "human"

    async def _update_participants(
        self,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        raw_participants = payload.get("participants")
        if not isinstance(raw_participants, list):
            return []

        participants: dict[str, Participant] = {}
        for item in raw_participants:
            if not isinstance(item, dict):
                continue
            participant_id = _clean_string(item.get("participant_id"))
            display_name = _clean_string(item.get("display_name"))
            role = item.get("role")
            if role not in {"human", "agent"}:
                role = "human"
            if participant_id is None or display_name is None:
                continue
            participants[participant_id] = Participant(
                participant_id=participant_id,
                display_name=display_name,
                role=role,
            )

        async with self._lock:
            self._participants = participants
            self._speaker_map = {
                label: assignment
                for label, assignment in self._speaker_map.items()
                if assignment.participant_id in participants
            }
            return [self._participant_list_event(), self._speaker_map_event()]

    async def _start_intro(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        participant_id = _clean_string(payload.get("participant_id"))
        if participant_id is None:
            return []

        async with self._lock:
            participant = self._participants.get(participant_id)
            if participant is None:
                return []
            expires_at_ms = (
                current_timestamp_ms() + self._settings.speaker_intro_window_ms
            )
            self._active_intro = IntroSession(
                participant_id=participant_id,
                known_labels=set(self._observed_labels),
                expires_at_ms=expires_at_ms,
            )
            started = {
                "type": "speaker.intro.started",
                "meeting_id": self._settings.meeting_id,
                "participant_id": participant.participant_id,
                "display_name": participant.display_name,
                "role": participant.role,
                "known_speaker_labels": sorted(self._observed_labels),
                "expires_at_ms": expires_at_ms,
                "server_timestamp_ms": current_timestamp_ms(),
            }

        if participant.role == "agent":
            asyncio.create_task(self._play_agent_intro(participant))
        return [started]

    async def _cancel_intro(self) -> list[dict[str, Any]]:
        async with self._lock:
            if self._active_intro is None:
                return []
            participant_id = self._active_intro.participant_id
            self._active_intro = None
            return [
                {
                    "type": "speaker.intro.cancelled",
                    "meeting_id": self._settings.meeting_id,
                    "participant_id": participant_id,
                    "server_timestamp_ms": current_timestamp_ms(),
                }
            ]

    async def _bind_from_command(
        self,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        speaker_label = _clean_string(payload.get("speaker_label"))
        participant_id = _clean_string(payload.get("participant_id"))
        if speaker_label is None or participant_id is None:
            return []

        async with self._lock:
            assignment = self._bind(
                speaker_label,
                participant_id,
                source="self_intro"
                if self._active_intro
                and self._active_intro.participant_id == participant_id
                else "manual",
            )
            if assignment is None:
                return []
            events: list[dict[str, Any]] = [self._speaker_map_event()]
            if (
                self._active_intro is not None
                and self._active_intro.participant_id == participant_id
            ):
                events.append(
                    {
                        "type": "speaker.intro.completed",
                        "meeting_id": self._settings.meeting_id,
                        "participant_id": participant_id,
                        "speaker_label": speaker_label,
                        "server_timestamp_ms": current_timestamp_ms(),
                    }
                )
                self._active_intro = None
            return events

    def _maybe_collect_intro_candidate(
        self,
        speaker_label: str | None,
    ) -> dict[str, Any] | None:
        if speaker_label is None or self._active_intro is None:
            return None
        if current_timestamp_ms() >= self._active_intro.expires_at_ms:
            return None
        if speaker_label in self._active_intro.known_labels:
            return None
        if speaker_label in self._speaker_map:
            return None
        if speaker_label in self._active_intro.candidates:
            return None
        self._active_intro.candidates.add(speaker_label)
        participant = self._participants.get(self._active_intro.participant_id)
        return {
            "type": "speaker.intro.candidate_detected",
            "meeting_id": self._settings.meeting_id,
            "participant_id": self._active_intro.participant_id,
            "display_name": participant.display_name if participant else None,
            "speaker_label": speaker_label,
            "candidates": sorted(self._active_intro.candidates),
            "server_timestamp_ms": current_timestamp_ms(),
        }

    def _bind(
        self,
        speaker_label: str,
        participant_id: str,
        *,
        source: str,
    ) -> SpeakerAssignment | None:
        participant = self._participants.get(participant_id)
        if participant is None:
            return None
        assignment = SpeakerAssignment(
            participant_id=participant.participant_id,
            display_name=participant.display_name,
            role=participant.role,
            source=source,
        )
        self._speaker_map[speaker_label] = assignment
        self._unassigned_seen.discard(speaker_label)
        self._observed_labels.add(speaker_label)
        return assignment

    def _assignment_for(self, speaker_label: str | None) -> SpeakerAssignment | None:
        if speaker_label is None:
            return None
        return self._speaker_map.get(speaker_label)

    def _participant_list_event(self) -> dict[str, Any]:
        return {
            "type": "participant.list.updated",
            "meeting_id": self._settings.meeting_id,
            "participants": [
                participant.as_dict()
                for participant in sorted(
                    self._participants.values(),
                    key=lambda participant: participant.participant_id,
                )
            ],
            "server_timestamp_ms": current_timestamp_ms(),
        }

    def _speaker_map_event(self) -> dict[str, Any]:
        return {
            "type": "speaker.map.updated",
            "meeting_id": self._settings.meeting_id,
            "speaker_map": {
                label: assignment.as_dict()
                for label, assignment in sorted(self._speaker_map.items())
            },
            "server_timestamp_ms": current_timestamp_ms(),
        }

    def _unassigned_event(self, speaker_label: str) -> dict[str, Any]:
        return {
            "type": "speaker.unassigned_detected",
            "meeting_id": self._settings.meeting_id,
            "speaker_label": speaker_label,
            "server_timestamp_ms": current_timestamp_ms(),
        }

    def _clear_expired_intro(self, events: list[dict[str, Any]]) -> None:
        if self._active_intro is None:
            return
        if current_timestamp_ms() < self._active_intro.expires_at_ms:
            return
        if self._active_intro.expired_notified:
            return
        participant_id = self._active_intro.participant_id
        candidates = sorted(self._active_intro.candidates)
        self._active_intro.expired_notified = True
        events.append(
            {
                "type": "speaker.intro.expired",
                "meeting_id": self._settings.meeting_id,
                "participant_id": participant_id,
                "candidates": candidates,
                "server_timestamp_ms": current_timestamp_ms(),
            }
        )

    async def _play_agent_intro(self, participant: Participant) -> None:
        prompt = (
            f"あなたは{participant.display_name}です。"
            f"「{participant.display_name}です。よろしくお願いします。」"
            "と一文だけ、聞き取りやすく自己紹介してください。"
        )
        try:
            with Pcm16AudioPlayer(
                sample_rate=self._settings.openai_realtime_output_sample_rate
            ) as player:
                await self._realtime_client.speak_text(
                    prompt,
                    on_audio_delta=player.write,
                )
        except Exception as exc:
            print(f"[speaker_registry] AI intro failed: {exc}", file=sys.stderr)


def _clean_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None

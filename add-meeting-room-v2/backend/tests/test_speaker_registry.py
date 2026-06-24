import asyncio

from app.core.config import Settings
from app.soniox.client import TranscriptEvent
from app.speakers.registry import SpeakerRegistry


def transcript_event(*, speaker_label: str) -> TranscriptEvent:
    return TranscriptEvent(
        type="transcript.final",
        meeting_id="meeting_001",
        segment_id="segment_001",
        speaker_label=speaker_label,
        text="hello",
        is_final=True,
        endpoint_detected=True,
        start_ms=0,
        end_ms=1000,
        server_timestamp_ms=1000,
    )


def test_manual_bind_assigns_an_observed_speaker_to_existing_participant() -> None:
    async def scenario() -> None:
        registry = SpeakerRegistry(settings=Settings())
        await registry.handle_command(
            {
                "type": "participant.list.update",
                "participants": [
                    {
                        "participant_id": "p_001",
                        "display_name": "田中",
                        "role": "human",
                    }
                ],
            }
        )

        enriched, events = await registry.enrich_transcript(
            transcript_event(speaker_label="1")
        )

        assert enriched["speaker_status"] == "unassigned"
        assert [event["type"] for event in events] == [
            "speaker.unassigned_detected"
        ]

        bind_events = await registry.handle_command(
            {
                "type": "speaker.bind",
                "speaker_label": "1",
                "participant_id": "p_001",
            }
        )

        assert bind_events[0]["type"] == "speaker.map.updated"
        assert bind_events[0]["speaker_map"]["1"] == {
            "participant_id": "p_001",
            "display_name": "田中",
            "role": "human",
            "source": "manual",
        }

    asyncio.run(scenario())


def test_state_events_restore_observed_unassigned_speakers() -> None:
    async def scenario() -> None:
        registry = SpeakerRegistry(settings=Settings())
        await registry.enrich_transcript(transcript_event(speaker_label="2"))

        state_events = await registry.state_events()

        assert [event["type"] for event in state_events] == [
            "participant.list.updated",
            "speaker.map.updated",
            "speaker.unassigned_detected",
        ]
        assert state_events[-1]["speaker_label"] == "2"

    asyncio.run(scenario())

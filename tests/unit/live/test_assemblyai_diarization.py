"""AssemblyAI streaming diarization provider の単体テスト."""
from __future__ import annotations

import json

from das.asr.live._assemblyai_diarization import AssemblyAIStreamingDiarizationProvider


def test_parse_turn_groups_final_words_by_speaker() -> None:
    provider = AssemblyAIStreamingDiarizationProvider("k")
    msg = {
        "type": "Turn",
        "turn_order": 1,
        "end_of_turn": True,
        "turn_is_formatted": True,
        "speaker_label": "A",
        "words": [
            {"text": "はい", "start": 1000, "end": 1300,
             "speaker": "A", "word_is_final": True},
            {"text": "そうです", "start": 1400, "end": 1900,
             "speaker": "A", "word_is_final": True},
            {"text": "違います", "start": 2200, "end": 2600,
             "speaker": "B", "word_is_final": True},
        ],
    }

    events = provider._parse_message(json.dumps(msg))

    assert [(e.start_ms, e.end_ms, e.speaker, e.source) for e in events] == [
        (1000, 1900, "A", "assemblyai"),
        (2200, 2600, "B", "assemblyai"),
    ]


def test_parse_turn_ignores_unknown_speakers() -> None:
    provider = AssemblyAIStreamingDiarizationProvider("k")
    msg = {
        "type": "Turn",
        "turn_order": 1,
        "end_of_turn": True,
        "speaker_label": "UNKNOWN",
        "words": [
            {"text": "はい", "start": 1000, "end": 1100,
             "speaker": "UNKNOWN", "word_is_final": True},
        ],
    }

    assert provider._parse_message(json.dumps(msg)) == []


def test_parse_partial_turn_exposes_active_event_without_draining() -> None:
    provider = AssemblyAIStreamingDiarizationProvider("k")
    msg = {
        "type": "Turn",
        "turn_order": 2,
        "end_of_turn": False,
        "speaker_label": "A",
        "words": [
            {"text": "途中", "start": 3000, "end": 3600,
             "speaker": "A", "word_is_final": True},
        ],
    }

    assert provider._parse_message(json.dumps(msg)) == []
    active = provider.active_events()

    assert len(active) == 1
    assert active[0].start_ms == 3000
    assert active[0].end_ms == 3600
    assert active[0].speaker == "A"


def test_parse_speaker_revision_returns_revised_events() -> None:
    provider = AssemblyAIStreamingDiarizationProvider("k")
    msg = {
        "type": "SpeakerRevision",
        "revisions": [{
            "turn_order": 1,
            "speaker_label": "B",
            "words": [
                {"text": "修正", "start": 1000, "end": 1500, "speaker": "B"},
            ],
        }],
    }

    events = provider._parse_message(json.dumps(msg))

    assert len(events) == 1
    assert events[0].speaker == "B"
    assert events[0].start_ms == 1000
    assert events[0].end_ms == 1500


def test_start_after_close_clears_stop_and_active_turns(monkeypatch) -> None:
    class WS:
        def recv(self) -> str:
            raise RuntimeError("stop")

        def send(self, payload: str) -> None:
            pass

        def close(self) -> None:
            pass

    provider = AssemblyAIStreamingDiarizationProvider("k")
    provider._stop.set()
    provider._active_by_turn[1] = provider._event_from_turn_label([
        {"start": 1000, "end": 1500},
    ], "A")[0]
    monkeypatch.setattr("websockets.sync.client.connect", lambda *a, **k: WS())

    provider.start()

    assert not provider._stop.is_set()
    assert provider._active_by_turn == {}

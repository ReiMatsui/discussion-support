"""pyannoteAI streaming diarization provider の単体テスト."""
from __future__ import annotations

import json
import struct

import numpy as np

from das.asr.live._pyannote_diarization import (
    PyannoteStreamingDiarizationProvider,
    pcm16_to_pyannote_f32,
)


def test_pcm16_to_pyannote_f32_converts_to_float32_le() -> None:
    got = pcm16_to_pyannote_f32(struct.pack("<hhh", 0, 16384, -32768))
    vals = struct.unpack("<fff", got)

    assert vals == (0.0, 0.5, -1.0)


def test_parse_message_returns_closed_turn_on_speaker_end() -> None:
    provider = PyannoteStreamingDiarizationProvider("k")

    start = {
        "type": "diarization_speaker_start",
        "data": {"timestamp": 1.25, "speaker": "SPEAKER_00"},
    }
    end = {
        "type": "diarization_speaker_end",
        "data": {"timestamp": 2.50, "speaker": "SPEAKER_00"},
    }

    assert provider._parse_message(json.dumps(start)) is None
    event = provider._parse_message(json.dumps(end))

    assert event is not None
    assert event.start_ms == 1250
    assert event.end_ms == 2500
    assert event.speaker == "SPEAKER_00"
    assert event.source == "pyannote"


def test_active_events_exposes_open_speaker_turns() -> None:
    provider = PyannoteStreamingDiarizationProvider("k")
    start = {
        "type": "diarization_speaker_start",
        "data": {"timestamp": 1.25, "speaker": "SPEAKER_00"},
    }

    assert provider._parse_message(json.dumps(start)) is None

    event = provider.active_events()[0]
    assert event.start_ms == 1250
    assert event.end_ms is None
    assert event.speaker == "SPEAKER_00"
    assert event.source == "pyannote"


def test_send_audio_uses_pyannote_float_payload() -> None:
    class WS:
        def __init__(self) -> None:
            self.sent: list[bytes] = []

        def send(self, payload: bytes) -> None:
            self.sent.append(payload)

    ws = WS()
    provider = PyannoteStreamingDiarizationProvider("k")
    provider._ws = ws

    provider.send_audio(struct.pack("<hh", 0, 32767))

    assert len(ws.sent) == 1
    got = np.frombuffer(ws.sent[0], dtype="<f4")
    assert got[0] == 0.0
    assert 0.9999 < got[1] < 1.0

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


def test_start_after_close_clears_stop_and_active_speakers(monkeypatch) -> None:
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self) -> bytes:
            return b'{"url":"ws://example"}'

    class WS:
        def recv(self) -> str:
            raise RuntimeError("stop")

        def send(self, payload: str) -> None:
            pass

        def close(self) -> None:
            pass

    provider = PyannoteStreamingDiarizationProvider("k")
    provider._stop.set()
    provider._active_starts["SPEAKER_00"] = 1
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: Resp())
    monkeypatch.setattr("websockets.sync.client.connect", lambda *a, **k: WS())

    provider.start()

    assert not provider._stop.is_set()
    assert provider._active_starts == {}


def test_send_audio_buffers_until_100ms_chunk_boundary() -> None:
    """Live-1は16kHz mono pcm_f32leの100ms固定チャンク(1600サンプル)を要求するため、
    provider内部でその境界まで送信を保留する必要がある(仕様: docs.pyannote.ai/
    tutorials/streaming-real-time)。"""

    class WS:
        def __init__(self) -> None:
            self.sent: list[bytes] = []

        def send(self, payload: bytes) -> None:
            self.sent.append(payload)

    ws = WS()
    provider = PyannoteStreamingDiarizationProvider("k")
    provider._ws = ws

    # 2サンプルだけでは100ms(1600サンプル)に満たないため、まだ送信されない。
    provider.send_audio(struct.pack("<hh", 0, 32767))
    assert ws.sent == []

    # 残り1598サンプル分を追加すると、ちょうど1600サンプル=6400バイトが1回で送られる。
    provider.send_audio(struct.pack("<1598h", *([0] * 1598)))

    assert len(ws.sent) == 1
    assert len(ws.sent[0]) == 1600 * 4
    got = np.frombuffer(ws.sent[0], dtype="<f4")
    assert got[0] == 0.0
    assert 0.9999 < got[1] < 1.0


def test_send_audio_flushes_multiple_full_chunks_at_once() -> None:
    class WS:
        def __init__(self) -> None:
            self.sent: list[bytes] = []

        def send(self, payload: bytes) -> None:
            self.sent.append(payload)

    ws = WS()
    provider = PyannoteStreamingDiarizationProvider("k")
    provider._ws = ws

    # 3200サンプル分(200ms)を一度に渡すと、6400バイトのチャンクが2回送られる。
    provider.send_audio(struct.pack("<3200h", *([0] * 3200)))

    assert len(ws.sent) == 2
    assert all(len(p) == 1600 * 4 for p in ws.sent)

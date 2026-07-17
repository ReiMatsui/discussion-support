"""pyannoteAI streaming diarization provider の単体テスト."""
from __future__ import annotations

import json
import struct
from typing import Any

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


def test_parse_message_discards_degenerate_zero_length_segment() -> None:
    """start と end が同時刻(区間長0)の縮退セグメントは下流に流さない."""
    provider = PyannoteStreamingDiarizationProvider("k")

    start = {
        "type": "diarization_speaker_start",
        "data": {"timestamp": 1.25, "speaker": "SPEAKER_00"},
    }
    end_same_ts = {
        "type": "diarization_speaker_end",
        "data": {"timestamp": 1.25, "speaker": "SPEAKER_00"},
    }

    assert provider._parse_message(json.dumps(start)) is None
    assert provider._parse_message(json.dumps(end_same_ts)) is None
    # 破棄後は active_starts からも取り除かれている
    assert provider.active_events() == []


def test_parse_message_discards_end_without_matching_start() -> None:
    """対応する speaker_start が無い speaker_end は縮退セグメント化を避けて破棄する."""
    provider = PyannoteStreamingDiarizationProvider("k")

    end_without_start = {
        "type": "diarization_speaker_end",
        "data": {"timestamp": 2.0, "speaker": "SPEAKER_01"},
    }

    assert provider._parse_message(json.dumps(end_without_start)) is None


def test_max_speakers_stored_but_not_sent_to_create_stream_api(monkeypatch) -> None:
    """Live-1のPOST /v1/liveはボディにプロパティを持たないため、max_speakersを
    指定してもAPIリクエストボディは常に空({})のまま送られる（配線だけ用意）."""
    captured: dict[str, Any] = {}

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

        def send(self, payload) -> None:
            pass

        def close(self) -> None:
            pass

    def fake_urlopen(req, *a, **k):
        captured["data"] = req.data
        return Resp()

    provider = PyannoteStreamingDiarizationProvider("k", max_speakers=3)
    provider._stop.set()
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("websockets.sync.client.connect", lambda *a, **k: WS())

    provider.start()

    assert provider.max_speakers == 3
    assert captured["data"] == b"{}"


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


def test_restart_advances_label_epoch_to_avoid_key_collision(monkeypatch) -> None:
    """close→start の再起動でラベルepochが進み、旧ラベル空間と衝突しない（F3）.

    _bootstrap の STT切断復旧は同一インスタンスに provider.close();
    provider.start() を行う。従来 start() が epoch を 0 にリセットしていたため、
    再起動後の SPEAKER_00 が旧セッションの確定名
    （cluster_namer._confirmed / diarization_speaker_keys の
    "pyannote:SPEAKER_00"）へ即誤帰属し得た（2026-07-15 レビューで確定）。
    epoch を引き継いでインクリメントすれば R{epoch}: 前置で自然に区別される。
    """
    provider = PyannoteStreamingDiarizationProvider("k", auto_reconnect=False)
    monkeypatch.setattr(provider, "_connect", lambda: None)

    def speaker_of(p):
        start = {"type": "diarization_speaker_start",
                 "data": {"timestamp": 1.0, "speaker": "SPEAKER_00"}}
        end = {"type": "diarization_speaker_end",
               "data": {"timestamp": 2.0, "speaker": "SPEAKER_00"}}
        assert p._parse_message(json.dumps(start)) is None
        return p._parse_message(json.dumps(end)).speaker

    provider.start()
    assert speaker_of(provider) == "SPEAKER_00"      # 初回は epoch=0（前置なし）

    provider._sent_audio_ms = 5000                   # 5秒送信済みの想定
    provider.close()
    provider.start()
    # 再起動後は raw キーが変わり（R1: 前置）、旧 SPEAKER_00 と衝突しない
    assert speaker_of(provider) == "R1:SPEAKER_00"
    # タイムラインは再接続時と同様に累計msを引き継いで単調を保つ
    assert provider._session_base_ms == 5000

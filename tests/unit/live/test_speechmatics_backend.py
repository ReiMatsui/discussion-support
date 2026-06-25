"""Speechmatics リアルタイムSTTバックエンドのテスト."""
from __future__ import annotations

from das.asr.live.stt._speechmatics import SpeechmaticsBackend


def test_start_message_enables_speaker_diarization_config() -> None:
    backend = SpeechmaticsBackend("k", max_speakers=3)

    msg = backend.start_message("unused", "ja")

    config = msg["transcription_config"]
    assert config["diarization"] == "speaker"
    assert config["speaker_diarization_config"] == {
        "prefer_current_speaker": True,
        "max_speakers": 3,
    }


def test_parse_message_converts_speaker_labels() -> None:
    backend = SpeechmaticsBackend("k")

    got = backend.parse_message({
        "message": "AddTranscript",
        "results": [{
            "type": "word",
            "start_time": 1.0,
            "end_time": 1.5,
            "alternatives": [{"content": "こんにちは", "speaker": "S2"}],
        }],
    }, "ja")

    assert got["tokens"][0]["speaker"] == "2"
    assert got["tokens"][0]["is_final"] is True

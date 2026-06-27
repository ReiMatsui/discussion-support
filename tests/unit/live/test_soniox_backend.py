"""Soniox バックエンドの開始メッセージ設定のテスト."""
from __future__ import annotations

from das.asr.live.stt._soniox import SonioxBackend


def test_speaker_diarization_always_enabled():
    m = SonioxBackend("k").start_message("stt-rt-v5", "ja")
    assert m["enable_speaker_diarization"] is True
    assert m["language_hints"] == ["ja"]


def test_endpoint_detection_on_by_default():
    """既定でエンドポイント検出はON（文の切れ目で区切る＝議事録が読みやすい）."""
    m = SonioxBackend("k").start_message("stt-rt-v5", "ja")
    assert m["enable_endpoint_detection"] is True


def test_endpoint_detection_can_be_disabled():
    """明示的にOFFにもできる."""
    m = SonioxBackend("k", enable_endpoint_detection=False).start_message("m", "ja")
    assert m["enable_endpoint_detection"] is False

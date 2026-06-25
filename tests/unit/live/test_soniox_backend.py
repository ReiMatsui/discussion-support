"""Soniox バックエンドの開始メッセージ設定のテスト.

公式ドキュメント: エンドポイント検出は話者分離の精度を下げる（早期確定で揺れる
ラベルをロックする）。既定はOFF、A/B比較用に切替可能であることを保証する。
"""
from __future__ import annotations

from das.asr.live.stt._soniox import SonioxBackend


def test_speaker_diarization_always_enabled():
    m = SonioxBackend("k").start_message("stt-rt-v4", "ja")
    assert m["enable_speaker_diarization"] is True
    assert m["language_hints"] == ["ja"]


def test_endpoint_detection_off_by_default():
    """既定でエンドポイント検出はOFF（話者分離の精度優先）."""
    m = SonioxBackend("k").start_message("stt-rt-v4", "ja")
    assert m["enable_endpoint_detection"] is False


def test_endpoint_detection_can_be_enabled_for_ab():
    """A/B比較のため明示的にONにできる."""
    m = SonioxBackend("k", enable_endpoint_detection=True).start_message("m", "ja")
    assert m["enable_endpoint_detection"] is True

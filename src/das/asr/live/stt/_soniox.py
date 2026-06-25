"""Soniox リアルタイムSTTバックエンド."""
from __future__ import annotations

from .._constants import SR

WS_URL = "wss://stt-rt.soniox.com/transcribe-websocket"


class SonioxBackend:
    """Soniox WebSocket STTプロバイダ.

    Sonioxのトークン形式がそのまま内部形式なので、parse_messageはパススルー。
    """

    def __init__(self, api_key: str, enable_endpoint_detection: bool = False):
        self._api_key = api_key
        # 公式ドキュメント: エンドポイント検出はトークンを早期確定させ、揺れている
        # 途中の話者ラベルをロックして話者分離の精度を下げる。既定はOFF。
        # https://soniox.com/docs/stt/concepts/speaker-diarization
        self._enable_endpoint_detection = enable_endpoint_detection

    @property
    def name(self) -> str:
        return "soniox"

    def ws_url(self) -> str:
        return WS_URL

    def ws_headers(self) -> dict[str, str] | None:
        return None

    def start_message(self, model: str, lang: str) -> dict:
        return {
            "api_key": self._api_key,
            "model": model,
            "language_hints": [lang],
            "enable_speaker_diarization": True,
            "enable_endpoint_detection": self._enable_endpoint_detection,
            "audio_format": "pcm_s16le",
            "sample_rate": SR,
            "num_channels": 1,
        }

    def parse_message(self, raw: dict, lang: str) -> dict:
        return raw

    def make_end_message(self, seq: int) -> str | bytes:
        return ""

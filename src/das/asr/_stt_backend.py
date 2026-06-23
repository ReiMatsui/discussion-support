"""STTプロバイダの共通インターフェース.

新しいSTTプロバイダを追加するには、このProtocolを実装するクラスを作成し、
soniox_live.py の _build_backend() に選択肢を追加するだけでよい。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class STTBackend(Protocol):
    """リアルタイムSTTプロバイダの抽象インターフェース.

    内部トークン形式（parse_messageの戻り値）:
        {"tokens": [{"text": str, "speaker": str,
                      "start_ms": int, "end_ms": int, "is_final": bool}]}
        特殊トークン: {"text": "<end>", "is_final": True}  # 発話境界
        ストリーム終了: {"finished": True, "tokens": []}
        エラー: {"error_code": str, "error_message": str}
    """

    @property
    def name(self) -> str:
        """プロバイダ名（"soniox", "speechmatics" 等）."""
        ...

    def ws_url(self) -> str:
        """WebSocket接続先URL."""
        ...

    def ws_headers(self) -> dict[str, str] | None:
        """WebSocket接続時の追加ヘッダー（認証等）。不要ならNone."""
        ...

    def start_message(self, model: str, lang: str) -> dict:
        """接続直後に送信する開始メッセージ."""
        ...

    def parse_message(self, raw: dict, lang: str) -> dict:
        """受信メッセージを内部トークン形式に変換."""
        ...

    def make_end_message(self, seq: int) -> str | bytes:
        """音声送信終了時の終端メッセージ."""
        ...

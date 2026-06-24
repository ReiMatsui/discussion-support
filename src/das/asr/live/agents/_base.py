"""RealtimeAgent / ConversationPartner の共通基底クラス（Phase 3 R3）.

両エージェントで重複していた再生キュー・声紋・受信ループ等の実装を集約する。
サブクラスは固有の _handle / セッション設定 / 送信系メソッドのみ持つ。
"""
from __future__ import annotations

import collections
import queue
import time
from typing import TYPE_CHECKING

from .._voice_profiles import _best_text_similarity

if TYPE_CHECKING:
    from .._voice_profiles import VoiceProfiles


class _RealtimeBase:
    """Realtime API ベースの音声エージェントの共通実装.

    サブクラスが __init__ で以下の属性を用意することを前提とする:
      _audio_q / _play_epoch / ai_speaking / _last_speech_end /
      _voice_tracker / _recent_ai_texts / _ai_text_buf
    """

    # サブクラスが __init__ で設定する共有属性（mypy strict 用の型注釈）
    _audio_q: queue.Queue[tuple[int, bytes | None]]
    _play_epoch: int
    ai_speaking: bool
    _last_speech_end: float
    _voice_tracker: VoiceProfiles | None
    _recent_ai_texts: collections.deque
    _ai_text_buf: str

    def set_tracker(self, tracker: VoiceProfiles):
        """VoiceProfilesを外部から注入。connect()の前後いつでも可。"""
        self._voice_tracker = tracker

    def _q_put(self, payload: bytes | None):
        """再生キューに現在の応答世代(epoch)タグを付けて積む（Bug 6）.

        payload=None は応答の終端マーカー。
        """
        self._audio_q.put((self._play_epoch, payload))

    def _on_playback_terminator(self, epoch: int):
        """終端マーカー取り出し時、最新応答の終端のみ ai_speaking を倒す（Bug 6）."""
        if epoch >= self._play_epoch:
            self.ai_speaking = False
            self._last_speech_end = time.monotonic()

    def _best_similarity(self, text: str) -> float:
        return _best_text_similarity(text, list(self._recent_ai_texts),
                                     self._ai_text_buf)

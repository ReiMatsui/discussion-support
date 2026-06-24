"""RealtimeAgent / ConversationPartner の共通基底クラス（Phase 3 R3）.

両エージェントで重複していた再生キュー・声紋・受信ループ等の実装を集約する。
サブクラスは固有の _handle / セッション設定 / 送信系メソッドのみ持つ。
"""
from __future__ import annotations

import collections
import queue
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from .._voice_profiles import _best_text_similarity, _resample_24_to_16

if TYPE_CHECKING:
    from .._voice_profiles import VoiceProfiles


class _RealtimeBase:
    """Realtime API ベースの音声エージェントの共通実装.

    サブクラスが __init__ で以下の属性を用意することを前提とする:
      _audio_q / _play_epoch / ai_speaking / _last_speech_end /
      _voice_tracker / _recent_ai_texts / _ai_text_buf /
      _stop / _played_bytes / _ai_voice_enrolled / _ai_voice_buf /
      _ai_voice_sec / _playback_thread
    サブクラスは AI_VOICE_KEY と _LABEL を上書きする。
    """

    # サブクラスで上書きするクラス属性
    AI_VOICE_KEY: str = "__BASE__"   # VoiceProfiles内のAI声紋キー
    _AI_ENROLL_SEC: float = 3.0      # 声紋登録に必要な最小秒数
    _LABEL: str = "Agent"            # ログ用ラベル

    # サブクラスが __init__ で設定する共有属性（mypy strict 用の型注釈）
    _audio_q: queue.Queue[tuple[int, bytes | None]]
    _play_epoch: int
    ai_speaking: bool
    _last_speech_end: float
    _voice_tracker: VoiceProfiles | None
    _recent_ai_texts: collections.deque
    _ai_text_buf: str
    _stop: threading.Event
    _played_bytes: int
    _ai_voice_enrolled: bool
    _ai_voice_buf: list[np.ndarray]
    _ai_voice_sec: float
    _playback_thread: threading.Thread | None

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

    # --- 声紋登録 ---

    def _try_enroll_voice(self):
        """蓄積した音声から声紋を計算しVoiceProfilesに登録する。

        再生スレッドから呼ばれる。十分な音声が溜まったら1回だけ実行。
        """
        if self._ai_voice_enrolled or self._voice_tracker is None:
            return
        if self._ai_voice_sec < self._AI_ENROLL_SEC:
            return
        wav = np.concatenate(self._ai_voice_buf)
        tracker = self._voice_tracker
        emb = tracker._embed(wav)
        if emb is None:
            return
        with tracker._lock:
            tracker.profiles[self.AI_VOICE_KEY] = emb
            tracker._active_keys.add(self.AI_VOICE_KEY)
        self._ai_voice_enrolled = True
        self._ai_voice_buf.clear()   # メモリ解放
        print(f"# {self._LABEL}: 声紋を登録しました（{self._ai_voice_sec:.1f}秒の音声から）",
              flush=True)

    # --- ストリーミング音声再生 ---

    def _start_playback_thread(self):
        """PCMキューから読み出して逐次再生するスレッド。

        再生済みバイト数を_played_bytesに蓄積（truncate用）。
        声紋未登録時は16kHzリサンプル音声を蓄積して自動登録。
        キュー要素は (epoch, payload)。payload=None は応答の終端マーカー。
        """
        def _player():
            try:
                import sounddevice as sd
                stream = sd.OutputStream(samplerate=24000, channels=1,
                                         dtype="float32", blocksize=2400)
                stream.start()
                while not self._stop.is_set():
                    epoch, chunk = self._audio_q.get()
                    if chunk is None:          # 1応答の終端
                        self._on_playback_terminator(epoch)
                        continue
                    pcm = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
                    stream.write(pcm.reshape(-1, 1))
                    self._played_bytes += len(chunk)
                    # 声紋登録用: 16kHzにリサンプルして蓄積
                    if not self._ai_voice_enrolled and self._voice_tracker is not None:
                        ref16 = _resample_24_to_16(pcm)
                        if len(ref16) > 0:
                            self._ai_voice_buf.append(ref16.copy())
                            self._ai_voice_sec += len(ref16) / 16000.0
                            self._try_enroll_voice()
                stream.stop()
                stream.close()
            except Exception as e:
                print(f"# {self._LABEL} 音声再生異常: {e}", flush=True)

        self._playback_thread = threading.Thread(target=_player, daemon=True)
        self._playback_thread.start()

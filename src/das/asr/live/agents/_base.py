"""LiveAgent / LivePartner の共通基底クラス（Phase 3 R3 で導入、WP8 で GPT-Live 専用に）.

再生キュー・声紋（エコー除去）・受信ループ等、話す層に依存しない実装を集約する。
サブクラスは固有の _handle / セッション設定 / 送信系メソッドのみ持つ。
"""
from __future__ import annotations

import collections
import contextlib
import queue
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from .._voice_profiles import _best_text_similarity, _resample_24_to_16

if TYPE_CHECKING:
    from .._voice_profiles import VoiceProfiles


class _VoiceAgentBase:
    """WebSocket 型の音声エージェント（GPT-Live）の共通実装.

    サブクラスが __init__ で以下の属性を用意することを前提とする:
      _audio_q / _play_epoch / ai_speaking / _last_speech_end /
      _voice_tracker / _recent_ai_texts / _ai_text_buf /
      _stop / _played_bytes / _ai_voice_enrolled / _ai_voice_buf /
      _ai_voice_sec / _playback_thread
    サブクラスは AI_VOICE_KEY と _LABEL を上書きする。
    """

    # サブクラスで上書きするクラス属性
    AI_VOICE_KEY: str = "__BASE__"   # VoiceProfiles内のAI声紋キー
    _AI_ENROLL_SEC: float = 3.0      # 声紋登録に必要な最小秒数（有声部分のみ）
    _AI_ENROLL_MAX_SEC: float = 30.0  # これ以上溜めても登録できないなら諦める
    _AI_ENROLL_MIN_RMS: float = 200.0 / 32768.0   # float32 正規化後の無音判定（_live._SILENCE_RMS と同じ）
    _LABEL: str = "Agent"            # ログ用ラベル
    _conn_error: str = ""            # 接続エラーメッセージ（UI表示用、共通デフォルト）

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
    _connected: bool
    ws: Any

    def set_tracker(self, tracker: VoiceProfiles):
        """VoiceProfilesを外部から注入。connect()の前後いつでも可。"""
        self._voice_tracker = tracker

    def _q_put(self, payload: bytes | None):
        """再生キューに現在の応答世代(epoch)タグを付けて積む（Bug 6）.

        payload=None は応答の終端マーカー。
        """
        if payload:
            with self._queued_lock:
                self._queued_ms += len(payload) * 1000 // (24000 * 2)
        self._audio_q.put((self._play_epoch, payload))

    def _queued_audio_ms(self) -> int:
        """再生待ち（キュー）と鳴らし中の残り（ジッタバッファ）の合計 ms."""
        with self._queued_lock:
            q = self._queued_ms
        return q + self._jitter.buffered_ms

    def _on_playback_terminator(self, epoch: int):
        """終端マーカー取り出し時、最新応答の終端のみ ai_speaking を倒す（Bug 6）."""
        if epoch >= self._play_epoch:
            self._end_speech()

    def _end_speech(self) -> None:
        """AI発話の終了を確定し、終了フック（on_speech_end）があれば通知する.

        自然終了（終端マーカー）と割り込み終了の共通処理。P2-1 のエコー区間記録が
        このフックでAI再生区間を閉じる。
        """
        self.ai_speaking = False
        self._last_speech_end = time.monotonic()
        cb = getattr(self, "on_speech_end", None)
        if cb:
            with contextlib.suppress(Exception):
                cb()

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
        """再生スレッド: 受信キューの PCM をジッタバッファへ移し、デバイスはコールバックで
        一定速度に取り出す（届いた順に書く方式だと到着の揺れがそのまま途切れになる）。

        キュー要素は (epoch, payload)。payload=None は応答の終端マーカーで、
        バッファが鳴り終わってから ai_speaking を倒す。
        声紋未登録時は有声チャンクを16kHzにして蓄積し、自動登録する。
        """
        def _player():
            jb = self._jitter
            try:
                import sounddevice as sd

                def _cb(outdata, frames, _t, status):
                    if status and not self._playback_status_warned:
                        self._playback_status_warned = True
                        print(f"# {self._LABEL} 再生デバイス: {status}", flush=True)
                    pcm = jb.pull(frames * 2)
                    outdata[:, 0] = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0

                stream = sd.OutputStream(samplerate=24000, channels=1,
                                         dtype="float32", blocksize=1200, callback=_cb)
                stream.start()
                while not self._stop.is_set():
                    epoch, chunk = self._audio_q.get()
                    if chunk is None:          # 1応答の終端: 鳴り終わるまで待ってから
                        deadline = time.monotonic() + 3.0
                        while (jb.buffered_ms > 0 and not self._stop.is_set()
                               and time.monotonic() < deadline):
                            time.sleep(0.02)
                        self._on_playback_terminator(epoch)
                        continue
                    with self._queued_lock:
                        self._queued_ms = max(0, self._queued_ms - len(chunk) * 1000 // (24000 * 2))
                    jb.push(chunk)
                    self._played_bytes += len(chunk)
                    # 声紋登録用: 16kHzにリサンプルして蓄積。無音のチャンク（文の間、
                    # 終端前の 1.2 秒）は声紋を薄めるだけなので入れない。登録に失敗し
                    # 続けるときは溜め続けない（メモリと毎回の連結を抑える）
                    if (not self._ai_voice_enrolled and self._voice_tracker is not None
                            and self._ai_voice_sec < self._AI_ENROLL_MAX_SEC):
                        pcm = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
                        if float(np.sqrt(np.mean(pcm * pcm))) >= self._AI_ENROLL_MIN_RMS:
                            ref16 = _resample_24_to_16(pcm)
                            if len(ref16) > 0:
                                self._ai_voice_buf.append(ref16.copy())
                                self._ai_voice_sec += len(ref16) / 16000.0
                                self._try_enroll_voice()
                stream.stop()
                stream.close()
            except Exception as e:
                print(f"# {self._LABEL} 音声再生異常: {e}", flush=True)

        if self._playback_thread is not None and self._playback_thread.is_alive():
            return
        self._playback_thread = threading.Thread(target=_player, daemon=True)
        self._playback_thread.start()

    def _playback_idle(self) -> bool:
        """再生待ちも鳴らし中の残りも無いか."""
        return self._audio_q.empty() and self._jitter.buffered_ms == 0

    # --- 終了処理 ---

    def close(self):
        """停止フラグを立て、再生スレッド・声紋・WebSocketを片付ける（共通）."""
        self._stop.set()
        self._q_put(None)  # playback threadを起こして終了させる
        if self._playback_thread is not None:
            self._playback_thread.join(timeout=2.0)
        # セッション限りのAI声紋をクリーンアップ
        if (self._voice_tracker is not None
                and self.AI_VOICE_KEY in self._voice_tracker.profiles):
            with self._voice_tracker._lock:
                self._voice_tracker.profiles.pop(self.AI_VOICE_KEY, None)
                self._voice_tracker._active_keys.discard(self.AI_VOICE_KEY)
        if self.ws:
            with contextlib.suppress(Exception):
                self.ws.close()

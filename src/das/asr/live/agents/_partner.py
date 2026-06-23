"""人間と音声で直接議論する Realtime API エージェント."""
from __future__ import annotations

import base64
import collections
import contextlib
import json
import queue
import threading
import time

import numpy as np

from .._constants import _PROMPT_DEBATE_PARTNER, REALTIME_URL
from .._voice_profiles import VoiceProfiles, _best_text_similarity, _resample_24_to_16


class ConversationPartner:
    """人間と音声で直接議論するRealtime APIエージェント.

    人間のマイク音声をinput_audio_buffer.appendで受け取り、
    server VADで自動的にターンを検出して応答する。
    ファシリテーター（既存RealtimeAgent）とは独立したセッション。

    使い方:
      partner = ConversationPartner(api_key, topic="AIツール導入の是非")
      partner.connect()
      partner.feed_audio(pcm_24k_bytes)  # マイク音声を継続的に送信
    """

    def __init__(self, api_key: str, voice: str = "echo", topic: str = ""):
        self.api_key = api_key
        self.voice = voice
        self.topic = topic
        self.ws = None
        self._stop = threading.Event()
        self._connected = False
        self.ai_speaking = False
        self._responding = False
        self._interrupted = False          # interrupt後の残留イベント破棄用
        self._audio_q: queue.Queue[bytes | None] = queue.Queue()
        self._playback_thread: threading.Thread | None = None
        self._ai_text_buf = ""
        self.on_ai_utterance = None       # callback(text: str)
        self._recent_ai_texts: collections.deque = collections.deque(maxlen=20)
        self._last_speech_end = 0.0
        self._echo_cooldown = 2.0
        # --- truncate用: 再生済み音声の追跡 ---
        self._current_item_id: str | None = None
        self._played_bytes = 0
        # --- AI声紋登録用 ---
        self._voice_tracker: VoiceProfiles | None = None
        self._ai_voice_buf: list[np.ndarray] = []
        self._ai_voice_sec = 0.0
        self._ai_voice_enrolled = False

    AI_VOICE_KEY = "__PARTNER__"   # ファシリテーターの__AI__と区別

    @property
    def in_echo_window(self) -> bool:
        """AI発話中 or 直後のエコーウィンドウ内か."""
        if self.ai_speaking or self._responding:
            return True
        if self._last_speech_end > 0:
            return (time.monotonic() - self._last_speech_end) < self._echo_cooldown
        return False

    def set_tracker(self, tracker: VoiceProfiles):
        self._voice_tracker = tracker

    def connect(self):
        """WebSocket接続を開始."""
        try:
            from websockets.sync.client import connect
        except ImportError:
            print("# Partner: websockets未インストール", flush=True)
            return
        try:
            self.ws = connect(
                REALTIME_URL,
                additional_headers={"Authorization": f"Bearer {self.api_key}"},
            )
        except Exception as e:
            print(f"# Partner: 接続失敗 ({e})", flush=True)
            return
        self._connected = True
        self._send_session_update()
        threading.Thread(target=self._recv_loop, daemon=True).start()
        self._start_playback_thread()
        print(f"# Partner: 接続完了（voice={self.voice}）", flush=True)

    def _send_session_update(self):
        """server VAD有効 + 音声入出力の設定."""
        if not self.ws:
            return
        prompt = _PROMPT_DEBATE_PARTNER
        if self.topic:
            prompt += f"\n\n今日の議題: {self.topic}"
        try:
            self.ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": prompt,
                    "audio": {
                        "input": {
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": 0.5,
                                "prefix_padding_ms": 300,
                                "silence_duration_ms": 500,
                            },
                        },
                        "output": {
                            "voice": self.voice,
                        },
                    },
                },
            }))
        except Exception as e:
            print(f"# Partner: session.update失敗 ({e})", flush=True)

    def feed_audio(self, pcm_16k: bytes):
        """16kHz 16bit PCMを24kHz PCMに変換してRealtime APIに送信."""
        if not self._connected or not self.ws:
            return
        # 16kHz → 24kHz アップサンプル
        samples = np.frombuffer(pcm_16k, dtype="<i2").astype(np.float32)
        n_out = int(len(samples) * 24000 / 16000)
        if n_out < 2:
            return
        indices = np.linspace(0, len(samples) - 1, n_out)
        samples_24k = np.interp(indices, np.arange(len(samples)), samples)
        pcm_24k = np.clip(samples_24k, -32768, 32767).astype("<i2").tobytes()
        with contextlib.suppress(Exception):
            self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm_24k).decode(),
            }))

    def inject_context(self, speaker: str, text: str, *,
                        request_response: bool = False):
        """外部テキストをPartnerの会話履歴に注入.

        request_response=True の場合、注入後にresponse.createを送信して
        応答を明示的に要求する。割り込み後に新しい質問へ応答させる場合に使う。
        """
        if not self._connected or not self.ws:
            return
        # interrupt()直後の呼び出し時、_interruptedをリセットしないと
        # 新しい応答の音声チャンクまで破棄されてしまう
        if request_response:
            self._interrupted = False
        try:
            self.ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text",
                                 "text": f"[{speaker}]: {text}"}],
                },
            }))
            if request_response:
                self.ws.send(json.dumps({"type": "response.create"}))
        except Exception:
            pass

    def interrupt(self):
        """外部からの割り込み（ファシリテーター介入時に使用）.

        AI同士の制御なので即停止。graceful yieldは不要。
        - _interruptedフラグで残留イベントを破棄
        - conversation.item.truncateで会話履歴を再生済み分に切り詰め
        """
        if not self.ai_speaking and not self._responding:
            return
        self._interrupted = True
        # キュー即排出
        while True:
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break
        self._audio_q.put(None)  # playback threadにEOSを通知
        self.ai_speaking = False
        self._responding = False
        self._last_speech_end = time.monotonic()
        if self.ws:
            try:
                self.ws.send(json.dumps({"type": "response.cancel"}))
                if self._current_item_id:
                    self.ws.send(json.dumps({
                        "type": "conversation.item.truncate",
                        "item_id": self._current_item_id,
                        "content_index": 0,
                        "audio_end_ms": int(self._played_bytes // 2 / 24),
                    }))
            except Exception:
                pass

    # --- ストリーミング音声再生 ---

    def _start_playback_thread(self):
        def _player():
            try:
                import sounddevice as sd
                stream = sd.OutputStream(samplerate=24000, channels=1,
                                         dtype="float32", blocksize=2400)
                stream.start()
                while not self._stop.is_set():
                    chunk = self._audio_q.get()
                    if chunk is None:
                        self.ai_speaking = False
                        self._last_speech_end = time.monotonic()
                        continue
                    self._played_bytes += len(chunk)
                    pcm = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
                    stream.write(pcm.reshape(-1, 1))
                    # AI声紋登録用
                    if not self._ai_voice_enrolled and self._voice_tracker is not None:
                        ref16 = _resample_24_to_16(pcm)
                        if len(ref16) > 0:
                            self._ai_voice_buf.append(ref16.copy())
                            self._ai_voice_sec += len(ref16) / 16000.0
                            if self._ai_voice_sec >= 3.0:
                                self._try_enroll_voice()
                stream.stop()
                stream.close()
            except Exception as e:
                print(f"# Partner 音声再生異常: {e}", flush=True)

        self._playback_thread = threading.Thread(target=_player, daemon=True)
        self._playback_thread.start()

    def _try_enroll_voice(self):
        if self._ai_voice_enrolled or self._voice_tracker is None:
            return
        wav = np.concatenate(self._ai_voice_buf)
        emb = self._voice_tracker._embed(wav)
        if emb is None:
            return
        with self._voice_tracker._lock:
            self._voice_tracker.profiles[self.AI_VOICE_KEY] = emb
            self._voice_tracker._active_keys.add(self.AI_VOICE_KEY)
        self._ai_voice_enrolled = True
        self._ai_voice_buf.clear()
        print(f"# Partner: 声紋を登録しました（{self._ai_voice_sec:.1f}秒の音声から）",
              flush=True)

    # --- WebSocket受信 ---

    def _recv_loop(self):
        while not self._stop.is_set():
            try:
                raw = self.ws.recv()
                ev = json.loads(raw)
            except Exception as e:
                if not self._stop.is_set():
                    print(f"# Partner: WebSocket切断 ({e})", flush=True)
                break
            self._handle(ev)
        self._connected = False

    def _best_similarity(self, text: str) -> float:
        return _best_text_similarity(text, list(self._recent_ai_texts),
                                     self._ai_text_buf)

    def _handle(self, ev: dict):
        etype = ev.get("type", "")

        if etype == "response.output_item.added":
            item = ev.get("item", {})
            self._current_item_id = item.get("id")
            self._played_bytes = 0

        elif etype == "response.output_audio.delta":
            if self._interrupted:
                return  # interrupt後の残留チャンクを破棄
            chunk = ev.get("delta", "")
            if chunk:
                self._audio_q.put(base64.b64decode(chunk))
                self.ai_speaking = True
                self._responding = True

        elif etype == "response.output_audio_transcript.delta":
            if self._interrupted:
                return
            self._ai_text_buf += ev.get("delta", "")

        elif etype == "response.output_audio_transcript.done":
            transcript = ev.get("transcript", "") or self._ai_text_buf
            self._ai_text_buf = ""
            if transcript:
                # エコー判定用には常に記録（中断されたテキストもASRに拾われうる）
                self._recent_ai_texts.append(transcript)
                # 中断された応答は議事録に載せない（音声が再生されていない）
                if not self._interrupted and self.on_ai_utterance:
                    self.on_ai_utterance(transcript)

        elif etype == "response.output_audio.done":
            if not self._interrupted:
                self._audio_q.put(None)

        elif etype == "response.done":
            resp = ev.get("response", {})
            status = resp.get("status", "")
            if status == "cancelled":
                # キャンセル済み応答: テキストをエコー判定用に記録、
                # _respondingはリセットしない（直後にrequest_responseで
                # 新しい応答が来る場合があるため）
                if self._ai_text_buf:
                    partial = self._ai_text_buf.strip()
                    if partial:
                        self._recent_ai_texts.append(partial)
                self._ai_text_buf = ""
                self._interrupted = False
            else:
                # 正常完了 or その他
                self._ai_text_buf = ""
                self._responding = False
                self._interrupted = False

        elif etype == "error":
            msg = ev.get("error", {}).get("message", "unknown")
            # response.cancel が active response なしに到達した場合は無視
            if "no active response" not in msg.lower():
                print(f"# Partner エラー: {msg}", flush=True)

    def close(self):
        self._stop.set()
        self._audio_q.put(None)
        if self._playback_thread:
            self._playback_thread.join(timeout=2.0)
        # 声紋プロファイルのクリーンアップ
        if self._voice_tracker is not None and self.AI_VOICE_KEY in self._voice_tracker.profiles:
            with self._voice_tracker._lock:
                self._voice_tracker.profiles.pop(self.AI_VOICE_KEY, None)
                self._voice_tracker._active_keys.discard(self.AI_VOICE_KEY)
        if self.ws:
            with contextlib.suppress(Exception):
                self.ws.close()

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

from .._constants import (
    _ECHO_COOLDOWN,
    _PROMPT_DEBATE_PARTNER,
    REALTIME_MODEL,
    realtime_url,
)
from .._voice_profiles import VoiceProfiles
from ._base import _RealtimeBase


class ConversationPartner(_RealtimeBase):
    """人間と音声で直接議論するRealtime APIエージェント.

    人間のマイク音声をinput_audio_buffer.appendで受け取り、
    server VADで自動的にターンを検出して応答する。
    ファシリテーター（既存RealtimeAgent）とは独立したセッション。

    使い方:
      partner = ConversationPartner(api_key, topic="AIツール導入の是非")
      partner.connect()
      partner.feed_audio(pcm_24k_bytes)  # マイク音声を継続的に送信
    """

    def __init__(self, api_key: str, voice: str = "echo", topic: str = "",
                 model: str = REALTIME_MODEL):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.topic = topic
        self.ws = None
        self._stop = threading.Event()
        self._connected = False
        self.ai_speaking = False
        self._responding = False
        self._interrupted = False          # interrupt後の残留イベント破棄用
        # 再生キュー要素は (epoch, payload)。payload=None は応答の終端マーカー。
        # epochは応答世代。古い応答の終端で新応答の再生中フラグを倒さない（Bug 6）。
        self._audio_q: queue.Queue[tuple[int, bytes | None]] = queue.Queue()
        self._play_epoch = 0               # 応答世代カウンタ（output_item.addedで+1）
        self._playback_thread: threading.Thread | None = None
        self._ai_text_buf = ""
        self.on_ai_utterance = None       # callback(text: str)
        self._recent_ai_texts: collections.deque = collections.deque(maxlen=20)
        self._last_speech_end = 0.0
        self._echo_cooldown = _ECHO_COOLDOWN
        # --- truncate用: 再生済み音声の追跡 ---
        self._current_item_id: str | None = None
        self._played_bytes = 0
        # --- AI声紋登録用 ---
        self._voice_tracker: VoiceProfiles | None = None
        self._ai_voice_buf: list[np.ndarray] = []
        self._ai_voice_sec = 0.0
        self._ai_voice_enrolled = False

    AI_VOICE_KEY = "__PARTNER__"   # ファシリテーターの__AI__と区別
    _LABEL = "Partner"             # ログ用ラベル（基底クラス用）
    # 良性エラー（実害なし）の判定用部分文字列。すべて小文字で比較する。
    #   - no active response: キャンセル対象の応答が無い
    #   - already has an active response: cancel→create と server VAD 自動応答の競合。
    #     VAD 側が応答するため、明示的な response.create が弾かれても問題ない。
    _BENIGN_ERROR_SUBSTRINGS = (
        "no active response",
        "already has an active response",
        "already an active response",
        "active response already",
    )

    @property
    def in_echo_window(self) -> bool:
        """AI発話中 or 直後のエコーウィンドウ内か."""
        if self.ai_speaking or self._responding:
            return True
        if self._last_speech_end > 0:
            return (time.monotonic() - self._last_speech_end) < self._echo_cooldown
        return False

    def connect(self):
        """WebSocket接続を開始."""
        try:
            from websockets.sync.client import connect
        except ImportError:
            print("# Partner: websockets未インストール", flush=True)
            return
        try:
            self.ws = connect(
                realtime_url(self.model),
                additional_headers={"Authorization": f"Bearer {self.api_key}"},
            )
        except Exception as e:
            print(f"# Partner: 接続失敗 ({e})", flush=True)
            return
        self._connected = True
        self._send_session_update()
        threading.Thread(target=self._recv_loop, daemon=True).start()
        self._start_playback_thread()
        print(f"# Partner: 接続完了（model={self.model}, voice={self.voice}）",
              flush=True)

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
        self._q_put(None)  # playback threadにEOSを通知（現epochタグ付き）
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

    # --- WebSocket受信 ---

    def _handle(self, ev: dict):
        etype = ev.get("type", "")

        if etype == "response.output_item.added":
            item = ev.get("item", {})
            self._current_item_id = item.get("id")
            self._played_bytes = 0
            self._play_epoch += 1  # 応答世代を進める（Bug 6）
            # 新応答の開始 → 前の中断状態を解除（response.done取りこぼし時の固着防止）
            self._interrupted = False

        elif etype == "response.output_audio.delta":
            if self._interrupted:
                return  # interrupt後の残留チャンクを破棄
            chunk = ev.get("delta", "")
            if chunk:
                self._q_put(base64.b64decode(chunk))
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
                self._q_put(None)

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
            low = msg.lower()
            if any(s in low for s in self._BENIGN_ERROR_SUBSTRINGS):
                # cancel/create と VAD 自動応答の競合など。実害がないので静かに無視。
                return
            print(f"# Partner エラー: {msg}", flush=True)
            # 想定外エラーで応答生成が中断された場合、_responding の固着を防ぐ
            # （固着すると in_echo_window が True のままになり進行が止まる）。
            if self._responding:
                self._responding = False
                self._interrupted = False

    # close() は _RealtimeBase の共通実装を使用（R3c）

"""Realtime API v2 ベースの AIエージェント."""
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
    _AGENT_TRIGGER,
    _PROMPT_CONVERSATION,
    _PROMPT_FACILITATOR,
    AGENT_VOICES,
    REALTIME_URL,
)
from .._voice_profiles import VoiceProfiles, _best_text_similarity, _resample_24_to_16


class RealtimeAgent:
    """OpenAI Realtime API v2 WebSocket で会議に参加するAIエージェント.

    エコー防止（マイク常時オン — 人間の割り込みを維持）:
      1. AI声紋フィルタ（主フィルタ）— 初回AI応答の音声から声紋を自動登録し、
         VoiceProfiles.classify()でAI声紋に一致するセグメントを除去。
         ラベル追従により、短い断片もAI扱いで除去される。
      2. テキスト類似度（安全網）— 声紋未登録時（最初の~3秒）の補助。
         エコーウィンドウ中のみテキスト類似度>0.35で除去。
      3. トリガーガード — エコーウィンドウ中はtrigger抑止（feedは即座）。
         応答生成中も新規triggerを抑止。フィードバックループの最終防衛線。

    interrupt()時はresponse.cancel + conversation.item.truncateで
    AIの会話履歴を正確に保つ。

    モード:
      off          = 無効
      facilitator  = N発話 or 沈黙でトリガー、介入不要なら黙る
      conversation = 毎発話でトリガー、必ず返答する
    """

    MODES = ("off", "facilitator", "conversation")

    def __init__(self, api_key: str, voice: str = "shimmer",
                 mode: str = "facilitator", trigger_n: int = _AGENT_TRIGGER):
        self.api_key = api_key
        self.voice = voice
        self.mode = mode                   # off / facilitator / conversation
        self.trigger_n = trigger_n
        self.ws = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pending: list[dict] = []     # 送信待ち発話
        self.ai_speaking = False           # AI音声再生中フラグ
        self._ai_text_buf = ""             # ストリーミング転写バッファ
        self._audio_q: queue.Queue[bytes | None] = queue.Queue()  # ストリーミング再生用
        self._connected = False
        self._conn_error = ""              # 接続エラーメッセージ（UI表示用）
        self.on_ai_utterance = None        # callback(text: str) AI発話確定時
        self.on_speech_start = None        # callback() 音声生成開始時（即座に通知）
        self._playback_thread: threading.Thread | None = None
        # --- エコー防止 ---
        self._responding = False           # response生成中フラグ
        self._interrupted = False          # 割り込みによるキャンセル中（残留音声を破棄）
        self._recent_ai_texts: collections.deque = collections.deque(maxlen=20)
        self._last_speech_end = 0.0        # ai_speaking が False になった時刻
        self._echo_cooldown = 2.0          # AI発話終了後のエコーウィンドウ秒数
        # --- truncate用: 再生済み音声の追跡 ---
        self._current_item_id: str | None = None    # 現在の応答のoutput item ID
        self._played_bytes = 0                       # 再生スレッドが出力したPCMバイト数
        # --- AI声紋登録用 ---
        self._voice_tracker: VoiceProfiles | None = None  # set_tracker()で外部から注入
        self._ai_voice_buf: list[np.ndarray] = []   # 16kHz float32 チャンク
        self._ai_voice_sec = 0.0                     # 蓄積秒数
        self._ai_voice_enrolled = False              # 登録済みフラグ
        # --- プリフライトバッファ（「介入不要」音声漏れ防止） ---
        self._preflight_buf: list[bytes] = []  # 再生前の音声チャンクバッファ
        self._preflight_cleared = False        # テキスト確認OK → 再生開始済み
        self._preflight_chars = 3              # この文字数まで蓄積して判定
        # --- 介入内容の保存（割り込まれても内容を失わない） ---
        self._pending_intervention: dict | None = None  # 割り込みで中断された介入内容
        self._INTERVENTION_TTL = 60.0                   # 保存した介入の有効期限（秒）
        self._INTERVENTION_MAX_RETRIES = 2              # 再試行上限

    AI_VOICE_KEY = "__AI__"             # VoiceProfiles内のAI声紋キー（セッション限り）
    _AI_ENROLL_SEC = 3.0                 # 声紋登録に必要な最小秒数

    @property
    def _prompt(self) -> str:
        return _PROMPT_CONVERSATION if self.mode == "conversation" else _PROMPT_FACILITATOR

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def set_tracker(self, tracker: VoiceProfiles):
        """VoiceProfilesを外部から注入。connect()の前後いつでも可。"""
        self._voice_tracker = tracker

    def _try_enroll_ai_voice(self):
        """蓄積したAI音声から声紋を計算しVoiceProfilesに登録する。

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
        print(f"# AI Agent: AI声紋を登録しました（{self._ai_voice_sec:.1f}秒の音声から）", flush=True)

    def connect(self):
        """WebSocket接続を開始し、受信スレッドを起動."""
        try:
            from websockets.sync.client import connect
        except ImportError:
            self._conn_error = "websockets未インストール"
            print("# AI Agent: websockets がインストールされていません", flush=True)
            return
        try:
            self.ws = connect(
                REALTIME_URL,
                additional_headers={
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
        except Exception as e:
            self._conn_error = str(e)[:80]
            print(f"# AI Agent: 接続失敗 ({e})", flush=True)
            return
        self._connected = True
        self._conn_error = ""
        self._send_session_update()
        threading.Thread(target=self._recv_loop, daemon=True).start()
        self._start_playback_thread()
        print(f"# AI Agent: 接続完了（voice={self.voice}, mode={self.mode}）", flush=True)

    def _send_session_update(self):
        """現在の設定でsession.updateを送信（GA API形式）.

        GA (gpt-realtime-2) WebSocket スキーマ:
          session.type = "realtime"           (必須)
          session.instructions               (フラット)
          session.audio.input.turn_detection  (None で VAD 無効)
          session.audio.output.voice          (ネスト)
        参照: https://developers.openai.com/api/docs/guides/realtime-conversations
        """
        if not self.ws:
            return
        try:
            self.ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "instructions": self._prompt,
                    "audio": {
                        "input": {
                            "turn_detection": None,
                        },
                        "output": {
                            "voice": self.voice,
                        },
                    },
                },
            }))
        except Exception as e:
            print(f"# AI Agent: session.update失敗 ({e})", flush=True)

    def apply_config(self, mode: str | None = None, voice: str | None = None,
                     trigger_n: int | None = None):
        """動的に設定変更（UIから呼ばれる）."""
        changed = False
        if mode is not None and mode in self.MODES and mode != self.mode:
            self.mode = mode
            changed = True
        if voice is not None and voice in AGENT_VOICES and voice != self.voice:
            self.voice = voice
            changed = True
        if trigger_n is not None and trigger_n > 0:
            self.trigger_n = trigger_n
        if changed and self._connected:
            self._send_session_update()

    # --- ストリーミング音声再生 ---

    def _start_playback_thread(self):
        """PCMキューから読み出して逐次再生するスレッド。
        再生済みバイト数を_played_bytesに蓄積（truncate用）。
        声紋未登録時は16kHzリサンプル音声を蓄積して自動登録。"""
        def _player():
            try:
                import sounddevice as sd
                stream = sd.OutputStream(samplerate=24000, channels=1,
                                         dtype="float32", blocksize=2400)
                stream.start()
                while not self._stop.is_set():
                    chunk = self._audio_q.get()
                    if chunk is None:          # 1応答の終端
                        self.ai_speaking = False
                        self._last_speech_end = time.monotonic()
                        continue
                    pcm = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
                    stream.write(pcm.reshape(-1, 1))
                    self._played_bytes += len(chunk)
                    # AI声紋登録用: 16kHzにリサンプルして蓄積
                    if not self._ai_voice_enrolled:
                        ref16 = _resample_24_to_16(pcm)
                        if len(ref16) > 0:
                            self._ai_voice_buf.append(ref16.copy())
                            self._ai_voice_sec += len(ref16) / 16000.0
                            self._try_enroll_ai_voice()
                stream.stop()
                stream.close()
            except Exception as e:
                print(f"# AI音声再生スレッド異常: {e}", flush=True)

        self._playback_thread = threading.Thread(target=_player, daemon=True)
        self._playback_thread.start()

    # --- WebSocket受信 ---

    def _recv_loop(self):
        while not self._stop.is_set():
            try:
                raw = self.ws.recv()
                ev = json.loads(raw)
            except Exception as e:
                if not self._stop.is_set():
                    self._conn_error = f"切断: {e}"[:80]
                    print(f"# AI Agent: WebSocket切断 ({e})", flush=True)
                break
            self._handle(ev)
        self._connected = False

    def _handle(self, ev: dict):
        etype = ev.get("type", "")

        if etype == "response.output_item.added":
            # 新しい出力アイテム開始 — item_idを記録、再生カウンタをリセット
            item = ev.get("item", {})
            self._current_item_id = item.get("id")
            self._played_bytes = 0
            # プリフライトバッファをリセット（新応答の開始）
            self._preflight_buf.clear()
            self._preflight_cleared = False

        elif etype == "response.output_audio.delta":
            if self._interrupted:
                return  # キャンセル後の残留チャンクを破棄
            chunk = ev.get("delta", "")
            if chunk:
                pcm = base64.b64decode(chunk)
                if self._preflight_cleared:
                    # テキスト確認済み → そのまま再生キューへ
                    self._audio_q.put(pcm)
                else:
                    # まだテキスト未確認 → バッファに溜める
                    self._preflight_buf.append(pcm)
                self.ai_speaking = True

        elif etype == "response.output_audio_transcript.delta":
            if not self._interrupted:
                self._ai_text_buf += ev.get("delta", "")
                # 「介入不要」を検出したら即座に応答をキャンセル
                if "介入不要" in self._ai_text_buf:
                    self._cancel_response()
                # プリフライト判定: 十分なテキストが来て「介入不要」でなければ再生開始
                elif (not self._preflight_cleared
                      and len(self._ai_text_buf) >= self._preflight_chars):
                    self._flush_preflight()

        elif etype == "response.output_audio_transcript.done":
            transcript = ev.get("transcript", "") or self._ai_text_buf
            self._ai_text_buf = ""
            # transcript.doneが来たのにまだプリフライト中なら確定フラッシュ
            if not self._preflight_cleared and not self._interrupted:
                if "介入不要" in (transcript or ""):
                    self._cancel_response()
                else:
                    self._flush_preflight()
            if transcript and "介入不要" not in transcript:
                self._recent_ai_texts.append(transcript)
                if not self._interrupted and self.on_ai_utterance:
                    self.on_ai_utterance(transcript)

        elif etype == "response.output_audio.done":
            if not self._interrupted:
                self._audio_q.put(None)   # 再生終端マーカー

        elif etype == "response.done":
            self._ai_text_buf = ""
            self._responding = False
            self._interrupted = False     # 次の応答に備えてリセット
            self._current_item_id = None
            self._preflight_buf.clear()
            self._preflight_cleared = False

        elif etype == "error":
            msg = ev.get("error", {}).get("message", "unknown")
            print(f"# AI Agent エラー: {msg}", flush=True)
            # エラーでresponse生成が中断された場合、_respondingをリセット
            # （固着するとtrigger()が永遠にスキップされる）
            if self._responding:
                self._responding = False
                self._interrupted = False

    # --- 発話送信 ---

    def feed(self, speaker: str, text: str, *, trigger_count: bool = True):
        """発話をエージェントに蓄積.

        trigger_count=False の場合、文脈としては送信されるが
        pending_count（trigger_n閾値判定）にはカウントしない。
        Partner発話など、文脈共有は必要だがtriggerは不要なケースで使う。
        """
        if not self._connected or not self.enabled:
            return
        with self._lock:
            self._pending.append({"speaker": speaker, "text": text,
                                  "_count": trigger_count})

    def trigger(self, *, topics: list[dict] | None = None):
        """蓄積した発話をRealtimeAPIに送信し応答を要求.

        topics: 現在の論点一覧（_topic_workerが抽出したもの）。
                渡された場合、コンテキストに含めて脱線検出の精度を上げる。
        保存された介入内容（割り込みで中断された発言）がある場合、
        コンテキストに追加して再試行の機会を与える。
        """
        if not self._connected or not self.enabled or not self.ws:
            return
        if self._responding:
            return  # 応答生成中は新規リクエストを抑止
        with self._lock:
            if not self._pending and self._pending_intervention is None:
                return
            conv = "\n".join(f"{u['speaker']}: {u['text']}" for u in self._pending)
            self._pending.clear()
        # --- 論点一覧をコンテキストに追加 ---
        if topics:
            topic_lines = "\n".join(
                f"  {i+1}. {t['topic']}（{t.get('speaker', '?')}）"
                for i, t in enumerate(topics[-8:])  # 最新8件まで
            )
            topic_note = (f"[現在の論点]\n{topic_lines}\n\n"
                          f"議論がこれらの論点からズレていたら、"
                          f"簡潔に指摘して元のテーマに戻してください。")
            conv = f"{topic_note}\n\n{conv}" if conv else topic_note
        # --- 保存された介入内容をコンテキストに追加 ---
        pi = self._pending_intervention
        if pi is not None:
            age = time.monotonic() - pi["created_at"]
            if age < self._INTERVENTION_TTL:
                retry_note = (f"[システム注記: あなたは先ほど以下の発言を試みましたが、"
                              f"参加者の発言と重なり中断されました。"
                              f"まだ重要であれば、簡潔に再度伝えてください]\n"
                              f"あなたの中断された発言: {pi['delivered']}")
                conv = f"{conv}\n\n{retry_note}" if conv else retry_note
                print("# AI Agent: 中断された介入を再試行コンテキストに追加", flush=True)
            else:
                print(f"# AI Agent: 中断された介入を期限切れで破棄（{age:.0f}秒経過）",
                      flush=True)
            self._pending_intervention = None
        if not conv:
            return  # 期限切れで破棄された場合など、送るものがない
        try:
            self.ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": conv}],
                },
            }))
            self.ws.send(json.dumps({"type": "response.create"}))
            self._responding = True
        except Exception as e:
            print(f"# AI Agent 送信エラー: {e}", flush=True)

    def interrupt(self):
        """人間の割り込みを検出。現在のAI応答をキャンセルし再生を停止する。

        response.cancelで生成を停止した後、conversation.item.truncateで
        実際に再生された分だけを会話履歴に残す。これによりAIが
        「全部喋った」と誤認して次の応答がずれるのを防ぐ。

        介入内容の保存: 割り込まれた時点の_ai_text_bufを保存し、
        次のトリガー機会で「先ほど言いかけた内容」として再利用可能にする。
        """
        if not self.ai_speaking and not self._responding:
            return
        self._interrupted = True
        # --- 介入内容の保存: 割り込まれた内容を記憶 ---
        delivered = self._ai_text_buf.strip()
        if delivered and "介入不要" not in delivered:
            existing = self._pending_intervention
            attempts = (existing["attempts"] if existing else 0) + 1
            if attempts <= self._INTERVENTION_MAX_RETRIES:
                self._pending_intervention = {
                    "delivered": delivered,
                    "created_at": time.monotonic(),
                    "attempts": attempts,
                }
                print(f"# AI Agent: 介入内容を保存（試行{attempts}回目、次の機会で再試行）",
                      flush=True)
            else:
                self._pending_intervention = None
                print("# AI Agent: 介入内容を破棄（再試行上限に達した）", flush=True)
        # --- Graceful yield: キュー内の音声を少しだけ残して自然に終了 ---
        # 24kHz 16bit PCM = 48000 bytes/sec → 300ms ≒ 14400 bytes
        _yield_keep_bytes = 14400
        played = self._played_bytes
        kept_bytes = 0
        kept_chunks: list[bytes] = []
        while True:
            try:
                chunk = self._audio_q.get_nowait()
            except queue.Empty:
                break
            if chunk is not None and kept_bytes < _yield_keep_bytes:
                kept_chunks.append(chunk)
                kept_bytes += len(chunk)
            # それ以降は破棄
        for c in kept_chunks:
            self._audio_q.put(c)
        self._audio_q.put(None)  # 終端マーカー → playback threadが停止処理
        self.ai_speaking = bool(kept_chunks)  # 残りがあれば再生中のまま
        self._responding = False
        if not kept_chunks:
            self._last_speech_end = time.monotonic()
        # Realtime APIの応答をキャンセル + 会話履歴をtruncate
        if self.ws:
            with contextlib.suppress(Exception):
                self.ws.send(json.dumps({"type": "response.cancel"}))
            # truncate: 再生済みバイト数からミリ秒を算出（24kHz, 16bit PCM）
            item_id = self._current_item_id
            if item_id:
                audio_end_ms = int(played / 2 * 1000 / 24000)  # 2bytes/sample, 24kHz
                with contextlib.suppress(Exception):
                    self.ws.send(json.dumps({
                        "type": "conversation.item.truncate",
                        "item_id": item_id,
                        "content_index": 0,
                        "audio_end_ms": audio_end_ms,
                    }))
        self._current_item_id = None
        print("# AI Agent: 割り込み検出 — 応答を中断", flush=True)

    def _flush_preflight(self):
        """プリフライトバッファの音声を再生キューに一括フラッシュ."""
        if self._preflight_cleared:
            return
        self._preflight_cleared = True
        # 音声生成開始を通知（Partner停止用）
        if not self.ai_speaking and self.on_speech_start:
            with contextlib.suppress(Exception):
                self.on_speech_start()
        for chunk in self._preflight_buf:
            self._audio_q.put(chunk)
        self._preflight_buf.clear()

    def _cancel_response(self):
        """「介入不要」応答を静かにキャンセル。音声再生を止め、会話履歴から削除する."""
        print("# AI Agent: 介入不要と判断 — 応答をキャンセル", flush=True)
        self._interrupted = True
        self._preflight_buf.clear()        # バッファも破棄
        self._preflight_cleared = False
        self._pending_intervention = None  # 介入不要の内容は再試行しない
        # 再生キューを空にして停止
        while True:
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break
        self._audio_q.put(None)
        self.ai_speaking = False
        self._responding = False
        self._ai_text_buf = ""
        # Realtime APIの応答をキャンセル + 会話履歴からこのアイテムを削除
        if self.ws:
            with contextlib.suppress(Exception):
                self.ws.send(json.dumps({"type": "response.cancel"}))
            # 介入不要の応答はtruncateではなく削除（会話履歴に残さない）
            item_id = self._current_item_id
            if item_id:
                with contextlib.suppress(Exception):
                    self.ws.send(json.dumps({
                        "type": "conversation.item.delete",
                        "item_id": item_id,
                    }))
        self._current_item_id = None

    @property
    def pending_count(self) -> int:
        """trigger_n判定に使うカウント（trigger_count=Falseのものは除外）."""
        with self._lock:
            return sum(1 for u in self._pending if u.get("_count", True))

    @property
    def in_echo_window(self) -> bool:
        """AI発話中、またはAI発話終了後のエコー残留期間中か。
        エコーウィンドウ外ではテキストフィルタを適用しない。"""
        if self.ai_speaking:
            return True
        if self._last_speech_end == 0.0:
            return False
        return time.monotonic() - self._last_speech_end < self._echo_cooldown

    def _best_similarity(self, text: str) -> float:
        return _best_text_similarity(text, list(self._recent_ai_texts),
                                     self._ai_text_buf)

    def close(self):
        self._stop.set()
        self._audio_q.put(None)
        if self._playback_thread is not None:
            self._playback_thread.join(timeout=2.0)
        if self.ws:
            with contextlib.suppress(Exception):
                self.ws.close()
        # セッション限りのAI声紋をクリーンアップ
        if self._voice_tracker is not None and self.AI_VOICE_KEY in self._voice_tracker.profiles:
            with self._voice_tracker._lock:
                self._voice_tracker.profiles.pop(self.AI_VOICE_KEY, None)
                self._voice_tracker._active_keys.discard(self.AI_VOICE_KEY)

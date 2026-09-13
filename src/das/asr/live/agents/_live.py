"""GPT-Live-1（全二重音声モデル）で介入を話すエージェント.

責務の切り分け（docs/design/live_native_plan_2026-09.md §3）:
  - 「いつ・誰に・何を」は FacilitationController が決めて `trigger()` を呼ぶ。
    本クラスはそれを GPT-Live への指示に変え、話させ、話した文字を返す。
  - 人の発話による停止と再開は GPT-Live が自分で行う（全二重）。本クラスは
    検出も再送もしない。持つのは「終了」「新しい会議」のときの再生停止だけ。
  - 参加者から直接呼びかけられたときの短い応答も GPT-Live に任せる。

GPT-Live の作法（docs/research/gpt_live_api_2026-09.md §1, §1.1 の実測）:
  - `session.start` で model / instructions / 音声形式 / 委任を決める。
    instructions と voice は開始後に変えられない → 変更はセッションの開き直し。
  - セッションの時間は入力音声の長さで進む。室内の音声を常に送り、送れない
    間は無音で時計を進める（`_start_clock`）。
  - 話させるには `session.thinking.append`（黙って読む文脈）と
    `session.instructions.append`（指示）。1件 500 トークンまで。
  - 出力音声は無音も含む連続ストリーム。音量で声の区間を切り、無音が
    `_SPEECH_END_GAP_SEC` 続いたら1発話。文字は再生に遅れて届くので、発話の
    終端で確定する。
  - `session.delegation.created` には「追加処理は不要」と返す。
"""
from __future__ import annotations

import base64
import collections
import contextlib
import json
import queue
import threading
import time
from typing import ClassVar

import numpy as np

from .._constants import _AGENT_TRIGGER, _ECHO_COOLDOWN
from .._voice_profiles import VoiceProfiles
from . import _notes
from ._base import _RealtimeBase

LIVE_URL = "wss://api.openai.com/v1/live/sessions"
LIVE_MODEL = "gpt-live-1"
LIVE_DEFAULT_VOICE = "marin"
LIVE_VOICES = ("marin", "cedar")
_OUT_RATE = 24000
_APPEND_MAX_CHARS = 600          # 500 トークンの目安（日本語）
_SPEECH_END_GAP_SEC = 1.2        # ストリーム上でこれ以上無音が続いたら発話終了
_STREAM_STALL_SEC = 3.0          # delta 自体が止まったときの保険
_NO_SPEECH_GIVEUP_SEC = 15.0     # 指示を送っても話し始めないときに諦めるまで
_SILENCE_RMS = 200.0             # int16 の RMS。これ未満は無音（約 -44 dBFS）

PROMPT_FACILITATOR = """\
あなたは対面会議の進行役AIです。日本語で話します。
基本は黙って聞きます。相槌や合いの手は出しません。
話すのは次の2つの場合だけです。
1. [指示] で始まる指示が届いたとき。その指示に沿って、1〜2文で短く話します。
2. 参加者があなた（進行役・AI・ファシリテーター）に直接呼びかけたとき。
   「はい」など一言か、聞かれたことに1文で短く答えます。
それ以外では、参加者が何を話していても口を挟みません。
参加者が話し始めたら、自分の発言の途中でもすぐにやめて聞きます。
前置きや記号は付けず、本題だけを落ち着いた口調で話します。"""

PROMPT_CONVERSATION = """\
あなたは会議に参加しているAIの参加者です。日本語で話します。
他の参加者と自然に議論してください。質問されたら答え、意見を求められたら
短く自分の考えを述べます。一度に話すのは2〜3文までにし、他の人が
話し始めたらすぐにやめて聞きます。
[指示] で始まる指示が届いたときは、その指示に沿って短く話します。
前置きや記号は付けず、本題だけを話します。"""


def _resample_16_to_24(pcm16: bytes) -> bytes:
    x = np.frombuffer(pcm16, dtype="<i2").astype(np.float32)
    if len(x) < 2:
        return b""
    n = int(len(x) * _OUT_RATE / 16000)
    y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return np.clip(y, -32768, 32767).astype("<i2").tobytes()


def _chunks(text: str, n: int) -> list[str]:
    text = text.strip()
    return [text[i:i + n] for i in range(0, len(text), n)] if text else []


class LiveAgent(_RealtimeBase):
    """GPT-Live-1 で会議に参加する進行役（`mode="conversation"` なら会話相手）."""

    MODES = ("off", "facilitator", "conversation")
    AI_VOICE_KEY = "__AI__"
    _LABEL = "AI Agent(Live)"
    _EVENT_HANDLERS: ClassVar[dict[str, str]] = {
        "session.started": "_on_session_started",
        "session.output_audio.delta": "_on_audio",
        "session.output_transcript.delta": "_on_transcript",
        "session.delegation.created": "_on_delegation",
        "session.closed": "_on_closed",
        "error": "_on_error",
    }
    debug_events: bool = False   # True なら未処理のイベント種別を1回ずつ表示（疎通確認用）

    def __init__(self, api_key: str, voice: str = LIVE_DEFAULT_VOICE,
                 mode: str = "facilitator", trigger_n: int = _AGENT_TRIGGER,
                 model: str = LIVE_MODEL):
        self.api_key = api_key
        self.model = model
        self.voice = voice if voice in LIVE_VOICES else LIVE_DEFAULT_VOICE
        self.mode = mode
        self.trigger_n = trigger_n
        self.ws = None
        self._stop = threading.Event()
        self._state_lock = threading.Lock()      # _pending / _responding を守る
        self._pending: list[dict] = []
        self.ai_speaking = False
        self._responding = False
        self._interrupted = False                # 基底の互換用。Live では常に False
        self._ai_text_buf = ""
        self._audio_q: queue.Queue[tuple[int, bytes | None]] = queue.Queue()
        self._play_epoch = 0
        self._connected = False
        self._conn_error = ""
        self.on_ai_utterance = None              # callback(text) 発話確定時
        self.on_speech_start = None              # callback() 最初の声が出たとき
        self.on_speech_end = None                # callback() 再生が終わったとき
        self._speech_started = False
        self._speak_trigger_at = 0.0
        self._last_speak_latency_ms: float | None = None
        self._playback_thread: threading.Thread | None = None
        self._recent_ai_texts: collections.deque = collections.deque(maxlen=20)
        self._last_speech_end = 0.0
        self._echo_cooldown = _ECHO_COOLDOWN
        self._played_bytes = 0
        self._voice_tracker: VoiceProfiles | None = None
        self._ai_voice_buf: list[np.ndarray] = []
        self._ai_voice_sec = 0.0
        self._ai_voice_enrolled = False
        # --- GPT-Live 固有 ---
        self._session_id: str | None = None
        self._started = threading.Event()
        self._ev_seq = 0
        self._last_audio_at = 0.0                # 声か文字が最後に届いた時刻（壁時計）
        self._audio_bytes_this_turn = 0
        self._silent_run_ms = 0                  # 発話中に続いた無音（ストリーム時間）
        self._last_input_at = 0.0
        self._seen_types: set[str] = set()
        self._reconnect_lock = threading.Lock()
        # --- WP2 で消す互換シム（workers の生成先行・再送がまだ参照する） ---
        self._pending_intervention: dict | None = None
        self._hold_playback = False

    # ------------------------------------------------------------ 状態

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def _prompt(self) -> str:
        return PROMPT_CONVERSATION if self.mode == "conversation" else PROMPT_FACILITATOR

    @property
    def in_echo_window(self) -> bool:
        """AI 発話中、または発話終了後のエコー残留期間中か（Soniox 側の除去に使う）."""
        if self.ai_speaking:
            return True
        if self._last_speech_end == 0.0:
            return False
        return (time.monotonic() - self._last_speech_end) < self._echo_cooldown

    def pending_count(self) -> int:
        with self._state_lock:
            return sum(1 for u in self._pending if u.get("_count", True))

    def reset_meeting(self):
        """「新しい会議」: 溜めた発話を捨て、話している最中なら止める（接続は維持）."""
        with self._state_lock:
            self._pending.clear()
        self.stop_playback()

    def _log_state(self, transition: str):
        print(f"# [state] {transition} "
              f"(responding={self._responding} speaking={self.ai_speaking} "
              f"epoch={self._play_epoch})", flush=True)

    # ------------------------------------------------------------ 接続

    def connect(self):
        try:
            from websockets.sync.client import connect
        except ImportError:
            self._conn_error = "websockets未インストール"
            print(f"# {self._LABEL}: websockets がインストールされていません", flush=True)
            return
        if not self.enabled:
            return
        try:
            self.ws = connect(LIVE_URL, additional_headers={
                "Authorization": f"Bearer {self.api_key}"})
        except Exception as e:
            self._conn_error = str(e)[:80]
            print(f"# {self._LABEL}: 接続失敗 ({e})", flush=True)
            return
        self._connected = True
        self._conn_error = ""
        self._started.clear()
        self._send({
            "type": "session.start",
            "session": {
                "model": self.model,
                "instructions": self._prompt,
                "audio": {"format": {"type": "audio/pcm", "rate": _OUT_RATE},
                          "output": {"voice": self.voice}},
                "delegation": {"type": "client"},
            },
        })
        threading.Thread(target=self._recv_loop, daemon=True).start()
        self._start_playback_thread()
        self._start_watchdog()
        self._start_clock()
        if not self._started.wait(timeout=10):
            self._conn_error = "session.started が届かない"
            print(f"# {self._LABEL}: セッション開始の確認が10秒以内に届きません", flush=True)
        else:
            print(f"# {self._LABEL}: 接続完了（model={self.model}, voice={self.voice}, "
                  f"mode={self.mode}）", flush=True)

    def _send(self, ev: dict) -> bool:
        ws = self.ws
        if ws is None:
            return False
        self._ev_seq += 1
        ev.setdefault("event_id", f"das_{self._ev_seq}")
        try:
            ws.send(json.dumps(ev, ensure_ascii=False))
            return True
        except Exception as e:
            print(f"# {self._LABEL} 送信エラー: {e}", flush=True)
            return False

    def _close_session(self) -> None:
        """今のセッションを閉じる（再接続と終了で共用）."""
        ws, self.ws = self.ws, None
        self._connected = False
        if ws is not None:
            with contextlib.suppress(Exception):
                ws.send(json.dumps({"type": "session.close", "event_id": "das_close"}))
                time.sleep(0.3)
            with contextlib.suppress(Exception):
                ws.close()
        self.stop_playback()

    def apply_config(self, mode: str | None = None, voice: str | None = None,
                     trigger_n: int | None = None):
        """UI からの設定変更。指示と声は開始後に変えられないので、変わるなら開き直す."""
        reopen = False
        if mode is not None and mode in self.MODES and mode != self.mode:
            self.mode = mode
            reopen = True
        if voice is not None and voice in LIVE_VOICES and voice != self.voice:
            self.voice = voice
            reopen = True
        if trigger_n is not None and trigger_n > 0:
            self.trigger_n = trigger_n
        if not reopen:
            return
        with self._reconnect_lock:
            self._close_session()
            if self.enabled:
                self._stop.clear()
                self.connect()
        print(f"# {self._LABEL}: 設定を反映（mode={self.mode} voice={self.voice}）", flush=True)

    def close(self):
        self._close_session()
        super().close()

    # ------------------------------------------------------------ 音声入力

    def feed_audio(self, pcm_16k: bytes) -> None:
        """室内の音声（16kHz PCM16）を常に送る。全二重の割り込み検出はこれが前提."""
        if not self._connected:
            return
        out = _resample_16_to_24(pcm_16k)
        if out:
            self._last_input_at = time.monotonic()
            self._send({"type": "session.input_audio.append",
                        "audio": base64.b64encode(out).decode("ascii")})

    def _start_clock(self) -> None:
        """実音声が来ない間、100ms ごとに無音を送ってセッションの時間を進める.

        GPT-Live の時間は入力音声の長さで進むので、音声が途切れると指示や文脈の
        注入が「予定」のまま実行されない。マイクが止まったときの保険。
        """
        silence = base64.b64encode(b"\x00" * (_OUT_RATE // 10 * 2)).decode("ascii")

        def _run():
            while not self._stop.is_set() and self._connected:
                time.sleep(0.1)
                if time.monotonic() - self._last_input_at >= 0.1:
                    self._last_input_at = time.monotonic()
                    self._send({"type": "session.input_audio.append", "audio": silence})
        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------ 発話の供給と指示

    def feed(self, speaker: str, text: str, *, trigger_count: bool = True):
        """発話を文脈として蓄積する（次の指示に添える）."""
        if not self._connected or not self.enabled:
            return
        with self._state_lock:
            self._pending.append({"speaker": speaker, "text": text, "_count": trigger_count})

    def trigger(self, *, topics=None, drift_reason=None, invite_target=None,
                fact_correction=None, manual_request=None, summary_focus=None,
                af_presentation=None, recent_agent_texts=None, **_ignored):
        """採択済みの介入を GPT-Live に話させる.

        文脈（直近の発話）を thinking、介入の指示を instructions として送る。
        `**_ignored` は Realtime 版にあった hold_playback / retry_intervention /
        is_retry を受け流すためのもの（WP2 で呼び出し側から消す）。
        """
        if not self._connected or not self.enabled or self.ws is None:
            return
        if self.mode == "conversation":
            return   # 会話相手は室内の音声に自分で応じる。指示は出さない
        with self._state_lock:
            if self._responding or self.ai_speaking:
                return
            has_intent = any((drift_reason, invite_target, fact_correction,
                              manual_request, summary_focus, af_presentation))
            if not self._pending and not has_intent:
                return
            self._responding = True
            snapshot = list(self._pending)
        conv = _notes.compose_trigger_notes(
            _notes.format_utterance_context(snapshot), topics=topics,
            drift_reason=drift_reason, invite_target=invite_target,
            fact_correction=fact_correction, manual_request=manual_request,
            summary_focus=summary_focus, af_presentation=af_presentation,
            recent_agent_texts=recent_agent_texts)
        directive, context = _notes.split_directive_and_context(conv)
        directive = directive or "直近の議論を踏まえ、必要なら短く一言だけ述べてください。"
        ok = True
        for chunk in _chunks(context, _APPEND_MAX_CHARS):
            ok = ok and self._send({"type": "session.thinking.append",
                                    "delegation_id": None, "content": chunk})
        self._begin_turn()
        self._last_speak_latency_ms = None
        self._speak_trigger_at = time.monotonic()
        for chunk in _chunks("[指示]\n" + directive, _APPEND_MAX_CHARS):
            ok = ok and self._send({"type": "session.instructions.append",
                                    "delegation_id": None, "content": chunk})
        if not ok:
            with self._state_lock:
                self._responding = False
            self._speak_trigger_at = 0.0
            print(f"# {self._LABEL} 送信エラー（発話は保持して次回に）", flush=True)
            return
        with self._state_lock:
            del self._pending[:len(snapshot)]
        self._log_state("→RESPONDING (指示を送信)")

    # ------------------------------------------------------------ 受信

    def _handle(self, ev: dict) -> None:
        etype = str(ev.get("type", ""))
        if self.debug_events and not etype.endswith("audio.delta") \
                and etype not in self._seen_types:
            self._seen_types.add(etype)
            print(f"# {self._LABEL} event: {json.dumps(ev, ensure_ascii=False)[:300]}", flush=True)
        name = self._EVENT_HANDLERS.get(etype)
        if name is not None:
            getattr(self, name)(ev)
        elif "error" in etype:
            self._on_error(ev)

    def _on_session_started(self, ev: dict) -> None:
        self._session_id = (ev.get("session") or {}).get("id")
        self._started.set()

    def _begin_turn(self) -> None:
        self._played_bytes = 0
        self._play_epoch += 1
        self._speech_started = False
        self._audio_bytes_this_turn = 0
        self._silent_run_ms = 0
        self._ai_text_buf = ""

    def _on_audio(self, ev: dict) -> None:
        chunk = ev.get("delta", "") or ev.get("audio", "")
        if not chunk:
            return
        pcm = base64.b64decode(chunk)
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
        voiced = rms >= _SILENCE_RMS
        chunk_ms = len(pcm) * 1000 // (_OUT_RATE * 2)
        in_speech = self.ai_speaking or self._audio_bytes_this_turn > 0
        if not voiced and not in_speech:
            return                      # 話していない間の無音は捨てる
        if voiced:
            if not self._responding and not self.ai_speaking:
                # こちらが求めていない発話（呼びかけへの応答）。受け皿を作って通す
                self._begin_turn()
                with self._state_lock:
                    self._responding = True
            self._silent_run_ms = 0
        else:
            self._silent_run_ms += chunk_ms
        self._last_audio_at = time.monotonic()
        self._audio_bytes_this_turn += len(pcm)
        if not self._speech_started:
            self._speech_started = True
            if self._speak_trigger_at:
                self._last_speak_latency_ms = round(
                    (time.monotonic() - self._speak_trigger_at) * 1000, 1)
                self._speak_trigger_at = 0.0
            self._log_state("→SPEAKING (first audio)")
            if self.on_speech_start:
                with contextlib.suppress(Exception):
                    self.on_speech_start()
        self._q_put(pcm)
        self.ai_speaking = True
        if self._silent_run_ms >= _SPEECH_END_GAP_SEC * 1000:
            self._finish_turn()

    def _on_transcript(self, ev: dict) -> None:
        self._ai_text_buf += ev.get("delta", "")
        self._last_audio_at = time.monotonic()

    def _on_delegation(self, ev: dict) -> None:
        did = (ev.get("delegation") or {}).get("id")
        if did:
            self._send({"type": "session.thinking.append", "delegation_id": did,
                        "content": "追加の処理は不要です。進行役として今の会話に短く応じてください。"})

    def _on_closed(self, ev: dict) -> None:
        print(f"# {self._LABEL}: セッション終了（{ev.get('reason')}） usage={ev.get('usage')}",
              flush=True)
        self._connected = False

    def _on_error(self, ev: dict) -> None:
        msg = (ev.get("error") or {}).get("message") or ev.get("message") or "unknown"
        print(f"# {self._LABEL} エラー: {msg}", flush=True)
        with self._state_lock:
            self._responding = False

    # ------------------------------------------------------------ 発話の終端

    def _start_watchdog(self) -> None:
        def _run():
            while not self._stop.is_set():
                time.sleep(0.1)
                if self._responding and not self.ai_speaking and self._speak_trigger_at \
                        and time.monotonic() - self._speak_trigger_at > _NO_SPEECH_GIVEUP_SEC:
                    self._speak_trigger_at = 0.0
                    with self._state_lock:
                        self._responding = False
                    self._ai_text_buf = ""
                    print(f"# {self._LABEL}: 指示から{_NO_SPEECH_GIVEUP_SEC:.0f}秒たっても"
                          "話し始めないので待つのをやめます", flush=True)
                    continue
                # 通常の終端は _on_audio がストリーム時間で決める。ここは delta 自体が
                # 止まったときの保険。
                if (self.ai_speaking or self._responding) and self._last_audio_at \
                        and time.monotonic() - self._last_audio_at > _STREAM_STALL_SEC \
                        and self._audio_bytes_this_turn > 0 and self._audio_q.empty():
                    self._finish_turn()
        threading.Thread(target=_run, daemon=True).start()

    def _finish_turn(self) -> None:
        """1発話の終わり。文字を確定して議事録へ渡し、再生の終端マーカーを流す."""
        self._last_audio_at = 0.0
        self._audio_bytes_this_turn = 0
        self._silent_run_ms = 0
        transcript = self._ai_text_buf.strip()
        self._ai_text_buf = ""
        if transcript:
            self._recent_ai_texts.append(transcript)
            if self.on_ai_utterance:
                with contextlib.suppress(Exception):
                    self.on_ai_utterance(transcript)
        self._q_put(None)
        with self._state_lock:
            self._responding = False
        self._log_state(f"→IDLE (発話終端 {len(transcript)}字)")

    # ------------------------------------------------------------ 再生の停止

    def stop_playback(self) -> None:
        """こちらの都合で再生を止める（終了・新しい会議・介入オフ）.

        人の発話による停止はモデルが行うので、ここでは呼ばない。
        """
        was = self.ai_speaking or self._responding
        while True:
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break
        self._ai_text_buf = ""
        self._audio_bytes_this_turn = 0
        self._silent_run_ms = 0
        self._last_audio_at = 0.0
        with self._state_lock:
            self._responding = False
        if self.ai_speaking:
            self._q_put(None)
            self._end_speech()
        if was:
            self._log_state("→IDLE (再生停止)")

    # --- WP2 で消す互換シム（workers / session_state がまだ呼ぶ） ---

    def interrupt(self) -> None:
        """人の発話による停止はモデルに任せる。何もしない（WP2 で呼び出しごと消す）."""
        return

    def cancel_held(self) -> None:
        return

    def release_playback(self) -> None:
        return

    @property
    def is_holding_playback(self) -> bool:
        return False

    @property
    def last_hold_to_release_ms(self) -> float | None:
        return None

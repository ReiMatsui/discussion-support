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
from ._base import _VoiceAgentBase

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
_CLOCK_IDLE_SEC = 0.3            # 入力がこの秒数ぶん壁時計に遅れたら無音で埋める
_CLOCK_REBASE_SEC = 5.0          # これ以上遅れたら（スリープ等）埋めずに基準を取り直す
_INPUT_BACKLOG_MAX = 30          # 送信待ちの室内音声（100ms 単位）。超えた古い分は捨てる（3秒）
_TRANSCRIPT_QUIET_SEC = 0.3      # 文字が止まってからでないと発話を確定しない（文字は音声に遅れる）
_RESUME_GRACE_SEC = 4.0          # 直前の発話の終端からこの秒数内の再開は同じ発話の続きとみなす
_PENDING_KEEP = 40               # 溜めておく発話の上限。指示が長く出ないと際限なく増える
_CONTEXT_MAX_UTTS = 15           # 指示に添える発話の上限（モデルは室内の音声も聞いている）

PROMPT_FACILITATOR = """\
あなたは対面会議の進行役AIです。日本語で話します。
基本は黙って聞きます。相槌や合いの手は出しません。
話すのは次の2つの場合だけです。
1. [指示] で始まる指示が届いたとき。その指示に沿って、1〜2文で短く話します。
2. 参加者があなた（進行役・AI・ファシリテーター）に直接呼びかけたとき。
   「はい」など一言か、聞かれたことに1文で短く答えます。
   議題の中に「AI」という言葉が出てくるだけでは呼びかけではありません。
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

PROMPT_PARTNER = """\
あなたは会議で人間と議論するAIの相手役です。日本語で話します。
議題: {topic}
自分の意見を持ち、根拠を添えて短く述べます。相手の意見には賛成でも反対でも
率直に応じます。一度に話すのは2〜3文までにし、相手が話し始めたらすぐに
やめて聞きます。相手が黙っているときに自分から話し続けることはしません。
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


class LiveAgent(_VoiceAgentBase):
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
        self._queued_ms = 0                      # 再生待ちの音声の長さ（ジッタ吸収の判断用）
        self._queued_lock = threading.Lock()
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
        self._input_base_at = 0.0                # 入力時計の基準（接続時）
        self._input_sent_ms = 0                  # 接続後に送った入力音声の合計
        self._seen_types: set[str] = set()
        # 開き直し（mode/voice 変更）の制御。セッションごとに世代番号を振り、
        # 古いセッションの受信スレッドが新しいセッションの状態を触らないようにする。
        self._reconnect_lock = threading.RLock()
        self._session_gen = 0
        self._aux_started = False                # 時計・監視スレッドは1回だけ起動
        self._reopening = False
        self.last_reopen_ms: float | None = None  # 直近の開き直しに要した時間（記録用）
        # 直近の発話の観測値（§3.5）。_finish_turn で確定し、議事録側が介入ログに写す
        self.last_turn_stats: dict | None = None
        self._turn_requested = False             # こちらの指示で始まった発話か
        self._turn_first_voice_at = 0.0          # 最初の声（壁時計）
        self._turn_last_voice_at = 0.0           # 最後の声（壁時計）
        self._voiced_ms_this_turn = 0            # 声の区間の合計（ストリーム時間）
        self._stream_ms_this_turn = 0            # 発話中に届いた音声の総量（無音含む）
        self.unrequested_turns = 0               # 指示なしで話した回数（会議通算）
        # こちらの都合で止めた後、モデルがまだ流している残りを捨てる（無音1.2秒で解除）
        self._discarding = False
        self._discard_silent_ms = 0
        self._last_transcript_at = 0.0           # 文字が最後に届いた時刻（壁時計）
        self._last_turn_end_at = 0.0             # 直前の発話を確定した時刻
        self._last_turn_requested = False
        self.last_unrequested_speech_at = 0.0    # 指示なしで話し始めた直近の時刻（呼びかけ応答の検出用）
        self._finish_lock = threading.Lock()     # _finish_turn の二重実行を防ぐ
        self._in_q: queue.Queue[bytes] = queue.Queue()   # 室内音声の送信待ち
        self._playback_restart_at = 0.0

    # ------------------------------------------------------------ 状態

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def ready(self) -> bool:
        """指示を送れる状態か（接続済みで session.started を受け取り、開き直し中でない）."""
        return self._connected and self._started.is_set() and not self._reopening

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

    @property
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
        """セッションを開く。開き直し中（別スレッドがロック中）なら何もしない."""
        if not self._reconnect_lock.acquire(blocking=False):
            return
        try:
            self._connect_locked()
        finally:
            self._reconnect_lock.release()

    def _connect_locked(self):
        if self._connected or not self.enabled:
            return
        try:
            from websockets.sync.client import connect
        except ImportError:
            self._conn_error = "websockets未インストール"
            print(f"# {self._LABEL}: websockets がインストールされていません", flush=True)
            return
        try:
            ws = connect(LIVE_URL, additional_headers={
                "Authorization": f"Bearer {self.api_key}"})
        except Exception as e:
            self._conn_error = str(e)[:80]
            print(f"# {self._LABEL}: 接続失敗 ({e})", flush=True)
            return
        self._session_gen += 1
        gen = self._session_gen
        self.ws = ws
        self._discarding = False      # 新しいセッションの音声は最初から通す
        self._input_base_at = time.monotonic()
        self._input_sent_ms = 0
        while True:                   # 前セッション宛ての送信待ちは捨てる
            try:
                self._in_q.get_nowait()
            except queue.Empty:
                break
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
        threading.Thread(target=self._recv_loop, args=(ws, gen), daemon=True).start()
        self._start_playback_thread()
        if not self._aux_started:
            self._aux_started = True
            self._start_watchdog()
            self._start_clock()
        if not self._started.wait(timeout=10):
            self._conn_error = "session.started が届かない"
            print(f"# {self._LABEL}: セッション開始の確認が10秒以内に届きません"
                  "（閉じて再接続を待ちます）", flush=True)
            self._close_session()
        else:
            print(f"# {self._LABEL}: 接続完了（model={self.model}, voice={self.voice}, "
                  f"mode={self.mode}）", flush=True)

    def _recv_loop(self, ws, gen: int):  # type: ignore[override]
        """1セッション分の受信ループ。世代が進んだら（開き直し）静かに終わる."""
        while not self._stop.is_set():
            try:
                raw = ws.recv()
                ev = json.loads(raw)
            except Exception as e:
                if gen == self._session_gen and not self._stop.is_set():
                    self._conn_error = f"切断: {e}"[:80]
                    print(f"# {self._LABEL}: WebSocket切断 ({e})", flush=True)
                break
            if gen != self._session_gen:
                # 閉じたセッションの残りイベント。使用量の記録だけ拾って状態は触らない
                if ev.get("type") == "session.closed":
                    self._on_closed(ev, stale=True)
                continue
            try:
                self._handle(ev)
            except Exception as e:
                # 1イベントの処理失敗で受信スレッドを死なせない。死ぬと _connected が
                # True のまま永久に沈黙する（レビュー 2026-09-13）
                print(f"# {self._LABEL}: イベント処理エラー ({ev.get('type')}: {e})", flush=True)
        if gen == self._session_gen:
            self._connected = False

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
        """今のセッションを閉じる（開き直しと終了で共用）."""
        ws, self.ws = self.ws, None
        self._session_gen += 1        # 以降、旧セッションの受信スレッドは状態を触らない
        self._connected = False
        self._started.clear()
        if ws is not None:
            with contextlib.suppress(Exception):
                ws.send(json.dumps({"type": "session.close", "event_id": "das_close"}))
                time.sleep(0.3)   # session.closed（使用量つき）を受け取る猶予
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
            self._forget_ai_voice()   # 前の声の声紋で新しい声は除去できない
            reopen = True
        if trigger_n is not None and trigger_n > 0:
            self.trigger_n = trigger_n
        if not reopen:
            return
        # 開き直しの間（実測 1〜3 秒）は ready=False になり、ワーカーは介入を出さない。
        t0 = time.monotonic()
        with self._reconnect_lock:
            self._reopening = True
            try:
                self._close_session()
                if self.enabled:
                    self._connect_locked()
            finally:
                self._reopening = False
        self.last_reopen_ms = (time.monotonic() - t0) * 1000
        print(f"# {self._LABEL}: 設定を反映（mode={self.mode} voice={self.voice} "
              f"開き直し {self.last_reopen_ms:.0f}ms）", flush=True)

    def _forget_ai_voice(self) -> None:
        """AI 声紋を次の発話から登録し直す（声の変更時）.

        古い声紋は新しい声紋で上書きされるまで残す。消してしまうと、直前に
        再生済みで Soniox の確定待ちになっている旧声のエコーが人の声として
        通ってしまう（レビュー 2026-09-13）。
        """
        self._ai_voice_enrolled = False
        self._ai_voice_buf = []
        self._ai_voice_sec = 0.0

    def close(self):
        self._close_session()
        super().close()

    # ------------------------------------------------------------ 音声入力

    def feed_audio(self, pcm_16k: bytes) -> None:
        """室内の音声（16kHz PCM16）を送信待ちに積む。全二重の割り込み検出はこれが前提.

        マイクのコールバックから呼ばれるので、ここでは WebSocket に触らない
        （送信が詰まるとコールバックが遅れ、マイク入力そのものが欠ける）。
        送るのは送信スレッド（_start_clock）。
        """
        if not self._connected or not pcm_16k:
            return
        self._in_q.put(pcm_16k)
        while self._in_q.qsize() > _INPUT_BACKLOG_MAX:
            with contextlib.suppress(queue.Empty):
                self._in_q.get_nowait()   # 溜まりすぎた古い音声は捨てる（実時間に追従）

    def _send_input_pcm24(self, out: bytes) -> None:
        self._last_input_at = time.monotonic()
        self._input_sent_ms += len(out) * 1000 // (_OUT_RATE * 2)
        self._send({"type": "session.input_audio.append",
                    "audio": base64.b64encode(out).decode("ascii")})

    def _flush_input(self, max_items: int = 100) -> int:
        """送信待ちの室内音声を送る。送った個数を返す（送信スレッドとテストから）."""
        n = 0
        while n < max_items:
            try:
                pcm = self._in_q.get_nowait()
            except queue.Empty:
                break
            out = _resample_16_to_24(pcm)
            if out and self._connected:
                self._send_input_pcm24(out)
            n += 1
        return n

    def _fill_clock(self, now: float | None = None) -> int:
        """入力の合計が壁時計に遅れていれば、その分だけ無音を送って追いつかせる.

        GPT-Live のセッション時間は入力音声の長さで進み、出力もそれに合わせて
        流れてくる。入力が実時間より遅いと、指示の実行も出力音声も遅れて
        途切れる（マイクなしのシミュレーションでは 0.3 秒ごとに 100ms しか
        送っておらず、時間が 1/3 の速さでしか進んでいなかった。2026-09-14）。
        戻り値: 送った無音の ms。
        """
        now = time.monotonic() if now is None else now
        if not self._connected or self._input_base_at == 0.0:
            return 0
        elapsed_ms = int((now - self._input_base_at) * 1000)
        deficit = elapsed_ms - self._input_sent_ms
        if deficit > _CLOCK_REBASE_SEC * 1000:
            # 長い停止（スリープ・切断）。埋めると「先行しすぎ」になるので基準を取り直す
            self._input_base_at = now
            self._input_sent_ms = 0
            return 0
        if deficit < _CLOCK_IDLE_SEC * 1000:
            return 0
        sent = 0
        chunk = b"\x00" * (_OUT_RATE // 10 * 2)          # 100ms
        while deficit - sent >= 100:
            self._send_input_pcm24(chunk)
            sent += 100
        return sent

    def _start_clock(self) -> None:
        """室内音声の送信スレッド。入力が壁時計に遅れたら無音で埋めて時間を進める."""
        def _run():
            while not self._stop.is_set():
                if not self._flush_input():
                    time.sleep(0.02)
                self._fill_clock()
        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------ 発話の供給と指示

    def feed(self, speaker: str, text: str, *, trigger_count: bool = True):
        """発話を文脈として蓄積する（次の指示に添える）."""
        if not self._connected or not self.enabled:
            return
        with self._state_lock:
            self._pending.append({"speaker": speaker, "text": text, "_count": trigger_count})
            if len(self._pending) > _PENDING_KEEP:
                del self._pending[:len(self._pending) - _PENDING_KEEP]

    def trigger(self, *, topics=None, drift_reason=None, invite_target=None,
                fact_correction=None, manual_request=None, summary_focus=None,
                af_presentation=None, recent_agent_texts=None,
                silence_sec=None) -> bool:
        """採択済みの介入を GPT-Live に話させる.

        文脈（直近の発話）を thinking、介入の指示を instructions として送る。
        戻り値: 指示を送れたか。False のとき呼び出し側は候補を消費せず次の機会を待つ
        （接続前・開き直し中・話している最中・送信失敗）。
        """
        if not self.ready or not self.enabled or self.ws is None:
            return False
        if self.mode == "conversation":
            return False   # 会話相手は室内の音声に自分で応じる。指示は出さない
        with self._state_lock:
            if self._responding or self.ai_speaking:
                return False
            has_intent = any((drift_reason, invite_target, fact_correction,
                              manual_request, summary_focus, af_presentation,
                              silence_sec))
            if not self._pending and not has_intent:
                return False
            self._responding = True
            snapshot = list(self._pending)
            self._discarding = False      # 止めた直後でも新しい指示の声は通す
        context_utts = snapshot[-_CONTEXT_MAX_UTTS:]
        conv = _notes.compose_trigger_notes(
            _notes.format_utterance_context(context_utts), topics=topics,
            drift_reason=drift_reason, invite_target=invite_target,
            fact_correction=fact_correction, manual_request=manual_request,
            summary_focus=summary_focus, af_presentation=af_presentation,
            recent_agent_texts=recent_agent_texts, silence_sec=silence_sec)
        directive, context = _notes.split_directive_and_context(conv)
        directive = directive or "直近の議論を踏まえ、必要なら短く一言だけ述べてください。"
        ok = True
        for chunk in _chunks(context, _APPEND_MAX_CHARS):
            ok = ok and self._send({"type": "session.thinking.append",
                                    "delegation_id": None, "content": chunk})
        self._begin_turn(requested=True)
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
            return False
        with self._state_lock:
            del self._pending[:len(snapshot)]
        self._log_state("→RESPONDING (指示を送信)")
        return True

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

    def _begin_turn(self, *, requested: bool) -> None:
        if self.ai_speaking:
            # 前の発話の終端マーカーがまだ再生待ち。区間を閉じてから新しい発話にする
            # （古いマーカーは epoch が進むので無視される）
            self._end_speech()
        self._played_bytes = 0
        self._play_epoch += 1
        self._speech_started = False
        self._audio_bytes_this_turn = 0
        self._silent_run_ms = 0
        self._ai_text_buf = ""
        self._turn_requested = requested
        self._turn_first_voice_at = 0.0
        self._turn_last_voice_at = 0.0
        self._voiced_ms_this_turn = 0
        self._stream_ms_this_turn = 0
        if not requested:
            self.unrequested_turns += 1
            self.last_unrequested_speech_at = time.monotonic()

    def _on_audio(self, ev: dict) -> None:
        chunk = ev.get("delta", "") or ev.get("audio", "")
        if not chunk:
            return
        pcm = base64.b64decode(chunk)
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
        voiced = rms >= _SILENCE_RMS
        chunk_ms = len(pcm) * 1000 // (_OUT_RATE * 2)
        if self._discarding:
            # こちらの都合で止めた発話の残り。声は捨て、1.2秒の無音で通常に戻る
            if voiced:
                self._discard_silent_ms = 0
            else:
                self._discard_silent_ms += chunk_ms
                if self._discard_silent_ms >= _SPEECH_END_GAP_SEC * 1000:
                    self._discarding = False
            return
        # 発話の中かどうかは「この発話で受けたバイト数」で決める。ai_speaking は
        # 再生スレッドが終端マーカーを取り出すまで残る遅れた指標なので、終端後に
        # 続く無音を「発話中」と誤認して幽霊の発話を作ってしまう（レビュー #2）。
        in_speech = self._audio_bytes_this_turn > 0
        if not voiced and not in_speech:
            return                      # 話していない間の無音は捨てる
        if voiced:
            if not self._responding:
                # こちらが求めていない発話。前の発話の終端がまだ再生中でも新しい
                # 発話として区切る。ただし直前の発話の終端からすぐの再開（文の間の
                # 長い間、人の声で止まった後の「続けます」）は同じ介入の続きとして
                # 数える。呼びかけへの応答は unrequested として数える
                now = time.monotonic()
                continuation = (self._last_turn_requested
                                and now - self._last_turn_end_at < _RESUME_GRACE_SEC)
                self._begin_turn(requested=continuation)
                with self._state_lock:
                    self._responding = True
            self._silent_run_ms = 0
            self._voiced_ms_this_turn += chunk_ms
            self._turn_last_voice_at = time.monotonic()
        else:
            self._silent_run_ms += chunk_ms
        self._last_audio_at = time.monotonic()
        self._audio_bytes_this_turn += len(pcm)
        self._stream_ms_this_turn += chunk_ms
        if not self._speech_started:
            self._speech_started = True
            self._turn_first_voice_at = time.monotonic()
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
        if self._silent_run_ms >= _SPEECH_END_GAP_SEC * 1000 \
                and time.monotonic() - self._last_transcript_at >= _TRANSCRIPT_QUIET_SEC:
            # 文字は音声に遅れて届く。無音が続いていても文字が動いている間は待つ
            self._finish_turn()

    def _on_transcript(self, ev: dict) -> None:
        self._ai_text_buf += ev.get("delta", "")
        self._last_audio_at = time.monotonic()
        self._last_transcript_at = self._last_audio_at

    def _on_delegation(self, ev: dict) -> None:
        did = (ev.get("delegation") or {}).get("id")
        if did:
            self._send({"type": "session.thinking.append", "delegation_id": did,
                        "content": "追加の処理は不要です。呼びかけへの返事でなければ何も言わず、"
                                   "呼びかけなら短く答えてください。"})

    def _on_closed(self, ev: dict, *, stale: bool = False) -> None:
        print(f"# {self._LABEL}: セッション終了（{ev.get('reason')}） usage={ev.get('usage')}",
              flush=True)
        if not stale:
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
                self._watchdog_tick()
        threading.Thread(target=_run, daemon=True).start()

    def _watchdog_tick(self) -> None:
        """監視の1周（諦め・停止したストリームの終端・再生スレッドの立て直し）."""
        if self._responding and not self.ai_speaking and self._speak_trigger_at \
                and time.monotonic() - self._speak_trigger_at > _NO_SPEECH_GIVEUP_SEC:
            self._speak_trigger_at = 0.0
            with self._state_lock:
                self._responding = False
            self._ai_text_buf = ""
            # 指示はモデル側に残っている。取り消さないと、後で静かになった
            # ときに古い内容を話し出す（次の指示と重なる）
            self._discarding = True
            self._discard_silent_ms = 0
            self._send({"type": "session.instructions.append", "delegation_id": None,
                        "content": "[指示]\n先ほどの指示は取り消します。"
                                   "次の指示があるまで黙ってください。"})
            print(f"# {self._LABEL}: 指示から{_NO_SPEECH_GIVEUP_SEC:.0f}秒たっても"
                  "話し始めないので待つのをやめます（指示を取り消し）", flush=True)
            return
        # 再生スレッドが死ぬと AI は永久に黙り、ai_speaking も戻らない
        pt = self._playback_thread
        if self._connected and (pt is None or not pt.is_alive()) \
                and time.monotonic() - self._playback_restart_at > 2.0:
            self._playback_restart_at = time.monotonic()
            print(f"# {self._LABEL}: 再生スレッドが止まっていたので立て直します", flush=True)
            self.ai_speaking = False
            self._start_playback_thread()
        # 通常の終端は _on_audio がストリーム時間で決める。ここは delta 自体が
        # 止まったときの保険。
        if (self.ai_speaking or self._responding) and self._last_audio_at \
                and time.monotonic() - self._last_audio_at > _STREAM_STALL_SEC \
                and self._audio_bytes_this_turn > 0 and self._audio_q.empty():
            self._finish_turn(end_reason="stall")

    def _finish_turn(self, *, end_reason: str = "silence") -> None:
        """1発話の終わり。文字を確定して議事録へ渡し、再生の終端マーカーを流す.

        end_reason: silence（1.2秒の無音）/ stall（ストリーム停止の保険）。
        受信スレッドと監視スレッドの両方から呼ばれうるので、1発話につき1回に絞る。
        """
        with self._finish_lock:
            if self._audio_bytes_this_turn == 0 and not self._ai_text_buf:
                return                  # もう確定済み（二重呼び出し）
            self._last_audio_at = 0.0
            self._audio_bytes_this_turn = 0
            self._silent_run_ms = 0
            transcript = self._ai_text_buf.strip()
            self._ai_text_buf = ""
        now = time.monotonic()
        self._last_turn_end_at = now
        self._last_turn_requested = self._turn_requested
        self.last_turn_stats = {
            "requested": self._turn_requested,
            "speak_start_latency_ms": self._last_speak_latency_ms if self._turn_requested else None,
            "voiced_sec": round(self._voiced_ms_this_turn / 1000, 2),
            "stream_sec": round(self._stream_ms_this_turn / 1000, 2),
            "span_sec": (round(self._turn_last_voice_at - self._turn_first_voice_at, 2)
                         if self._turn_first_voice_at else 0.0),
            "chars": len(transcript),
            "end_reason": end_reason,
            # 最後の声から確定までの経過（壁時計）。議事録側が捕捉msに換算するのに使う
            "since_last_voice_sec": (round(now - self._turn_last_voice_at, 2)
                                     if self._turn_last_voice_at else None),
            "unrequested_turns": self.unrequested_turns,
        }
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
        with self._queued_lock:
            self._queued_ms = 0
        if self._ai_text_buf.strip():
            # 止めた発話は議事録に入れないが、再生済みの分は Soniox の確定待ちで
            # まだ文字になって返ってくる。エコー照合の参照には残す
            self._recent_ai_texts.append(self._ai_text_buf.strip())
        self._ai_text_buf = ""
        self._audio_bytes_this_turn = 0
        self._silent_run_ms = 0
        self._last_audio_at = 0.0
        self._speak_trigger_at = 0.0
        with self._state_lock:
            self._responding = False
        if was:
            # モデルはまだ話し続けている。残りの音声は捨て、やめるよう伝える
            self._discarding = True
            self._discard_silent_ms = 0
            if self._connected and self.ws is not None:
                self._send({"type": "session.instructions.append", "delegation_id": None,
                            "content": "[指示]\n今の発言を直ちにやめて、次の指示があるまで黙ってください。"})
        if self.ai_speaking:
            self._q_put(None)
            self._end_speech()
        if was:
            self._log_state("→IDLE (再生停止)")


class LivePartner(LiveAgent):
    """「AIと会話」モードの相手役。GPT-Live の別セッションで室内の音声に自分で応じる.

    ファシリテーター（`LiveAgent`）とは声紋キーとラベルを分け、議事録の
    エコー除去で区別できるようにする。Controller からの指示は受けない
    （`trigger` は何もしない）。人の発話による停止と応答はモデルが行う。
    """

    AI_VOICE_KEY = "__PARTNER__"
    _LABEL = "Partner"

    def __init__(self, api_key: str, voice: str = "cedar", topic: str = "",
                 model: str = LIVE_MODEL):
        super().__init__(api_key=api_key, voice=voice, mode="conversation", model=model)
        self.topic = topic

    @property
    def _prompt(self) -> str:
        return PROMPT_PARTNER.format(topic=self.topic or "（自由）")

    def trigger(self, **_kw):
        return

    def inject_context(self, speaker: str, text: str, **_kw) -> None:
        """文字起こしを黙って読む文脈として渡す（音声を聞き取れなかったときの補い）."""
        if self._connected and text.strip():
            self._send({"type": "session.thinking.append", "delegation_id": None,
                        "content": f"{speaker}: {text.strip()}"[:_APPEND_MAX_CHARS]})


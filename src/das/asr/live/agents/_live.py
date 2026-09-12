"""GPT-Live-1（全二重音声モデル）で介入を話すエージェント（2026-09 試験導入）.

Realtime API（`RealtimeAgent`）との違いは通信の作法だけで、上流の判断は変えない。
「いつ・誰に・何を」は従来どおり FacilitationController が決め、本クラスは
採択済みの介入を GPT-Live に話させる。

GPT-Live の作法（docs/research/gpt_live_api_2026-09.md）:
  - `session.start` で model / instructions / 音声形式 / 委任方式を決める。
    instructions と voice は開始後に変えられない。
  - 話させるには `session.instructions.append`（指示）と
    `session.thinking.append`（黙って読む文脈）を送る。1件 500 トークンまで。
  - 音声は `session.output_audio.delta` で来る。終端イベントは無く、
    無音は送られてこないので「一定時間 delta が来ない」を発話の終わりとみなす。
  - 文字は `session.output_transcript.delta`（区切りは音声の都合で、文の切れ目ではない）。
  - モデルが自分で背後の処理を求めると `session.delegation.created` が来る。
    本システムは介入の内容を自前で決めるので「追加処理は不要」と返して閉じる。
  - 割り込みは本来モデルが室内の音声を聞いて自分で止まる（全二重）。ただし
    スピーカーの音がマイクに回り込む環境ではモデルが自分の声を聞くことになる
    ため、既定では AI が話している間はマイク音声を送らず（`listen="idle"`）、
    割り込みの検出は従来の STT 側の仕組み（`interrupt()`）に任せる。
    `listen="always"` にすると全二重に任せる（エコー除去がある機材向け）。

Realtime 版と同じ公開 API（connect / feed / trigger / interrupt / apply_config /
close / ai_speaking / in_echo_window …）を持つので、`--agent-engine live` で
差し替えられる。
"""
from __future__ import annotations

import base64
import contextlib
import json
import threading
import time
from typing import ClassVar

import numpy as np

from .._constants import _AGENT_TRIGGER
from ._realtime import RealtimeAgent

LIVE_URL = "wss://api.openai.com/v1/live/sessions"
LIVE_MODEL = "gpt-live-1"
LIVE_DEFAULT_VOICE = "marin"
_APPEND_MAX_CHARS = 600          # 500 トークンの目安（日本語）
_SPEECH_END_GAP_SEC = 0.7        # これ以上 delta が途切れたら発話終了とみなす
_OUT_RATE = 24000

_PROMPT_LIVE = """\
あなたは対面会議の進行役AIです。日本語で話します。
基本は黙って聞きます。相槌や合いの手は出しません。
話すのは次の2つの場合だけです。
1. [指示] で始まる指示が届いたとき。その指示に沿って、1〜2文で短く話します。
2. 参加者があなた（進行役・AI）に直接呼びかけて質問したとき。短く答えます。
それ以外では、参加者が何を話していても口を挟みません。
前置きや記号は付けず、本題だけを落ち着いた口調で話します。"""


def _resample_16_to_24(pcm16: bytes) -> bytes:
    x = np.frombuffer(pcm16, dtype="<i2").astype(np.float32)
    if len(x) < 2:
        return b""
    n = int(len(x) * _OUT_RATE / 16000)
    y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return np.clip(y, -32768, 32767).astype("<i2").tobytes()


class LiveAgent(RealtimeAgent):
    """GPT-Live-1 で会議に参加するファシリテーター（RealtimeAgent と同じ公開 API）."""

    _LABEL = "AI Agent(Live)"
    _EVENT_HANDLERS: ClassVar[dict[str, str]] = {
        "session.started": "_on_session_started",
        "session.output_audio.delta": "_on_live_audio",
        "session.output_transcript.delta": "_on_live_transcript",
        "session.input_transcript.delta": "_on_live_input_transcript",
        "session.delegation.created": "_on_delegation",
        "session.closed": "_on_closed",
        "error": "_on_error",
    }

    def __init__(self, api_key: str, voice: str = LIVE_DEFAULT_VOICE,
                 mode: str = "facilitator", trigger_n: int = _AGENT_TRIGGER,
                 model: str = LIVE_MODEL, listen: str = "idle"):
        super().__init__(api_key=api_key, voice=voice, mode=mode,
                         trigger_n=trigger_n, model=model)
        self.listen = listen                 # "idle" | "always" | "never"
        self._session_id: str | None = None
        self._started = threading.Event()
        self._last_audio_at = 0.0
        self._audio_bytes_this_turn = 0
        self._ev_seq = 0
        self._watchdog: threading.Thread | None = None
        self._last_input_at = 0.0
        self._clock: threading.Thread | None = None
        self._silent_run_ms = 0          # 発話中に続いた無音の長さ（ストリーム時間）

    # ------------------------------------------------------------ 接続

    def connect(self):
        try:
            from websockets.sync.client import connect
        except ImportError:
            self._conn_error = "websockets未インストール"
            print(f"# {self._LABEL}: websockets がインストールされていません", flush=True)
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
        self._send({
            "type": "session.start",
            "session": {
                "model": self.model,
                "instructions": _PROMPT_LIVE,
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
                  f"listen={self.listen}）", flush=True)

    def _send(self, ev: dict) -> bool:
        if not self.ws:
            return False
        self._ev_seq += 1
        ev.setdefault("event_id", f"das_{self._ev_seq}")
        try:
            self.ws.send(json.dumps(ev, ensure_ascii=False))
            return True
        except Exception as e:
            print(f"# {self._LABEL} 送信エラー: {e}", flush=True)
            return False

    def _send_session_update(self):
        """GPT-Live は開始後に instructions / voice を変えられない。モードは
        本クラス内の判断（trigger の使い方）にだけ効く。"""
        return

    # ------------------------------------------------------------ 音声入力

    def feed_audio(self, pcm_16k: bytes) -> None:
        """室内の音声を渡す（16kHz PCM16）。listen の設定に従って送る.

        送らない場合も時計は止めない（`_start_clock` が無音を送る）。GPT-Live の
        時間は入力音声の長さで進むので、音声を送らないと指示や文脈の注入が
        いつまでも「予定」のまま実行されない（2026-09-12 の疎通で確認:
        音声ゼロだと20秒待っても話さず、close 時に "closed before the estimated
        context injection completed" が返り usage も 0 秒だった）。
        """
        if not self._connected or self.listen == "never":
            return
        if self.listen == "idle" and (self.ai_speaking or self.in_echo_window):
            return
        out = _resample_16_to_24(pcm_16k)
        if out:
            self._last_input_at = time.monotonic()
            self._send({"type": "session.input_audio.append",
                        "audio": base64.b64encode(out).decode("ascii")})

    def _start_clock(self) -> None:
        """実音声を送っていない間、100ms ごとに無音を送ってセッションの時間を進める."""
        silence = base64.b64encode(b"\x00" * (_OUT_RATE // 10 * 2)).decode("ascii")

        def _run():
            while not self._stop.is_set() and self._connected:
                time.sleep(0.1)
                if time.monotonic() - self._last_input_at >= 0.1:
                    self._last_input_at = time.monotonic()
                    self._send({"type": "session.input_audio.append", "audio": silence})
        self._clock = threading.Thread(target=_run, daemon=True)
        self._clock.start()

    # ------------------------------------------------------------ 送信

    def _send_trigger(self, conv: str, *, hold_playback, pending_snapshot,
                      active_retryable, retry_fallback, pi, include_pi) -> None:
        """組み立てた文脈を thinking、指示を instructions として送る.

        `conv` は RealtimeAgent が作る「[指示類]…\\n\\n[参加者発話]…」の文字列。
        [参加者発話] 以降は黙って読む文脈（thinking）、それより前は指示
        （instructions）として分けて送る。どちらも 500 トークン上限があるので
        文字数で刻む。
        """
        head, sep, tail = conv.partition("[参加者発話]")
        context = (sep + tail) if sep else ""
        directive = head.strip() or "直近の議論を踏まえ、必要なら短く一言だけ述べてください。"
        ok = True
        for chunk in _chunks(context, _APPEND_MAX_CHARS):
            ok = ok and self._send({"type": "session.thinking.append",
                                    "delegation_id": None, "content": chunk})
        if hold_playback:
            with self._state_lock:
                self._hold_playback = True
                self._held_audio = []
                self._hold_start_at = time.monotonic()
                self._last_hold_to_release_ms = None
        self._last_speak_latency_ms = None
        self._speak_trigger_at = time.monotonic()
        self._begin_turn()
        for chunk in _chunks("[指示]\n" + directive, _APPEND_MAX_CHARS):
            ok = ok and self._send({"type": "session.instructions.append",
                                    "delegation_id": None, "content": chunk})
        if not ok:
            self._responding = False
            self._speak_trigger_at = 0.0
            print(f"# {self._LABEL} 送信エラー（内容を保持して再試行）", flush=True)
            return
        with self._state_lock:
            self._active_intervention_retryable = active_retryable
            self._active_intervention_fallback = retry_fallback
            del self._pending[:len(pending_snapshot)]
            if include_pi and self._pending_intervention is pi:
                self._pending_intervention = None
        self._log_state("→RESPONDING (Live: instructions.append)")

    def _begin_turn(self) -> None:
        """新しい発話の受け皿を用意する（Realtime の output_item.added に相当）."""
        self._current_item_id = None
        self._played_bytes = 0
        self._play_epoch += 1
        self._speech_started = False
        self._interrupted = False
        self._audio_bytes_this_turn = 0
        self._silent_run_ms = 0

    # ------------------------------------------------------------ 受信

    debug_events: bool = False   # True なら未処理のイベント種別を1回ずつ表示（疎通確認用）
    _seen_types: set[str] | None = None

    def _handle(self, ev: dict) -> None:
        etype = str(ev.get("type", ""))
        if self.debug_events and not etype.endswith("audio.delta"):
            if self._seen_types is None:
                self._seen_types = set()
            if etype not in self._seen_types:
                self._seen_types.add(etype)
                body = json.dumps(ev, ensure_ascii=False)
                print(f"# {self._LABEL} event: {body[:300]}", flush=True)
        if "error" in etype and etype not in self._EVENT_HANDLERS:
            self._on_error(ev)
            return
        super()._handle(ev)

    def _on_session_started(self, ev: dict) -> None:
        self._session_id = (ev.get("session") or {}).get("id") or ev.get("session_id")
        self._started.set()

    _SILENCE_RMS = 200.0        # int16 の RMS。これ未満は無音とみなす（約 -44 dBFS）
    _NO_SPEECH_GIVEUP_SEC = 15.0   # 指示を送っても話し始めないときに諦めるまで

    def _on_live_audio(self, ev: dict) -> None:
        """出力音声は途切れない連続ストリームとして届く（無音も含む）.

        文書には「無音は省かれる」とあるが、実測（2026-09-12）では話していない間も
        実時間で無音の delta が届き続けた。したがって「delta が途切れた」では
        発話の終わりを判定できず、音量で話している区間を切り出す。無音の delta は
        発話中だけ（文の間の息継ぎとして）再生キューへ流し、それ以外は捨てる。
        """
        if self._interrupted:
            return
        chunk = ev.get("delta", "") or ev.get("audio", "")
        if not chunk:
            return
        pcm = base64.b64decode(chunk)
        x = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
        voiced = rms >= self._SILENCE_RMS
        chunk_ms = len(pcm) * 1000 // (_OUT_RATE * 2)
        in_speech = self.ai_speaking or self._audio_bytes_this_turn > 0
        if not voiced and not in_speech:
            return                      # 話していない間の無音は捨てる
        if voiced:
            if not self._responding and not self.ai_speaking:
                # こちらが求めていない発話（呼びかけへの返答など）。受け皿を作って通す
                self._begin_turn()
                self._responding = True
            self._silent_run_ms = 0
        else:
            self._silent_run_ms += chunk_ms
        self._last_audio_at = time.monotonic()
        self._audio_bytes_this_turn += len(pcm)
        self._on_audio_delta({"delta": chunk})
        # 終端は「ストリーム上の時間」で測る。届く間隔（壁時計）はネットワークや
        # CPU の都合で 1 秒以上空くことがあり、それを終端と見ると1つの発話が
        # 「テスト／段階での／フィードバック／…」のように細切れになる
        # （2026-09-12 のシミュレーション実走で発生）。無音の delta が連続して
        # _SPEECH_END_GAP_SEC 分たまったときだけ閉じる。
        if self._silent_run_ms >= _SPEECH_END_GAP_SEC * 1000:
            self._finish_turn()

    def _on_live_transcript(self, ev: dict) -> None:
        if not self._interrupted:
            self._ai_text_buf += ev.get("delta", "")
            self._last_audio_at = time.monotonic()   # 文字も「まだ続いている」合図

    def _on_live_input_transcript(self, ev: dict) -> None:
        return   # 室内の文字起こしは Soniox 側が正本。ここでは使わない

    def _on_delegation(self, ev: dict) -> None:
        did = (ev.get("delegation") or {}).get("id")
        if did:
            self._send({"type": "session.thinking.append", "delegation_id": did,
                        "content": "追加の処理は不要です。進行役として今の会話に短く応じてください。"})

    def _on_closed(self, ev: dict) -> None:
        usage = ev.get("usage")
        print(f"# {self._LABEL}: セッション終了（{ev.get('reason')}） usage={usage}", flush=True)
        self._connected = False

    def _on_error(self, ev: dict) -> None:
        msg = (ev.get("error") or {}).get("message") or ev.get("message") or "unknown"
        print(f"# {self._LABEL} エラー: {msg}", flush=True)
        if self._responding:
            self._responding = False
            self._interrupted = False

    # ------------------------------------------------------------ 発話終端の監視

    def _start_watchdog(self) -> None:
        def _run():
            while not self._stop.is_set():
                time.sleep(0.1)
                # 指示を送ったのに話し始めない → 諦めて次の介入に備える
                if self._responding and not self.ai_speaking and self._speak_trigger_at \
                        and time.monotonic() - self._speak_trigger_at > self._NO_SPEECH_GIVEUP_SEC:
                    self._speak_trigger_at = 0.0
                    self._responding = False
                    self._ai_text_buf = ""
                    print(f"# {self._LABEL}: 指示から{self._NO_SPEECH_GIVEUP_SEC:.0f}秒たっても"
                          "話し始めないので待つのをやめます", flush=True)
                    continue
                # 終端の条件: 音声も文字も一定時間届かず、かつ再生が追いついている。
                # モデルは1発話分をまとめて先に送ってくることがあり、再生中に
                # 文字の delta が遅れて届く。届いていない段階で閉じると文字が
                # 空のまま確定してしまう（2026-09-12 の疎通で発生）。
                # 通常の終端は _on_live_audio がストリーム時間で決める。ここは
                # ストリーム自体が止まった（delta が3秒来ない）ときの保険。
                if (self.ai_speaking or self._responding) and self._last_audio_at \
                        and time.monotonic() - self._last_audio_at > 3.0 \
                        and self._audio_bytes_this_turn > 0 \
                        and self._audio_q.empty():
                    self._finish_turn()
        self._watchdog = threading.Thread(target=_run, daemon=True)
        self._watchdog.start()

    def _finish_turn(self) -> None:
        """delta が途切れた → 1発話の終わり。文字を確定し、終端マーカーを流す."""
        self._last_audio_at = 0.0
        self._audio_bytes_this_turn = 0
        self._silent_run_ms = 0
        transcript = self._ai_text_buf.strip()
        self._ai_text_buf = ""
        if transcript:
            self._recent_ai_texts.append(transcript)
            if not self._interrupted and self.on_ai_utterance:
                with contextlib.suppress(Exception):
                    self.on_ai_utterance(transcript)
        if self._hold_playback:
            self._held_audio.append((self._play_epoch, None))
        else:
            self._q_put(None)
        self._responding = False
        self._interrupted = False
        self._active_intervention_fallback = ""
        self._log_state(f"→IDLE (Live: 発話終端 {len(transcript)}字)")

    # ------------------------------------------------------------ 割り込み

    def interrupt(self):
        """人間の割り込み。再生を止め、モデルにも黙るよう伝える.

        Realtime の response.cancel / truncate に相当するものは無い。
        再生キューを捨てて ai_speaking を倒し（親クラスの処理）、モデルには
        指示で「今すぐ話をやめて聞く」を送る。listen="always" ならモデル自身も
        室内の声で止まる。
        """
        was_active = self.ai_speaking or self._responding
        # 親の処理のうち WebSocket へ送る部分（response.cancel 等）は ws を
        # 一時的に外して抑止する
        ws, self.ws = self.ws, None
        try:
            super().interrupt()
        finally:
            self.ws = ws
        if was_active:
            self._last_audio_at = 0.0
            self._audio_bytes_this_turn = 0
            self._send({"type": "session.instructions.append", "delegation_id": None,
                        "content": "[指示] 参加者が話し始めました。今の発言を直ちにやめ、黙って聞いてください。"})

    # ------------------------------------------------------------ 終了

    def close(self):
        if self.ws:
            with contextlib.suppress(Exception):
                self._send({"type": "session.close"})
                time.sleep(0.3)
        super().close()


def _chunks(text: str, n: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [text[i:i + n] for i in range(0, len(text), n)]

"""LiveAgent（GPT-Live-1）の単体テスト。偽の WebSocket で通信の作法を確かめる.

実測で分かった GPT-Live の挙動（docs/research/gpt_live_api_2026-09.md §1.1）を
前提にしている: 出力音声は無音も含む連続ストリーム、文字は再生に遅れて届く、
指示と文脈は別々の append で送る。
"""
from __future__ import annotations

import base64
import json
import time

import numpy as np
import pytest

from das.asr.live.agents import _live
from das.asr.live.agents._live import LiveAgent


class FakeWS:
    def __init__(self):
        self.sent: list[dict] = []
        self.closed = False

    def send(self, msg: str):
        self.sent.append(json.loads(msg))

    def close(self):
        self.closed = True

    def types(self) -> list[str]:
        return [m["type"] for m in self.sent]


def _voiced(ms: int = 100) -> str:
    n = _live._OUT_RATE * ms // 1000
    pcm = (np.sin(np.arange(n) * 0.3) * 8000).astype("<i2").tobytes()
    return base64.b64encode(pcm).decode()


def _silent(ms: int = 100) -> str:
    return base64.b64encode(b"\x00" * (_live._OUT_RATE * ms // 1000 * 2)).decode()


def _settle(agent):
    """文字待ちの発話を今ある文字で確定させる（本番では遅れ分を待ってから確定）."""
    agent._flush_texts(force=True)


@pytest.fixture
def agent(monkeypatch):
    # スレッド（再生・時計・監視）は起動しない。受信は _handle を直接呼ぶ
    monkeypatch.setattr(LiveAgent, "_start_playback_thread", lambda self: None)
    monkeypatch.setattr(LiveAgent, "_start_clock", lambda self: None)
    monkeypatch.setattr(LiveAgent, "_start_watchdog", lambda self: None)
    # 文字の静止待ち（0.3秒）は専用テストで見る。他のテストは即時に確定させる
    monkeypatch.setattr(_live, "_TRANSCRIPT_QUIET_SEC", 0.0)
    a = LiveAgent(api_key="k")
    a.ws = FakeWS()
    a._connected = True
    a._started.set()
    a.said: list[str] = []
    a.on_ai_utterance = a.said.append
    return a


def test_trigger_sends_context_as_thinking_and_directive_as_instructions(agent):
    agent.feed("Aさん", "今日は名前を決めます")
    agent.feed("Bさん", "候補は三つです")
    agent.trigger(invite_target="C")
    types = agent.ws.types()
    assert types == ["session.thinking.append", "session.instructions.append"]
    thinking, instr = agent.ws.sent
    assert "Aさん: 今日は名前を決めます" in thinking["content"]
    assert instr["content"].startswith("[指示]")
    assert "Cさんに" in instr["content"]
    assert "[参加者発話]" not in instr["content"]
    assert agent._responding is True
    assert agent.pending_count == 0          # 送った分は消費される


def test_trigger_is_skipped_while_speaking_or_without_intent(agent):
    agent.trigger()                              # 発話も意図もない → 何も送らない
    assert agent.ws.sent == []
    agent.feed("A", "x")
    agent.ai_speaking = True
    agent.trigger(invite_target="B")
    assert agent.ws.sent == []                   # 話している間は送らない
    assert agent.pending_count == 1            # 発話は保持される


def test_long_context_is_chunked_under_the_token_limit(agent):
    agent.feed("A", "あ" * 1500)
    agent.trigger(summary_focus="要点")
    thinking = [m for m in agent.ws.sent if m["type"] == "session.thinking.append"]
    assert len(thinking) >= 3
    assert all(len(m["content"]) <= _live._APPEND_MAX_CHARS for m in thinking)


def test_silence_outside_speech_is_dropped_and_speech_is_played(agent):
    agent._handle({"type": "session.output_audio.delta", "delta": _silent()})
    assert agent._audio_q.empty() and not agent.ai_speaking
    agent._responding = True
    agent._speak_trigger_at = time.monotonic() - 0.5
    agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})
    assert agent.ai_speaking
    assert agent._audio_q.qsize() == 1
    assert agent._last_speak_latency_ms is not None and agent._last_speak_latency_ms >= 400


def test_turn_ends_on_stream_silence_not_wall_clock(agent):
    agent._responding = True
    for _ in range(5):
        agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})
    agent._handle({"type": "session.output_transcript.delta", "delta": "テスト"})
    for _ in range(4):                           # 400ms の無音: まだ続く
        agent._handle({"type": "session.output_audio.delta", "delta": _silent()})
    time.sleep(0.3)                              # 受信の空白は終端ではない
    for _ in range(3):
        agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})
    agent._handle({"type": "session.output_transcript.delta", "delta": "段階での"})
    assert agent.said == []
    for _ in range(int(_live._SPEECH_END_GAP_SEC * 10)):
        agent._handle({"type": "session.output_audio.delta", "delta": _silent()})
    assert agent._responding is False
    _settle(agent)
    assert agent.said == ["テスト段階での"]     # 1発話にまとまる
    # 終端マーカーが再生キューの最後に入る
    items = []
    while not agent._audio_q.empty():
        items.append(agent._audio_q.get_nowait())
    assert items[-1][1] is None


def test_unrequested_speech_gets_its_own_turn(agent):
    """呼びかけへの応答（こちらの指示なし）も1発話として記録される."""
    agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})
    assert agent._responding and agent.ai_speaking
    agent._handle({"type": "session.output_transcript.delta", "delta": "はい"})
    for _ in range(int(_live._SPEECH_END_GAP_SEC * 10)):
        agent._handle({"type": "session.output_audio.delta", "delta": _silent()})
    _settle(agent)
    assert agent.said == ["はい"]


def test_delegation_is_answered_with_no_op(agent):
    agent._handle({"type": "session.delegation.created",
                   "delegation": {"id": "item_1", "target": "client"}})
    assert agent.ws.sent[-1]["type"] == "session.thinking.append"
    assert agent.ws.sent[-1]["delegation_id"] == "item_1"


def test_stop_playback_clears_queue_and_flags(agent):
    agent._responding = True
    agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})
    agent._handle({"type": "session.output_transcript.delta", "delta": "途中"})
    agent.stop_playback()
    assert agent.ai_speaking is False and agent._responding is False
    assert agent._ai_text_buf == ""
    assert agent.said == []                      # 止めた発話は議事録に入れない


def test_apply_config_reopens_session_only_when_prompt_or_voice_changes(agent, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(LiveAgent, "_connect_locked", lambda self: calls.append("connect"))
    old_ws = agent.ws
    agent.apply_config(trigger_n=5)
    assert calls == [] and agent.ws is old_ws    # trigger_n だけなら開き直さない
    agent.apply_config(mode="conversation")
    assert calls == ["connect"]
    assert old_ws.closed and old_ws.sent[-1]["type"] == "session.close"
    assert agent.last_reopen_ms is not None and agent.last_reopen_ms >= 0
    agent.apply_config(mode="off")
    assert calls == ["connect"]                  # off は閉じるだけ


def test_reopen_generation_ignores_stale_session_events(agent, monkeypatch):
    """開き直しの後、古いセッションの受信スレッドが新しい接続状態を壊さない."""
    monkeypatch.setattr(LiveAgent, "_connect_locked", lambda self: None)
    old_gen = agent._session_gen
    agent.apply_config(voice="cedar")            # 世代が進む
    assert agent._session_gen > old_gen
    # 新しいセッションが立ったとみなす
    agent.ws = FakeWS()
    agent._connected = True
    agent._started.set()
    # 旧セッションの session.closed が遅れて届いても切断扱いにしない
    agent._on_closed({"type": "session.closed", "reason": "client", "usage": {}}, stale=True)
    assert agent._connected and agent.ready


def test_trigger_waits_until_session_started(agent):
    agent._started.clear()                       # 接続はしたが session.started 前
    agent.feed("A", "x")
    agent.trigger(invite_target="B")
    assert agent.ws.sent == []
    assert agent.pending_count == 1
    agent._started.set()
    agent.trigger(invite_target="B")
    assert agent.ws.types() == ["session.thinking.append", "session.instructions.append"]


def test_recv_loop_from_old_generation_does_not_clear_connected(agent):
    """旧世代の受信ループが例外で終わっても、新世代の _connected は残る."""
    class DeadWS:
        def recv(self):
            raise RuntimeError("closed")
    stale_gen = agent._session_gen
    agent._session_gen += 1                      # 開き直し済み
    agent._connected = True
    agent._recv_loop(DeadWS(), stale_gen)
    assert agent._connected is True
    agent._recv_loop(DeadWS(), agent._session_gen)   # 現世代の切断は反映される
    assert agent._connected is False


def test_conversation_mode_does_not_send_directives(agent):
    agent.mode = "conversation"
    agent.feed("A", "どう思う？")
    agent.trigger(invite_target="B")
    assert agent.ws.sent == []


def test_feed_audio_resamples_to_24k(agent):
    agent.feed_audio(b"\x00\x00" * 1600)         # 100ms @16k
    assert agent.ws.sent == []                   # コールバックでは送らない（積むだけ）
    assert agent._flush_input() == 1             # 送信スレッドが送る
    msg = agent.ws.sent[-1]
    assert msg["type"] == "session.input_audio.append"
    assert len(base64.b64decode(msg["audio"])) == 2400 * 2


# --- LivePartner: 「AIと会話」の相手役（GPT-Live の別セッション） -------------


def test_partner_uses_its_own_voice_key_and_topic_prompt(monkeypatch):
    from das.asr.live.agents._live import LivePartner
    monkeypatch.setattr(LiveAgent, "_start_playback_thread", lambda self: None)
    monkeypatch.setattr(LiveAgent, "_start_clock", lambda self: None)
    monkeypatch.setattr(LiveAgent, "_start_watchdog", lambda self: None)
    p = LivePartner(api_key="k", topic="AIツール導入の是非")
    assert p.AI_VOICE_KEY == "__PARTNER__" and p.AI_VOICE_KEY != LiveAgent.AI_VOICE_KEY
    assert p.mode == "conversation"
    assert "AIツール導入の是非" in p._prompt
    p.ws = FakeWS()
    p._connected = True
    p._started.set()
    p.feed("人間", "どう思う？")
    p.trigger(invite_target="B")                 # 指示は受けない
    assert p.ws.sent == []
    p.inject_context("ファシリテーター", "本題に戻しましょう")
    assert p.ws.sent[-1]["type"] == "session.thinking.append"
    assert "本題に戻しましょう" in p.ws.sent[-1]["content"]


# --- 発話の観測値（§3.5）: 介入ログに写す last_turn_stats ------------------------


def test_turn_stats_for_requested_speech(agent):
    agent.feed("A", "x")
    agent.trigger(invite_target="B")
    agent._speak_trigger_at = time.monotonic() - 0.8
    for _ in range(6):
        agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})
    agent._handle({"type": "session.output_transcript.delta", "delta": "Bさんはどうですか"})
    for _ in range(int(_live._SPEECH_END_GAP_SEC * 10)):
        agent._handle({"type": "session.output_audio.delta", "delta": _silent()})
    _settle(agent)
    st = agent.last_turn_stats
    assert st["requested"] is True
    assert st["speak_start_latency_ms"] >= 700
    assert st["voiced_sec"] == 0.6 and st["chars"] == 9
    assert st["end_reason"] == "silence" and st["unrequested_turns"] == 0


def test_turn_stats_count_unrequested_speech(agent):
    for _ in range(3):
        agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})
    agent._handle({"type": "session.output_transcript.delta", "delta": "はい"})
    agent._finish_turn(end_reason="stall")
    _settle(agent)
    st = agent.last_turn_stats
    assert st["requested"] is False and st["speak_start_latency_ms"] is None
    assert st["end_reason"] == "stall" and st["unrequested_turns"] == 1
    assert agent.said == ["はい"]


# --- レビュー（2026-09-13）で見つかった欠陥の再発防止 ------------------------------


def test_pending_count_is_a_value_not_a_method(agent):
    """Controller は pending_count を数値として比較する（メソッドだと毎 tick 落ちる）."""
    assert isinstance(LiveAgent.pending_count, property)
    agent.feed("A", "x")
    assert agent.pending_count > 0


def test_trailing_silence_after_turn_end_does_not_start_a_ghost_turn(agent):
    """終端後、再生スレッドが終端マーカーを取り出す前に届く無音で発話を再開しない."""
    agent._responding = True
    for _ in range(3):
        agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})
    agent._handle({"type": "session.output_transcript.delta", "delta": "一言"})
    for _ in range(int(_live._SPEECH_END_GAP_SEC * 10)):
        agent._handle({"type": "session.output_audio.delta", "delta": _silent()})
    _settle(agent)
    assert agent.said == ["一言"]
    assert agent.ai_speaking is True                 # 再生スレッドがまだ終端を取り出していない
    finished = agent.last_turn_stats
    for _ in range(60):                               # 6秒の無音が流れ続ける
        agent._handle({"type": "session.output_audio.delta", "delta": _silent()})
    _settle(agent)
    assert agent.said == ["一言"]
    assert agent.last_turn_stats is finished          # 0字の幽霊ターンで上書きされない
    assert agent._responding is False


def test_stop_playback_discards_the_rest_of_the_model_turn(agent):
    """こちらの都合で止めた後、モデルが流し続ける残りは再生も記録もしない."""
    agent._responding = True
    agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})
    agent.stop_playback()
    assert agent.ws.sent[-1]["type"] == "session.instructions.append"
    assert "やめて" in agent.ws.sent[-1]["content"]
    for _ in range(5):                                # 残りの声
        agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})
    agent._handle({"type": "session.output_transcript.delta", "delta": "…の続き"})
    assert agent.ai_speaking is False and agent._responding is False
    assert all(c is None for _, c in list(agent._audio_q.queue))   # 終端マーカーだけ
    assert agent.unrequested_turns == 0
    for _ in range(int(_live._SPEECH_END_GAP_SEC * 10)):
        agent._handle({"type": "session.output_audio.delta", "delta": _silent()})
    assert agent._discarding is False                 # 1.2秒の無音で通常に戻る
    agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})
    assert agent.ai_speaking is True and agent.unrequested_turns == 1


def test_trigger_reports_whether_it_could_send(agent):
    agent.feed("A", "x")
    agent.ai_speaking = True
    assert agent.trigger(invite_target="B") is False
    agent.ai_speaking = False
    assert agent.trigger(invite_target="B") is True


def test_voice_change_reenrolls_the_ai_voiceprint_but_keeps_the_old_one(agent, monkeypatch):
    monkeypatch.setattr(LiveAgent, "_connect_locked", lambda self: None)

    class Tracker:
        def __init__(self):
            import threading
            self.profiles = {LiveAgent.AI_VOICE_KEY: object()}
            self._active_keys = {LiveAgent.AI_VOICE_KEY}
            self._lock = threading.Lock()
    agent._voice_tracker = Tracker()
    agent._ai_voice_enrolled = True
    agent._ai_voice_sec = 3.0
    agent.apply_config(voice="cedar")
    assert agent._ai_voice_enrolled is False and agent._ai_voice_sec == 0.0
    # 旧声の声紋は新声で上書きされるまで残す（確定待ちの旧声エコーを落とすため）
    assert LiveAgent.AI_VOICE_KEY in agent._voice_tracker.profiles


# --- セルフレビュー（2026-09-13）で直した挙動 ------------------------------------


def test_late_transcript_still_lands_in_the_finished_utterance(agent):
    """音声の終端のあとに届いた文字も、遅れ分の窓の中ならその発話の文字になる.

    音響リハーサル（2026-09-14）では文字が音声より 1.5 秒以上遅れ、終端で確定した
    あとに届いた分が次の発話の開始で捨てられて「などが考」「せて従業員へ」の
    ように行が欠けていた。
    """
    agent._responding = True
    for _ in range(3):
        agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})
    for _ in range(int(_live._SPEECH_END_GAP_SEC * 10)):
        agent._handle({"type": "session.output_audio.delta", "delta": _silent()})
    assert agent._responding is False and agent.said == []   # 音声は終わったが文字待ち
    agent._handle({"type": "session.output_transcript.delta", "delta": "遅れて届く"})
    agent._handle({"type": "session.output_transcript.delta", "delta": "文字"})
    assert agent.said == []                          # まだ遅れの窓の中
    agent._flush_texts(agent._turns[0]["last"] + agent._text_lag + _live._TEXT_TAIL_SEC + 0.1)
    assert agent.said == ["遅れて届く文字"]


def test_transcript_arriving_after_the_next_turn_started_goes_to_the_right_turn(agent):
    """前の発話の末尾の文字が、次の発話の声が出たあとに届いても前の発話に付く."""
    agent._text_lag = 1.0
    t0 = time.monotonic()
    for _ in range(3):                             # 指示なしの発話（受け皿ができる）
        agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})
    agent._turns[0]["first"] = t0 - 3.0            # 前の発話: 3秒前に始まり
    for _ in range(int(_live._SPEECH_END_GAP_SEC * 10)):
        agent._handle({"type": "session.output_audio.delta", "delta": _silent()})
    agent._turns[0]["last"] = t0 - 0.5             # 0.5秒前に声が終わった
    agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})   # 次の発話が始まる
    assert len(agent._turns) == 2
    # いま届いた文字は「遅れ 1 秒」を引くと前の発話の区間（-3.0〜-0.5+0.3）に入る
    agent._handle({"type": "session.output_transcript.delta", "delta": "前の末尾"})
    assert agent._turns[0]["text"] == "前の末尾" and agent._turns[1]["text"] == ""
    # 1.5 秒後に届く文字は次の発話のもの
    agent._turns[1]["first"] = t0 + 0.2
    assert agent._turn_for_text(t0 + 1.6) is agent._turns[1]


def test_resume_shortly_after_a_requested_turn_is_a_continuation(agent):
    """指示で始まった発話が 1.2 秒の間で割れても、続きは指示なし発話に数えない."""
    agent.feed("A", "x")
    agent.trigger(invite_target="B")
    agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})
    agent._handle({"type": "session.output_transcript.delta", "delta": "前半"})
    for _ in range(int(_live._SPEECH_END_GAP_SEC * 10)):
        agent._handle({"type": "session.output_audio.delta", "delta": _silent()})
    _settle(agent)
    assert agent.said == ["前半"]
    agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})   # すぐ再開
    assert agent.unrequested_turns == 0 and agent._turn_requested is True
    agent._finish_turn()
    agent._last_turn_end_at -= 10.0              # 十分たってからの発話は別扱い
    agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})
    assert agent.unrequested_turns == 1
    assert agent.last_unrequested_speech_at > 0


def test_context_sent_with_a_directive_is_capped(agent):
    for i in range(60):
        agent.feed("A", f"発話{i}")
    assert agent.pending_count <= _live._PENDING_KEEP
    agent.trigger(invite_target="B")
    thinking = "".join(m["content"] for m in agent.ws.sent
                       if m["type"] == "session.thinking.append")
    assert "発話59" in thinking and "発話30" not in thinking
    assert agent.pending_count == 0


def test_silence_directive_tells_the_model_what_to_do(agent):
    agent.feed("A", "x")
    agent.trigger(silence_sec=18.4, topics=[{"topic": "議題X", "speaker": "議題"}])
    instr = next(m for m in agent.ws.sent if m["type"] == "session.instructions.append")
    assert "[沈黙]" in instr["content"] and "18秒" in instr["content"]
    assert "一つだけ提案" in instr["content"]


def test_feed_audio_drops_old_backlog(agent):
    for _ in range(_live._INPUT_BACKLOG_MAX + 10):
        agent.feed_audio(b"\x00\x00" * 1600)
    assert agent._in_q.qsize() == _live._INPUT_BACKLOG_MAX


def test_handle_exception_does_not_kill_the_recv_loop(agent):
    class WS:
        def __init__(self):
            self.n = 0

        def recv(self):
            self.n += 1
            if self.n == 1:
                return '{"type": "session.output_audio.delta", "delta": "not-base64!!"}'
            raise RuntimeError("closed")
    agent._recv_loop(WS(), agent._session_gen)
    assert agent._connected is False             # 例外で死なず、切断まで回って正しく落とす


def test_giveup_cancels_the_pending_directive(agent):
    agent.feed("A", "x")
    agent.trigger(invite_target="B")
    agent._speak_trigger_at = time.monotonic() - (_live._NO_SPEECH_GIVEUP_SEC + 1)
    agent._watchdog_tick()
    assert agent._responding is False
    assert agent.ws.sent[-1]["type"] == "session.instructions.append"
    assert "取り消し" in agent.ws.sent[-1]["content"]
    assert agent._discarding is True             # 「了解」のような返事は捨てる


# --- 再生のジッタ吸収（2026-09-14: 発話がぶつ切りに聞こえた） ------------------


def test_queued_audio_is_tracked_for_buffering(agent):
    agent._responding = True
    for _ in range(3):
        agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})
    assert agent._queued_audio_ms() == 300
    agent.stop_playback()
    assert agent._queued_audio_ms() == 0


def test_turn_stats_include_stream_seconds(agent):
    agent._responding = True
    for _ in range(3):
        agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})
    for _ in range(int(_live._SPEECH_END_GAP_SEC * 10)):
        agent._handle({"type": "session.output_audio.delta", "delta": _silent()})
    _settle(agent)
    st = agent.last_turn_stats
    assert st["voiced_sec"] == 0.3 and st["stream_sec"] == 1.5   # 無音 1.2 秒を含む


# --- 入力時計（2026-09-14: 無音の埋め方が壁時計に追いついていなかった） ------------


def test_clock_fills_silence_to_keep_input_in_step_with_wall_time(agent):
    agent._input_base_at = time.monotonic() - 1.0    # 接続から 1 秒たったのに入力なし
    agent._input_sent_ms = 0
    sent = agent._fill_clock()
    assert sent == 1000                              # 1 秒ぶんの無音で追いつく
    assert agent._input_sent_ms == 1000
    assert agent._fill_clock() == 0                  # 追いついた直後は送らない
    assert all(m["type"] == "session.input_audio.append" for m in agent.ws.sent)


def test_clock_counts_real_audio_and_only_fills_the_gap(agent):
    agent._input_base_at = time.monotonic() - 0.5
    agent.feed_audio(b"\x00\x00" * 1600)             # 100ms の実音声
    agent._flush_input()
    assert agent._input_sent_ms == 100
    assert agent._fill_clock() == 400                 # 残り 400ms を無音で


def test_clock_rebases_after_a_long_stall(agent):
    agent._input_base_at = time.monotonic() - 30.0   # スリープ明けなど
    assert agent._fill_clock() == 0
    assert agent._input_sent_ms == 0 and time.monotonic() - agent._input_base_at < 0.1


def test_unrequested_turn_triggers_a_quiet_reminder_at_most_once_a_minute(agent):
    for _ in range(3):
        agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})
    for _ in range(int(_live._SPEECH_END_GAP_SEC * 10)):
        agent._handle({"type": "session.output_audio.delta", "delta": _silent()})
    reminders = [m for m in agent.ws.sent if m["type"] == "session.instructions.append"]
    assert len(reminders) == 1 and "求められていません" in reminders[0]["content"]
    for _ in range(3):                                # すぐにもう一度、指示なしで話す
        agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})
    for _ in range(int(_live._SPEECH_END_GAP_SEC * 10)):
        agent._handle({"type": "session.output_audio.delta", "delta": _silent()})
    reminders = [m for m in agent.ws.sent if m["type"] == "session.instructions.append"]
    assert len(reminders) == 1                        # 60 秒以内は送らない


def test_requested_turn_does_not_trigger_a_reminder(agent):
    agent.feed("A", "x")
    agent.trigger(invite_target="B")
    n = len(agent.ws.sent)
    for _ in range(3):
        agent._handle({"type": "session.output_audio.delta", "delta": _voiced()})
    for _ in range(int(_live._SPEECH_END_GAP_SEC * 10)):
        agent._handle({"type": "session.output_audio.delta", "delta": _silent()})
    assert len(agent.ws.sent) == n

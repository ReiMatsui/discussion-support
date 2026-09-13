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


@pytest.fixture
def agent(monkeypatch):
    # スレッド（再生・時計・監視）は起動しない。受信は _handle を直接呼ぶ
    monkeypatch.setattr(LiveAgent, "_start_playback_thread", lambda self: None)
    monkeypatch.setattr(LiveAgent, "_start_clock", lambda self: None)
    monkeypatch.setattr(LiveAgent, "_start_watchdog", lambda self: None)
    a = LiveAgent(api_key="k")
    a.ws = FakeWS()
    a._connected = True
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
    assert agent.pending_count() == 0          # 送った分は消費される


def test_trigger_is_skipped_while_speaking_or_without_intent(agent):
    agent.trigger()                              # 発話も意図もない → 何も送らない
    assert agent.ws.sent == []
    agent.feed("A", "x")
    agent.ai_speaking = True
    agent.trigger(invite_target="B")
    assert agent.ws.sent == []                   # 話している間は送らない
    assert agent.pending_count() == 1            # 発話は保持される


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
    assert agent.said == ["テスト段階での"]     # 1発話にまとまる
    assert agent._responding is False
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
    monkeypatch.setattr(LiveAgent, "connect", lambda self: calls.append("connect"))
    old_ws = agent.ws
    agent.apply_config(trigger_n=5)
    assert calls == [] and agent.ws is old_ws    # trigger_n だけなら開き直さない
    agent.apply_config(mode="conversation")
    assert calls == ["connect"]
    assert old_ws.closed and old_ws.sent[-1]["type"] == "session.close"
    agent.apply_config(mode="off")
    assert calls == ["connect"]                  # off は閉じるだけ


def test_conversation_mode_does_not_send_directives(agent):
    agent.mode = "conversation"
    agent.feed("A", "どう思う？")
    agent.trigger(invite_target="B")
    assert agent.ws.sent == []


def test_feed_audio_resamples_to_24k(agent):
    agent.feed_audio(b"\x00\x00" * 1600)         # 100ms @16k
    msg = agent.ws.sent[-1]
    assert msg["type"] == "session.input_audio.append"
    assert len(base64.b64decode(msg["audio"])) == 2400 * 2

"""RealtimeAgent（ファシリテーター）の介入・割り込みロジックのユニットテスト."""
from __future__ import annotations

import pytest

from .conftest import make_chunk


# ---------------------------------------------------------------------------
# trigger()
# ---------------------------------------------------------------------------

def test_trigger_sends_create_and_response(agent):
    agent.feed("人間", "これはテスト発言です")
    agent.trigger()
    assert agent.ws.types() == ["conversation.item.create", "response.create"]
    assert agent._responding is True
    assert "これはテスト発言です" in agent.ws.last_create_text()
    # _pending は送信後にクリアされている
    assert agent.pending_count == 0


def test_trigger_noop_when_nothing_pending(agent):
    agent.trigger()
    assert agent.ws.sent == []
    assert agent._responding is False


def test_trigger_skipped_while_responding(agent):
    agent.feed("人間", "発言")
    agent._responding = True
    agent.trigger()
    assert agent.ws.sent == []  # 応答生成中は新規送信しない


def test_trigger_with_drift_reason_injects_context(agent):
    agent.trigger(drift_reason="ラーメンの雑談に逸脱")
    text = agent.ws.last_create_text()
    assert "[脱線検出]" in text
    assert "ラーメンの雑談に逸脱" in text
    assert agent._responding is True


def test_trigger_with_topics_includes_topic_note(agent):
    agent.feed("人間", "本題の発言")
    agent.trigger(topics=[{"topic": "AI導入の是非", "speaker": "松井"}])
    text = agent.ws.last_create_text()
    assert "現在の論点" in text
    assert "AI導入の是非" in text


# ---------------------------------------------------------------------------
# interrupt() と介入内容の保存
# ---------------------------------------------------------------------------

def test_interrupt_saves_pending_intervention(agent):
    agent.ai_speaking = True
    agent._ai_text_buf = "重要な指摘です"
    agent._current_item_id = "item-1"
    agent._played_bytes = 4800
    agent.interrupt()

    pi = agent._pending_intervention
    assert pi is not None
    assert pi["delivered"] == "重要な指摘です"
    assert pi["attempts"] == 1
    assert "response.cancel" in agent.ws.types()
    assert "conversation.item.truncate" in agent.ws.types()


def test_interrupt_increments_attempts(agent):
    agent._pending_intervention = {"delivered": "前回", "created_at": 0.0, "attempts": 1}
    agent.ai_speaking = True
    agent._ai_text_buf = "再度の指摘"
    agent.interrupt()
    assert agent._pending_intervention["attempts"] == 2


def test_interrupt_discards_after_max_retries(agent):
    # MAX_RETRIES=2。既に2回保存済みの状態でさらに割り込まれたら破棄
    agent._pending_intervention = {"delivered": "x", "created_at": 0.0, "attempts": 2}
    agent.ai_speaking = True
    agent._ai_text_buf = "3回目"
    agent.interrupt()
    assert agent._pending_intervention is None


def test_interrupt_noop_when_idle(agent):
    agent.interrupt()
    assert agent.ws.sent == []
    assert agent._pending_intervention is None


# ---------------------------------------------------------------------------
# _cancel_response（「介入不要」）
# ---------------------------------------------------------------------------

def test_cancel_response_clears_intervention_and_deletes_item(agent):
    agent._pending_intervention = {"delivered": "x", "created_at": 0.0, "attempts": 1}
    agent._current_item_id = "item-1"
    agent._cancel_response()
    assert agent._pending_intervention is None
    assert agent.ai_speaking is False
    assert "response.cancel" in agent.ws.types()
    assert "conversation.item.delete" in agent.ws.types()


# ---------------------------------------------------------------------------
# プリフライト: 「介入不要」応答で音声を漏らさない（Bug 1 の回帰テスト）
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="Bug 1 未修正: _preflight_chars(3) < マーカー長のため判定前にflushしてしまう",
    strict=True,
)
def test_preflight_no_leak_on_cancel_marker(agent):
    """「（介入不要）」と判定される応答では、音声再生も on_speech_start も起きない。

    プリフライトの本来の目的（介入不要の音声漏れ防止）を保証する。
    """
    started: list[bool] = []
    agent.on_speech_start = lambda: started.append(True)

    agent._handle({"type": "response.output_item.added", "item": {"id": "it1"}})
    # 音声が先着（preflightバッファに溜まる）
    agent._handle({"type": "response.output_audio.delta", "delta": make_chunk()})
    # 「（介入不要）」が1文字ずつ届く
    for ch in "（介入不要）":
        agent._handle({"type": "response.output_audio_transcript.delta", "delta": ch})

    assert started == [], "介入不要の応答で on_speech_start が呼ばれてはならない"
    assert agent.ai_speaking is False


def test_preflight_flushes_for_real_intervention(agent):
    """通常の介入では、テキスト確認後に音声がフラッシュされ on_speech_start が発火する."""
    started: list[bool] = []
    agent.on_speech_start = lambda: started.append(True)

    agent._handle({"type": "response.output_item.added", "item": {"id": "it1"}})
    agent._handle({"type": "response.output_audio.delta", "delta": make_chunk()})
    for ch in "それは論点がずれています":
        agent._handle({"type": "response.output_audio_transcript.delta", "delta": ch})

    assert started == [True]
    assert agent._preflight_cleared is True


# ---------------------------------------------------------------------------
# in_echo_window
# ---------------------------------------------------------------------------

def test_echo_window_states(agent):
    import time
    assert agent.in_echo_window is False           # idle
    agent.ai_speaking = True
    assert agent.in_echo_window is True             # speaking
    agent.ai_speaking = False
    agent._last_speech_end = time.monotonic()
    assert agent.in_echo_window is True             # cooldown内
    agent._last_speech_end = time.monotonic() - 10
    assert agent.in_echo_window is False            # cooldown経過

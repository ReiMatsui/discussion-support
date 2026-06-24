"""ConversationPartner（対話相手）の割り込み・コンテキスト注入のユニットテスト."""
from __future__ import annotations


def test_inject_context_without_response(partner):
    partner.inject_context("ファシリテーター", "論点に戻しましょう")
    assert partner.ws.types() == ["conversation.item.create"]
    text = partner.ws.last_create_text()
    assert "ファシリテーター" in text
    assert "論点に戻しましょう" in text


def test_inject_context_request_response_resets_interrupted(partner):
    partner._interrupted = True
    partner.inject_context("人間", "それは違うと思う", request_response=True)
    assert partner.ws.types() == ["conversation.item.create", "response.create"]
    # request_response時は残留破棄フラグを解除しないと新応答の音声まで捨てられる
    assert partner._interrupted is False


def test_interrupt_sends_cancel_and_truncate(partner):
    partner.ai_speaking = True
    partner._responding = True
    partner._current_item_id = "p-1"
    partner._played_bytes = 4800
    partner.interrupt()
    assert "response.cancel" in partner.ws.types()
    assert "conversation.item.truncate" in partner.ws.types()
    assert partner.ai_speaking is False
    assert partner._responding is False
    assert partner._interrupted is True


def test_interrupt_noop_when_idle(partner):
    partner.interrupt()
    assert partner.ws.sent == []


def test_cancelled_response_keeps_responding(partner):
    """キャンセル応答の response.done では _responding を維持する（直後の再応答に備える）."""
    partner._responding = True
    partner._ai_text_buf = "途中まで喋った"
    partner._handle({"type": "response.done", "response": {"status": "cancelled"}})
    assert partner._responding is True          # 維持される
    assert partner._ai_text_buf == ""           # バッファはクリア
    assert "途中まで喋った" in list(partner._recent_ai_texts)


def test_completed_response_resets_responding(partner):
    partner._responding = True
    partner._ai_text_buf = "完結した発言"
    partner._handle({"type": "response.done", "response": {"status": "completed"}})
    assert partner._responding is False


def test_echo_window_includes_responding(partner):
    assert partner.in_echo_window is False
    partner._responding = True
    assert partner.in_echo_window is True

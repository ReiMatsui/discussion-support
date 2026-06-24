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


# ---------------------------------------------------------------------------
# 応答世代(epoch)による ai_speaking 管理（Bug 6）
# ---------------------------------------------------------------------------

def test_interrupted_self_heals_on_new_response(partner):
    """partnerでも、新応答開始で_interruptedが解除される（response.done非依存の堅牢化）."""
    partner._interrupted = True
    partner._handle({"type": "response.output_item.added", "item": {"id": "p-new"}})
    assert partner._interrupted is False

    partner._handle({"type": "response.output_audio.delta",
                     "delta": __import__("base64").b64encode(b"\x01\x02" * 10).decode()})
    assert partner.ai_speaking is True
    payloads = [p for (_e, p) in list(partner._audio_q.queue) if p is not None]
    assert len(payloads) == 1


def test_stale_terminator_does_not_clear_ai_speaking(partner):
    partner._play_epoch = 2
    partner.ai_speaking = True
    partner._on_playback_terminator(epoch=1)
    assert partner.ai_speaking is True


def test_latest_terminator_clears_ai_speaking(partner):
    partner._play_epoch = 2
    partner.ai_speaking = True
    partner._on_playback_terminator(epoch=2)
    assert partner.ai_speaking is False


def test_output_item_added_bumps_epoch(partner):
    partner._handle({"type": "response.output_item.added", "item": {"id": "p1"}})
    assert partner._play_epoch == 1
    partner._handle({"type": "response.output_audio.delta",
                     "delta": __import__("base64").b64encode(b"\x01\x02" * 10).decode()})
    epochs = [e for (e, _payload) in list(partner._audio_q.queue)]
    assert epochs == [1]


# ---------------------------------------------------------------------------
# error イベントの扱い（Bug 5: cancel→create と VAD自動応答の競合）
# ---------------------------------------------------------------------------

def test_benign_error_already_active_response_is_ignored(partner):
    """「already has an active response」は良性として無視し、状態を変えない."""
    partner._responding = True
    partner._handle({
        "type": "error",
        "error": {"message": "Conversation already has an active response"},
    })
    # 良性なので _responding をリセットしない（VAD応答が進行中）
    assert partner._responding is True


def test_benign_error_no_active_response_is_ignored(partner):
    partner._responding = True
    partner._handle({
        "type": "error",
        "error": {"message": "Cancellation failed: no active response found"},
    })
    assert partner._responding is True


def test_unexpected_error_resets_responding(partner):
    """想定外エラーでは _responding を解放して固着を防ぐ."""
    partner._responding = True
    partner._interrupted = True
    partner._handle({
        "type": "error",
        "error": {"message": "internal_server_error: something broke"},
    })
    assert partner._responding is False
    assert partner._interrupted is False

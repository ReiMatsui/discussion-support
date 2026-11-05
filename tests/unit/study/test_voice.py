from __future__ import annotations

import json


def test_unrequested_voice_is_blocked():
    from das.asr.live.agents._live import LiveAgent

    a = LiveAgent(api_key="fake", study_mode=True)
    assert not a.trigger(manual_request="自由に答えて")
    a._on_audio({"delta": "invalid base64 but must not reach decoder"})
    assert not a.ai_speaking


def test_prepared_voice_uses_exact_text_instruction():
    from das.asr.live.agents._live import LiveAgent

    class FakeWS:
        def __init__(self):
            self.sent = []

        def send(self, value):
            self.sent.append(json.loads(value))

    a = LiveAgent(api_key="fake", study_mode=True)
    a.ws = FakeWS()
    a._connected = True
    a._started.set()
    text = "D案を除外する提案ですね。理由の補足や訂正、異論はありますか。"
    assert a.trigger(prepared_text=text, prepared_deadline=12345)
    assert a._study_audio_deadline == 12345
    assert text in a.ws.sent[-1]["content"]
    assert "追加や回答はしない" in a.ws.sent[-1]["content"]
    assert not any(x["type"] == "session.thinking.append" for x in a.ws.sent)


def test_late_first_audio_is_dropped(monkeypatch):
    from das.asr.live.agents._live import LiveAgent

    a = LiveAgent(api_key="fake", study_mode=True)
    a._responding = True
    a._study_audio_deadline = 0
    stopped = []
    monkeypatch.setattr(a, "stop_playback", lambda: stopped.append(True))
    a._on_audio({"delta": "invalid base64 should not reach decoder"})
    assert stopped == [True]


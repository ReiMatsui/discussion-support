"""実行中モード切替（F3）のテスト: set_session_mode / /api/mode."""
from __future__ import annotations

import datetime
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from das.asr.live._session_state import SessionState
from das.asr.live._ui import _UIHandler
from das.asr.live._workers import set_session_mode


def _make_state():
    s = SessionState(
        args=object(), started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md", html_path="/tmp/o.html", diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns", wav_path="/tmp/o.wav",
    )
    s.save = lambda *a, **k: None  # type: ignore[assignment]  # ファイルI/Oを避ける
    return s


class FakeAgent:
    def __init__(self, mode="facilitator"):
        self.mode = mode
        self.voice = "shimmer"
        self.applied: list = []

    def apply_config(self, *, mode=None, voice=None, trigger_n=None):
        self.applied.append(mode)
        if mode:
            self.mode = mode

    @property
    def enabled(self):
        return self.mode != "off"


class FakePartner:
    def __init__(self, **kw):
        self.kw = kw
        self._connected = False
        self.closed = False
        self.on_ai_utterance = None

    def set_tracker(self, t):
        pass

    def connect(self):
        self._connected = True

    def close(self):
        self.closed = True


# --- set_session_mode -------------------------------------------------------

def test_mode_error_when_no_agent():
    r = set_session_mode(_make_state(), "facilitate")
    assert r["ok"] is False


def test_mode_error_unknown():
    s = _make_state()
    s.agent = FakeAgent()
    assert set_session_mode(s, "bogus")["ok"] is False


def test_mode_transcribe_turns_agent_off_and_detaches():
    s = _make_state()
    s.agent = FakeAgent(mode="facilitator")
    p = FakePartner()
    s.partner = p
    r = set_session_mode(s, "transcribe")
    assert r == {"ok": True, "mode": "transcribe"}
    assert s.agent.mode == "off"
    assert s.partner is None
    assert p.closed is True


def test_mode_facilitate_detaches_partner():
    s = _make_state()
    s.agent = FakeAgent(mode="off")
    p = FakePartner()
    s.partner = p
    r = set_session_mode(s, "facilitate")
    assert r == {"ok": True, "mode": "facilitate"}
    assert s.agent.mode == "facilitator"
    assert s.partner is None
    assert p.closed is True


def test_mode_converse_attaches_partner(monkeypatch):
    import das.asr.live.agents._partner as partner_mod
    monkeypatch.setattr(partner_mod, "ConversationPartner", FakePartner)
    s = _make_state()
    s.agent = FakeAgent(mode="facilitator")
    s._partner_cfg = {"api_key": "k", "voice": "echo", "topic": "テーマ"}
    r = set_session_mode(s, "converse")
    assert r == {"ok": True, "mode": "converse"}
    assert isinstance(s.partner, FakePartner)
    assert s.partner._connected is True
    assert s.partner.on_ai_utterance is not None


def test_mode_converse_without_key_stays_facilitate(monkeypatch):
    """api_key無しではパートナーを作れず facilitate のままになる."""
    s = _make_state()
    s.agent = FakeAgent(mode="facilitator")
    s._partner_cfg = {"api_key": "", "voice": "echo", "topic": ""}
    r = set_session_mode(s, "converse")
    assert r == {"ok": True, "mode": "facilitate"}
    assert s.partner is None


# --- HTTP /api/mode ---------------------------------------------------------

def test_http_mode_switch():
    s = _make_state()
    s.agent = FakeAgent(mode="off")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _UIHandler.create(s))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/mode",
            data=json.dumps({"mode": "facilitate"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read())
        assert out == {"ok": True, "mode": "facilitate"}
        assert s.agent.mode == "facilitator"
    finally:
        httpd.shutdown()

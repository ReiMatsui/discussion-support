"""UI バックエンドAPI（F1）のテスト: session_mode / api_snapshot / HTTP."""
from __future__ import annotations

import datetime
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from das.asr.live._session_state import SessionState
from das.asr.live._ui import _UIHandler


def _make_state():
    return SessionState(
        args=object(), started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md", html_path="/tmp/o.html", diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns", wav_path="/tmp/o.wav",
    )


class _FakeAgent:
    def __init__(self, mode="facilitator"):
        self.mode = mode
        self.voice = "shimmer"

    @property
    def enabled(self):
        return self.mode != "off"


# --- session_mode -----------------------------------------------------------

def test_session_mode_transcribe_when_no_agent():
    assert _make_state().session_mode() == "transcribe"


def test_session_mode_transcribe_when_agent_off():
    s = _make_state()
    s.agent = _FakeAgent(mode="off")
    assert s.session_mode() == "transcribe"


def test_session_mode_facilitate_without_partner():
    s = _make_state()
    s.agent = _FakeAgent(mode="facilitator")
    assert s.session_mode() == "facilitate"


def test_session_mode_converse_with_partner():
    s = _make_state()
    s.agent = _FakeAgent(mode="facilitator")
    s.partner = object()  # type: ignore[assignment]
    assert s.session_mode() == "converse"


# --- api_snapshot -----------------------------------------------------------

def test_api_snapshot_speakers_have_rename_labels():
    """話者にリネーム用ラベルが付き、AI話者は除外される（F5）."""
    s = _make_state()
    s.records = [
        {"speaker": "#1", "text": "a", "ms": 0, "end_ms": 500},
        {"speaker": "ファシリテーター", "text": "x", "ms": None, "end_ms": None},
    ]
    by_name = {sp["name"]: sp for sp in s.api_snapshot()["speakers"]}
    assert "話者1" in by_name
    assert by_name["話者1"]["label"] == "1"
    assert by_name["話者1"]["renameable"] is True
    assert "ファシリテーター" not in by_name  # AI話者はリネーム対象外


def test_http_rename_without_tracker():
    """/rename（声紋なし）で表示名が更新される（SPAのリネームフロー）."""
    s = _make_state()
    s.records = [{"speaker": "#2", "text": "hi", "ms": 0, "end_ms": 500}]
    httpd, port = _serve(s)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/rename",
            data=json.dumps({"label": "2", "name": "田中"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read())
        assert out.get("ok") is True
        assert s.names["#2"] == "田中"
    finally:
        httpd.shutdown()


def test_api_snapshot_structure():
    s = _make_state()
    s.records = [
        {"speaker": "話者1", "text": "こんにちは", "ms": 0, "end_ms": 1000},
        {"ms": 1000, "sys": "声を登録しました"},
    ]
    s.topics = [{"topic": "AI導入", "speaker": "話者1"}]
    snap = s.api_snapshot()
    assert snap["mode"] == "transcribe"
    assert snap["running"] is True
    assert {r["type"] for r in snap["records"]} == {"utt", "sys"}
    assert snap["records"][0]["speaker"] == "話者1"
    assert snap["topics"][0]["topic"] == "AI導入"
    assert snap["participation"][0]["speaker"] == "話者1"
    # JSON化できること
    json.dumps(snap, ensure_ascii=False)


# --- HTTP API ---------------------------------------------------------------

def _serve(state):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _UIHandler.create(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def test_reset_for_new_meeting():
    """リセットで議事録・論点・カーソルはクリア、話者名は引き継ぐ（F6）."""
    s = _make_state()
    s.records = [{"speaker": "#1", "text": "前の会議", "ms": 0, "end_ms": 500}]
    s.topics = [{"topic": "古い論点", "speaker": "#1"}]
    s.names["#1"] = "松井"
    s.agent_cursor = 1
    s.drift_cursor = 1
    s.drift_requests.put("脱線")
    old_out, old_started = s.out_path, s.started

    result = s.reset_for_new_meeting()

    assert result["ok"] is True
    assert s.records == []                 # 議事録クリア
    assert s.topics == []                  # 論点クリア
    assert s.agent_cursor == 0 and s.drift_cursor == 0
    assert s.drift_requests.empty()        # キューもクリア
    assert s.names["#1"] == "松井"          # 話者名は引き継ぐ
    assert s.out_path != old_out           # 新しい出力先
    assert s.started >= old_started        # 新しい開始時刻


def test_http_reset_clears_records():
    s = _make_state()
    s.records = [{"speaker": "#1", "text": "x", "ms": 0, "end_ms": 100}]
    httpd, port = _serve(s)
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/reset", method="POST")
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read())
        assert out["ok"] is True
        assert s.records == []
    finally:
        httpd.shutdown()


def test_http_get_root_serves_spa():
    """GET / が新SPA(HTML)を配信する."""
    state = _make_state()
    httpd, port = _serve(state)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as r:
            html = r.read().decode("utf-8")
            ctype = r.headers.get("Content-Type", "")
        assert "text/html" in ctype
        assert "<title>議論支援</title>" in html
        assert "/api/stream" in html   # SSEを使うSPA
        assert "/api/mode" in html     # モード切替
    finally:
        httpd.shutdown()


def test_http_get_state_and_post_stop():
    state = _make_state()
    state.records = [{"speaker": "話者1", "text": "やあ", "ms": 0, "end_ms": 500}]
    stop_calls = []
    state.request_stop = lambda: stop_calls.append(True)

    httpd, port = _serve(state)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state") as r:
            data = json.loads(r.read())
        assert data["mode"] == "transcribe"
        assert data["records"][0]["text"] == "やあ"

        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/stop", method="POST")
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read())
        assert out["ok"] is True
        assert stop_calls == [True]  # request_stop が呼ばれた
    finally:
        httpd.shutdown()


def test_http_stop_fallback_sets_event():
    """request_stop未設定でも /api/stop で stop イベントが立つ."""
    state = _make_state()  # request_stop は None
    httpd, port = _serve(state)
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/stop", method="POST")
        with urllib.request.urlopen(req) as r:
            json.loads(r.read())
        assert state.stop.is_set()
    finally:
        httpd.shutdown()


def test_sse_stream_sends_snapshot():
    """/api/stream が SSE で最新スナップショットを配信する（F2）."""
    state = _make_state()
    state.records = [{"speaker": "話者1", "text": "やあ", "ms": 0, "end_ms": 500}]
    httpd, port = _serve(state)
    resp = None
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/stream", timeout=5)
        data_line = None
        for _ in range(20):
            line = resp.readline().decode("utf-8")
            if line.startswith("data: "):
                data_line = line[len("data: "):].strip()
                break
        assert data_line is not None
        snap = json.loads(data_line)
        assert snap["records"][0]["text"] == "やあ"
        assert "rev" in snap
    finally:
        if resp is not None:
            resp.close()
        state.stop.set()
        httpd.shutdown()

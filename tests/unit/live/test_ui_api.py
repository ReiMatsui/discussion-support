"""UI バックエンドAPI（F1）のテスト: session_mode / api_snapshot / HTTP."""
from __future__ import annotations

import datetime
import json
import threading
import urllib.error
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
        self.trigger_n = 10
        self._connected = False
        self._conn_error = None

    @property
    def enabled(self):
        return self.mode != "off"

    def apply_config(self, *, mode=None, voice=None, trigger_n=None):
        if mode is not None:
            self.mode = mode
        if voice is not None:
            self.voice = voice
        if trigger_n is not None:
            self.trigger_n = trigger_n


class _FakeTracker:
    """声紋トラッカーの最小スタブ（事前登録・名簿テスト用）."""
    def __init__(self, auto=True):
        self.auto = auto
        self.model = "redimnet"
        self.profiles: dict = {}
        self.enrolled: list = []

    def enroll_from_audio(self, name, wav):
        self.enrolled.append((name, len(wav)))
        self.profiles[name] = 1
        return True

    def all_profile_names(self):
        return list(self.profiles)

    def active_profile_names(self):
        return list(self.profiles)


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
    """声紋確定済みで名前未登録の参加者だけがリネーム可能、暫定ラベルとAIは対象外（F5）."""
    s = _make_state()
    s.records = [
        {"speaker": "人物1", "text": "a", "ms": 0, "end_ms": 500},
        {"speaker": "#1", "text": "b", "ms": 500, "end_ms": 1000},
        {"speaker": "ファシリテーター", "text": "x", "ms": None, "end_ms": None},
    ]
    by_name = {sp["name"]: sp for sp in s.api_snapshot()["speakers"]}
    assert by_name["参加者A"]["label"] == "人物1"
    assert by_name["参加者A"]["renameable"] is True
    assert by_name["参加者B"]["renameable"] is False  # 暫定#Nは登録対象外
    assert "ファシリテーター" not in by_name        # AI話者はリネーム対象外


def test_api_snapshot_exposes_intervention_settings():
    s = _make_state()
    s.agent = _FakeAgent()
    s.set_proactivity("controlled")
    s.add_intervention_event("count", "10>=10発話")

    snap = s.api_snapshot()

    assert snap["intervention"] == {
        "enabled": True,
        "proactivity": "controlled",
        "trigger_n": 10,
    }
    assert snap["intervention_events"] == [{
        "time": snap["intervention_events"][0]["time"],
        "reason": "count",
        "detail": "10>=10発話",
    }]


def test_api_snapshot_exposes_startup_setup_state():
    s = _make_state()
    s.waiting_to_start = True

    assert s.api_snapshot()["setup"] == {"waiting": True}


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
    assert snap["records"][0]["speaker"] == "参加者A"
    assert snap["topics"][0]["topic"] == "AI導入"
    assert snap["topics"][0]["speaker"] == "参加者A"
    assert snap["participation"][0]["speaker"] == "参加者A"
    # JSON化できること
    json.dumps(snap, ensure_ascii=False)


# --- HTTP API ---------------------------------------------------------------

def _serve(state):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _UIHandler.create(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def test_reset_for_new_meeting():
    """リセットで議事録・論点・カーソル・話者ラベリングをクリアする（課題③）."""
    s = _make_state()
    s.records = [{"speaker": "#1", "text": "前の会議", "ms": 0, "end_ms": 500}]
    s.topics = [{"topic": "古い論点", "speaker": "#1"}]
    s.names["#1"] = "松井"
    s.colors["#1"] = "\033[36m"
    s.partial_text = "認識中…"
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
    assert s.names == {}                   # 話者名もリセット（課題③）
    assert s.colors == {}                  # 色割り当てもリセット
    assert s.partial_text == ""            # 認識途中もクリア
    assert s.out_path != old_out           # 新しい出力先
    assert s.started >= old_started        # 新しい開始時刻


def test_reset_calls_tracker_reset_session():
    """リセット時に声紋トラッカーのセッション状態もリセットする（課題③）."""
    s = _make_state()
    calls = []
    s.tracker = type("T", (), {
        "reset_session": lambda self: calls.append(True),
        "all_profile_names": lambda self: [],
        "active_profile_names": lambda self: [],
        "profiles": {},
    })()
    s.reset_for_new_meeting()
    assert calls == [True]


# --- 課題①: 認識途中(partial)の配信 -----------------------------------------

def test_show_partial_updates_state_and_snapshot():
    """show_partial が partial を保持し、api_snapshot に載る（課題①）."""
    s = _make_state()
    s.records = [{"speaker": "#1", "text": "x", "ms": 0, "end_ms": 100}]
    s.show_partial("1", "いまここを認識中")
    assert s.partial_text == "いまここを認識中"
    snap = s.api_snapshot()
    assert snap["partial"]["text"] == "いまここを認識中"
    assert snap["partial"]["speaker"] == "参加者A"
    # 空文字でクリアされる
    s.show_partial("1", "")
    assert s.api_snapshot()["partial"] == {"speaker": "", "text": ""}


# --- 未確定話者・相槌のUI表示 ----------------------------------------------

def test_snapshot_unsure_and_backchannel_flags():
    s = _make_state()
    s.records = [
        {"speaker": "?", "text": "はい", "ms": 0, "end_ms": 300, "bc": True},   # 相槌=未確定
        {"speaker": "#1", "text": "本題です", "ms": 300, "end_ms": 600},
    ]
    snap = s.api_snapshot()
    recs = snap["records"]
    assert recs[0]["speaker"] == "未確定"
    assert recs[0]["unsure"] is True and recs[0]["bc"] is True
    assert recs[1]["bc"] is False
    # 未確定は参加度・リネーム候補から除外
    assert all(p["speaker"] != "未確定" for p in snap["participation"])
    assert all(sp["name"] != "未確定" for sp in snap["speakers"])


def test_speakers_registration_targets():
    """登録対象は声紋確定済みで名前未登録の参加者のみ。暫定/命名済み/未確定/AIは対象外."""
    s = _make_state()
    s.records = [
        {"speaker": "人物1", "text": "a", "ms": 0, "end_ms": 500},
        {"speaker": "松井", "text": "b", "ms": 500, "end_ms": 1000},      # 命名済み
        {"speaker": "#3", "text": "c", "ms": 1000, "end_ms": 1500},       # 暫定
        {"speaker": "?", "text": "うん", "ms": 1500, "end_ms": 1700},      # 未確定
        {"speaker": "ファシリテーター", "text": "x", "ms": None, "end_ms": None},
    ]
    by = {sp["name"]: sp for sp in s.api_snapshot()["speakers"]}
    assert by["参加者A"]["renameable"] is True and by["参加者A"]["label"] == "人物1"
    assert by["参加者B"]["renameable"] is False    # 暫定#Nは登録対象外
    assert by["松井"]["renameable"] is False      # 命名済みは登録対象外
    assert "未確定" not in by                      # 未確定は出さない
    assert "ファシリテーター" not in by             # AIは出さない


def test_anonymous_label_survives_voiceprint_rekey():
    """暫定ラベルが声紋で確定しても、画面上の参加者名は変えない。"""
    s = _make_state()
    s.records = [{"speaker": "#2", "text": "a", "ms": 0, "end_ms": 500}]
    assert s.api_snapshot()["records"][0]["speaker"] == "参加者A"

    s.rekey("#2", "人物1")

    snap = s.api_snapshot()
    assert snap["records"][0]["speaker"] == "参加者A"
    assert snap["speakers"][0]["name"] == "参加者A"
    assert snap["speakers"][0]["label"] == "人物1"


# --- 声紋ステータスの可視化 -------------------------------------------------

def test_snapshot_vp_disabled_when_no_tracker():
    s = _make_state()
    assert s.api_snapshot()["vp"] == {"enabled": False, "model": None,
                                      "locked": False, "roster": []}


def test_snapshot_vp_enabled_reports_model_and_roster():
    s = _make_state()
    s.tracker = _FakeTracker(auto=False)
    s.tracker.profiles = {"黒田": 1, "としや": 1}
    vp = s.api_snapshot()["vp"]
    assert vp["enabled"] is True and vp["model"] == "redimnet"
    assert vp["locked"] is True                  # auto=False → 名簿確定
    assert set(vp["roster"]) == {"黒田", "としや"}


# --- 事前登録・名簿（UI登録フロー） ----------------------------------------

def test_http_enroll_from_buffer():
    """/api/enroll が直近のPCMバッファからその人を声紋登録する."""
    s = _make_state()
    s.tracker = _FakeTracker()
    s.pcm_buf = bytearray(b"\x01\x00" * (16000 * 3))   # 3秒ぶん
    httpd, port = _serve(s)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/enroll",
            data=json.dumps({"name": "黒田", "seconds": 2}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read())
        assert out["ok"] is True and out["name"] == "黒田"
        assert s.tracker.enrolled and s.tracker.enrolled[0][0] == "黒田"
    finally:
        httpd.shutdown()


def test_http_enroll_rejects_too_short():
    """音声が2秒未満なら登録を拒否する."""
    s = _make_state()
    s.tracker = _FakeTracker()
    s.pcm_buf = bytearray(b"\x01\x00" * 8000)   # 0.5秒
    httpd, port = _serve(s)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/enroll",
            data=json.dumps({"name": "黒田", "seconds": 5}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req)
            raise AssertionError("拒否されるはず")
        except urllib.error.HTTPError as e:
            assert e.code == 400
        assert not s.tracker.enrolled
    finally:
        httpd.shutdown()


def test_http_roster_lock_toggles_auto():
    """/api/roster で名簿を確定すると自動登録がオフになる."""
    s = _make_state()
    s.tracker = _FakeTracker(auto=True)
    httpd, port = _serve(s)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/roster",
            data=json.dumps({"locked": True}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read())
        assert out == {"ok": True, "locked": True}
        assert s.tracker.auto is False
    finally:
        httpd.shutdown()


def test_http_diarization_updates_max_speakers_for_next_meeting():
    """/api/diarization で想定話者数を更新できる."""
    s = _make_state()
    httpd, port = _serve(s)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/diarization",
            data=json.dumps({"max_speakers": 3}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read())
        assert out == {"ok": True, "max_speakers": 3}
        assert s.args.diarization_max_speakers == 3
        assert s.api_snapshot()["diarization"]["max_speakers"] == 3
        assert s.records[-1]["sys"] == "想定話者数を更新: 3（新しい会議/再接続で確実に反映）"
    finally:
        httpd.shutdown()


def test_http_start_clears_startup_wait():
    s = _make_state()
    s.waiting_to_start = True
    httpd, port = _serve(s)
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/start", method="POST")
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read())
        assert out == {"ok": True, "waiting": False}
        assert s.waiting_to_start is False
        assert s.start_requested.is_set()
    finally:
        httpd.shutdown()


# --- 課題②: 安定した話者色 --------------------------------------------------

def test_snapshot_speaker_colors_are_stable():
    """同じ話者には常に同じ色が付き、records/participation で一致する（課題②）."""
    s = _make_state()
    s.records = [
        {"speaker": "#1", "text": "a", "ms": 0, "end_ms": 500},
        {"speaker": "#2", "text": "b", "ms": 500, "end_ms": 1000},
        {"speaker": "#1", "text": "c", "ms": 1000, "end_ms": 1500},
    ]
    snap1 = s.api_snapshot()
    snap2 = s.api_snapshot()
    c1 = [r["color"] for r in snap1["records"]]
    assert c1[0] == c1[2] and c1[0] != c1[1]      # 同一話者は同色・別話者は別色
    assert c1 == [r["color"] for r in snap2["records"]]  # 再取得でも不変
    part = {p["speaker"]: p["color"] for p in snap1["participation"]}
    assert part["参加者A"] == c1[0]                # participationの色もrecordsと一致


def test_http_reset_clears_records():
    s = _make_state()
    s.records = [{"speaker": "#1", "text": "x", "ms": 0, "end_ms": 100}]
    httpd, port = _serve(s)
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/reset", method="POST")
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read())
        assert out["ok"] is True
        assert s.records == []  # request_reset未設定 → 状態クリアにフォールバック
    finally:
        httpd.shutdown()


def test_http_reset_uses_request_reset_when_set():
    """STT接続ありの本番では request_reset（メインスレッドが作り直す）を呼ぶ."""
    s = _make_state()
    s.records = [{"speaker": "#1", "text": "x", "ms": 0, "end_ms": 100}]
    calls = []
    s.request_reset = lambda: calls.append(True)
    httpd, port = _serve(s)
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/reset", method="POST")
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read())
        assert out == {"ok": True, "resetting": True}
        assert calls == [True]
        assert s.records != []  # 直接はクリアしない（メインスレッドが行う）
    finally:
        httpd.shutdown()


def test_reset_resets_recording_and_snapshot_has_resetting():
    """リセットで録音wavとPCMバッファが作り直され、snapshotにresettingが入る."""
    s = _make_state()
    assert s.api_snapshot()["resetting"] is False
    s.pcm_total_bytes = 12345
    old_wav = s.wav_path
    s.reset_for_new_meeting()
    assert s.wav_path != old_wav        # 会議ごとに新しい録音
    assert s.pcm_total_bytes == 0       # PCMバッファ（STTのmsと整合）リセット


def test_set_agenda_replaces_baseline_keeps_points():
    """議題を設定すると既存の議題は差し替え、抽出論点は残る."""
    s = _make_state()
    s.topics = [
        {"topic": "古い議題", "speaker": "議題"},
        {"topic": "コストの話", "speaker": "話者1"},
    ]
    r = s.set_agenda("新しい議題")
    assert r == {"ok": True, "agenda": "新しい議題"}
    assert s._current_agenda() == "新しい議題"
    topics = [t["topic"] for t in s.topics]
    assert topics[0] == "新しい議題"          # 議題は先頭
    assert "コストの話" in topics            # 抽出論点は残る
    assert "古い議題" not in topics          # 旧議題は差し替え


def test_set_agenda_empty_clears():
    s = _make_state()
    s.topics = [{"topic": "議題X", "speaker": "議題"}]
    s.set_agenda("")
    assert s._current_agenda() == ""
    assert s.topics == []


def test_api_snapshot_has_agenda():
    s = _make_state()
    s.topics = [{"topic": "AI導入", "speaker": "議題(自動)"}]
    assert s.api_snapshot()["agenda"] == "AI導入"


def test_http_set_topic():
    s = _make_state()
    httpd, port = _serve(s)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/topic",
            data=json.dumps({"topic": "来期計画"}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read())
        assert out == {"ok": True, "agenda": "来期計画"}
        assert s._current_agenda() == "来期計画"
    finally:
        httpd.shutdown()


def test_http_set_intervention_settings():
    s = _make_state()
    s.agent = _FakeAgent()
    httpd, port = _serve(s)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/intervention",
            data=json.dumps({
                "enabled": False,
                "proactivity": "controlled",
                "trigger_n": 18,
            }).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as r:
            out = json.loads(r.read())
        assert out == {
            "ok": True,
            "enabled": False,
            "proactivity": "controlled",
            "trigger_n": 18,
        }
        assert s.intervention_enabled is False
        assert s.proactivity_name == "controlled"
        assert s.agent.trigger_n == 18
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
        assert "/api/start" in html
        assert 'id="setup-panel"' in html
        assert 'id="start-session"' in html
        assert "/api/intervention" in html
        assert 'id="speaker-count-status"' in html
        assert 'id="intervention-enabled"' in html
        assert 'id="proactivity"' in html
        assert 'id="event-panel"' in html
        assert "list.length < 1" in html  # 発言量は1人でも表示する
        assert ".bar-fill { display: block;" in html
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


def test_legacy_html_rename_targets_only_confirmed_unregistered_participants():
    s = _make_state()
    s.tracker = _FakeTracker()
    s.records = [
        {"speaker": "人物1", "text": "a", "ms": 0, "end_ms": 500},
        {"speaker": "#2", "text": "b", "ms": 500, "end_ms": 1000},
        {"speaker": "松井", "text": "c", "ms": 1000, "end_ms": 1500},
    ]

    s.write_html()

    with open(s.html_path, encoding="utf-8") as f:
        html = f.read()
    assert 'data-label="人物1"' in html
    assert 'data-label="#2"' not in html
    assert 'data-label="松井"' not in html
    assert "参加者A" in html


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

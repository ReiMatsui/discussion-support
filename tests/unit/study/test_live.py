from __future__ import annotations

import json
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from das.study.engine import StudyConfig
from das.study.live import StudyRuntime
from tests.unit.study.test_engine import review


class Immediate:
    calls = 0

    def submit(self, fn, *args):
        self.calls += 1
        f = Future()
        try:
            f.set_result(fn(*args))
        except Exception as exc:
            f.set_exception(exc)
        return f


class Agent:
    ready = True
    ai_speaking = False
    _responding = False
    in_echo_window = False

    def __init__(self):
        self.spoken = []
        self.stops = 0

    def trigger(self, **kw):
        self.spoken.append(kw)
        return True

    def stop_playback(self):
        self.stops += 1
        self.ai_speaking = False
        self._responding = False


def state():
    return SimpleNamespace(
        records=[
            {"speaker": "人物1", "text": "D案なら平日夜間に参加できます。", "end_ms": 1000},
            {"speaker": "人物2", "text": "D案は費用が重いので除外しよう。", "end_ms": 2000},
        ],
        state_lock=threading.RLock(),
        partial_text="",
        _last_partial_change=0,
        study_model="fake-no-api",
        study_api_key="not-a-real-key",
        agent=Agent(),
        intervention_enabled=True,
    )


def model(*_args):
    r = review()
    for x in [r.reasons[0].source, r.proposal.source, r.proposal.current_reason]:
        x.turn_id = {"t1": "turn-00001", "t2": "turn-00002"}[x.turn_id]
    return r


def runtime(tmp_path, condition="system"):
    return StudyRuntime(
        StudyConfig(session_id="synthetic", task_id="t", title="人工課題", options=["A案", "D案"],
                    pause_seconds=0),
        condition,
        str(tmp_path / "run.study.jsonl"),
    )


def test_mock_end_to_end_and_log(tmp_path):
    r = runtime(tmp_path)
    s = state()
    ex = Immediate()
    r.tick(s, ex, model, now=100)
    r.tick(s, ex, model, now=101)
    assert len(s.agent.spoken) == 1 and "平日夜間" in s.agent.spoken[0]["prepared_text"]
    assert len(r.view()["records"]) == 1
    assert r.engine.counts == {"D案": 1}
    events = [json.loads(x) for x in r.log_path.read_text().splitlines()]
    assert {
        "session_config",
        "transcript_snapshot",
        "review_result",
        "decision",
        "voice_requested",
    } <= {x["type"] for x in events}


def test_control_does_not_call_model_or_show_records(tmp_path):
    r = runtime(tmp_path, "none")
    s = state()
    ex = Immediate()
    r.tick(s, ex, lambda *_: pytest.fail("no model in control"), now=100)
    assert ex.calls == 0 and not s.agent.spoken and r.view()["records"] == []


def test_new_input_discards_delayed_model_result(tmp_path):
    class Delayed:
        def __init__(self):
            self.f = Future()

        def submit(self, *_):
            return self.f

    ex = Delayed()
    r = runtime(tmp_path)
    s = state()
    r.tick(s, ex, model, now=100)
    s.records.append(
        {"speaker": "人物3", "text": "D案の除外は取り下げてください。", "end_ms": 3000}
    )
    ex.f.set_result(model())
    r.tick(s, ex, model, now=101)
    assert not s.agent.spoken and not r.engine.counts
    assert "stale_llm_result" in r.log_path.read_text()


def test_partial_blocks_delivery(tmp_path):
    r = runtime(tmp_path)
    s = state()
    ex = Immediate()
    r.tick(s, ex, model, now=100)
    s.partial_text = "待って"
    s._last_partial_change = 100.5
    r.tick(s, ex, model, now=101)
    assert not s.agent.spoken


def test_confirmed_human_turn_stops_requested_audio(tmp_path):
    r = runtime(tmp_path)
    s = state()
    ex = Immediate()
    r.tick(s, ex, model, now=100)
    r.tick(s, ex, model, now=101)
    s.agent._responding = True
    s.records.append({"speaker": "人物3", "text": "訂正します。", "end_ms": 3000})
    r.tick(s, ex, model, now=102)
    assert s.agent.stops == 1


def test_llm_error_is_recorded_without_speaking(tmp_path):
    def error(*_):
        raise RuntimeError("fake failure")

    r = runtime(tmp_path)
    s = state()
    ex = Immediate()
    r.tick(s, ex, error, now=100)
    r.tick(s, ex, error, now=101)
    assert not s.agent.spoken and "review_error" in r.log_path.read_text()


def test_study_cli_has_explicit_conditions(tmp_path, monkeypatch):
    from das.asr.live import _bootstrap, main

    f = tmp_path / "cfg.json"
    f.write_text("{}")
    captured = []
    monkeypatch.setattr(_bootstrap, "run_session", lambda args: captured.append(args))
    result = CliRunner().invoke(main, ["--study-config", str(f), "--study-condition", "none"])
    assert result.exit_code == 0 and captured[0].study_condition == "none"


def test_condition_changing_ui_calls_rejected():
    from das.asr.live._ui import _UIRequestHandler

    h = object.__new__(_UIRequestHandler)
    h._state = SimpleNamespace(study=object())
    h.path = "/api/mode"
    out = []
    h._json = lambda code, data: out.append((code, data))
    h._dispatch_post()
    assert out[0][0] == 409


def test_legacy_workers_not_started_in_study_mode(monkeypatch):
    from das.asr.live import _bootstrap

    threads = []

    class FakeThread:
        def __init__(self, **kw):
            threads.append(kw)

        def start(self):
            pass

    monkeypatch.setattr(_bootstrap.threading, "Thread", FakeThread)
    _bootstrap._start_llm_workers(
        SimpleNamespace(study=SimpleNamespace(condition="none")),
        SimpleNamespace(no_llm=False),
        oai_key="fake",
        oai_model="fake",
        out_path="unused",
        explicit_agenda=False,
    )
    assert len(threads) == 1 and threads[0]["target"].__name__ == "run_study_worker"


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


def test_participant_http_routes_without_private_material(tmp_path):
    import urllib.request
    from http.server import ThreadingHTTPServer

    from das.asr.live._ui import _UIHandler

    runtime_obj = runtime(tmp_path, "none")
    fake = SimpleNamespace(study=runtime_obj)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _UIHandler.create(fake))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(base + "/study") as response:
            html = response.read().decode()
        with urllib.request.urlopen(base + "/api/study") as response:
            payload = json.loads(response.read())
        assert "textContent" in html and "innerHTML" not in html
        assert payload["records"] == [] and "scores" not in payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_auth_error_does_not_print_key_fragment(monkeypatch, capsys):
    import io
    import urllib.error
    import urllib.request

    from das.asr.live._bootstrap import _post_chat_json

    def fail(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "https://example.invalid",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"error":"sk-secret-fragment"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    assert _post_chat_json({"model": "fake"}, "fake-key", timeout=1, label="study-test") is None
    captured = capsys.readouterr().out
    assert "401" in captured and "sk-secret-fragment" not in captured


def test_auth_failure_stops_subsequent_model_requests(tmp_path):
    import urllib.error

    def unauthorized(*_):
        raise urllib.error.HTTPError("https://example.invalid", 401, "Unauthorized", {}, None)

    r = runtime(tmp_path)
    s = state()
    executor = Immediate()
    r.tick(s, executor, unauthorized, now=100)
    r.tick(s, executor, unauthorized, now=101)
    s.records.append({"speaker": "人物3", "text": "別の理由も考えましょう。", "end_ms": 3000})
    r.tick(s, executor, unauthorized, now=102)
    assert executor.calls == 1
    assert not s.agent.spoken
    assert "authentication_failed" in r.log_path.read_text()


def test_confirmed_input_without_partial_waits_before_review(tmp_path):
    r = runtime(tmp_path)
    r.config.pause_seconds = 1.2
    s = state()
    ex = Immediate()
    r.tick(s, ex, model, now=100)
    assert ex.calls == 0
    r.tick(s, ex, model, now=101.19)
    assert ex.calls == 0
    r.tick(s, ex, model, now=101.21)
    assert ex.calls == 1


def test_input_arriving_during_voice_dispatch_cancels_and_does_not_commit(tmp_path):
    r = runtime(tmp_path)
    s = state()
    ex = Immediate()
    original = s.agent.trigger

    def dispatch(**kw):
        result = original(**kw)
        s.records.append({"speaker": "人物3", "text": "除外は撤回します。", "end_ms": 3000})
        return result

    s.agent.trigger = dispatch
    r.tick(s, ex, model, now=100)
    r.tick(s, ex, model, now=101)
    assert not r.engine.counts
    assert s.agent.stops == 1
    assert "input_changed_during_dispatch" in r.log_path.read_text()


def test_expiry_during_voice_dispatch_does_not_commit(tmp_path, monkeypatch):
    import das.study.live as live

    clock = [100.0]
    monkeypatch.setattr(live.time, "monotonic", lambda: clock[0])
    r = runtime(tmp_path)
    s = state()
    ex = Immediate()

    def dispatch(**_kw):
        clock[0] += 20
        return True

    s.agent.trigger = dispatch
    r.tick(s, ex, model, now=100)
    r.tick(s, ex, model, now=101)
    assert not r.engine.counts
    assert s.agent.stops == 1


@pytest.mark.parametrize("error", [TimeoutError("synthetic timeout"), ConnectionError("synthetic disconnect")])
def test_transient_failure_can_recover_on_next_input(tmp_path, error):
    r = runtime(tmp_path)
    s = state()
    ex = Immediate()

    def fail(*_):
        raise error

    r.tick(s, ex, fail, now=100)
    r.tick(s, ex, fail, now=101)
    s.records[1]["text"] += "理由を確認しましょう。"
    r.tick(s, ex, model, now=102)
    r.tick(s, ex, model, now=103)
    assert len(s.agent.spoken) == 1
    assert ex.calls == 2


def test_rejected_voice_dispatch_expires_without_consuming_count(tmp_path):
    r = runtime(tmp_path)
    s = state()
    s.agent.trigger = lambda **_: False
    ex = Immediate()
    r.tick(s, ex, model, now=100)
    r.tick(s, ex, model, now=101)
    assert r.pending is not None
    r.tick(s, ex, model, now=117)
    assert r.pending is None and not r.engine.counts
    assert '"reason": "expired"' in r.log_path.read_text()


def test_speaker_correction_invalidates_inflight_result(tmp_path):
    class Delayed:
        def __init__(self):
            self.future = Future()
        def submit(self, *_):
            return self.future

    r = runtime(tmp_path)
    s = state()
    ex = Delayed()
    r.tick(s, ex, model, now=100)
    s.records[0]["speaker"] = "人物3"
    ex.future.set_result(model())
    r.tick(s, ex, model, now=101)
    assert not s.agent.spoken and not r.view()["records"]
    assert "stale_llm_result" in r.log_path.read_text()


def test_twenty_minute_virtual_sessions(tmp_path):
    from das.study.preflight import run_preflight

    report = run_preflight(tmp_path / "load")
    assert report["status"] == "passed"
    assert [r["voice_requests"] for r in report["runs"]] == [12, 0]
    assert all(r["human_turns"] == 240 for r in report["runs"])

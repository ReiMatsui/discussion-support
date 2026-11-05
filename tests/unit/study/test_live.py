from __future__ import annotations

import json
from concurrent.futures import Future

import pytest

from tests.support.study import Immediate, model, runtime, state


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

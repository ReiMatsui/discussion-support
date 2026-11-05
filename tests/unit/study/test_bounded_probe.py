import json
import urllib.error
from pathlib import Path

import pytest

from das.study import bounded_probe as probe


def test_plan_is_offline_and_does_not_reserve(tmp_path, monkeypatch):
    ledger = tmp_path / "usage.jsonl"
    monkeypatch.setattr(probe, "DEFAULT_LEDGER", ledger)
    monkeypatch.setattr(probe, "send_once", lambda *_: pytest.fail("network forbidden"))
    assert probe.main([]) == 0
    assert not ledger.exists()


def test_reservation_stops_fourth_request_and_never_refunds(tmp_path):
    ledger = tmp_path / "usage.jsonl"
    for i in range(3):
        assert probe.ProbeUsage(ledger).reserve("gpt-5.4-mini", b"{}", 6000) == i + 1
    with pytest.raises(ValueError, match="3回"):
        probe.ProbeUsage(ledger).reserve("gpt-5.4-mini", b"{}", 6000)
    assert len(ledger.read_text().splitlines()) == 3


def test_large_input_is_rejected(tmp_path):
    data = json.loads(Path("study/synthetic_replay.json").read_text())
    data["steps"][1]["turns"][0]["text"] = "あ" * 40000
    fixture = tmp_path / "large.json"
    fixture.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="32KiB"):
        probe.build_request(fixture, "gpt-5.4-mini", 6000)


def test_output_limit_cannot_be_increased():
    with pytest.raises(ValueError):
        probe.build_request(Path("study/synthetic_replay.json"), "gpt-5.4-mini", 6001)


def test_http_failure_has_one_call_no_secret_and_consumes_slot(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(probe, "DEFAULT_LEDGER", tmp_path / "usage.jsonl")
    calls = []

    def fail(*_):
        calls.append(True)
        raise urllib.error.HTTPError("https://example.invalid", 401, "secret-fragment", {}, None)

    monkeypatch.setattr(probe, "send_once", fail)
    out = tmp_path / "result.json"
    assert probe.main(["--execute", "--out", str(out)]) == 2
    assert len(calls) == 1
    report = json.loads(out.read_text())
    assert report["attempt"] == 1 and report["http_status"] == 401
    assert "secret-fragment" not in out.read_text()
    with pytest.raises(FileExistsError):
        probe.main(["--execute", "--out", str(out)])
    assert len(calls) == 1


def test_incomplete_response_keeps_usage_without_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(probe, "DEFAULT_LEDGER", tmp_path / "usage.jsonl")
    monkeypatch.setattr(probe, "send_once", lambda *_: {
        "usage": {"prompt_tokens": 100, "completion_tokens": 6000},
        "choices": [{"finish_reason": "length", "message": {"content": ""}}],
    })
    out = tmp_path / "result.json"
    assert probe.main(["--execute", "--out", str(out)]) == 2
    report = json.loads(out.read_text())
    assert report["usage"]["completion_tokens"] == 6000
    assert report["finish_reason"] == "length"


def test_success_records_usage_and_grounded_output(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(probe, "DEFAULT_LEDGER", tmp_path / "usage.jsonl")
    review = json.loads(Path("study/synthetic_replay.json").read_text())["steps"][1]["review"]
    monkeypatch.setattr(probe, "send_once", lambda *_: {
        "usage": {"prompt_tokens": 100, "completion_tokens": 200},
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(review)}}],
    })
    out = tmp_path / "success.json"
    assert probe.main(["--execute", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["decision"] == "prepared"
    assert report["usage"]["prompt_tokens"] == 100


def test_concurrent_runs_share_three_slot_limit(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    ledger = tmp_path / "usage.jsonl"

    def reserve(_):
        try:
            return probe.ProbeUsage(ledger).reserve("gpt-5.4-mini", b"{}", 6000)
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reserve, range(10)))
    assert sorted(r for r in results if r is not None) == [1, 2, 3]


def test_budget_refusal_does_not_send_or_consume_attempt(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")
    monkeypatch.setattr(probe, "DEFAULT_LEDGER", tmp_path / "usage.jsonl")
    monkeypatch.setattr(probe, "send_once", lambda *_: pytest.fail("over budget"))
    out = tmp_path / "refused.json"
    assert probe.main(["--execute", "--budget-usd", "0.001", "--out", str(out)]) == 2
    report = json.loads(out.read_text())
    assert report["failed_phase"] == "budget_reservation"
    assert "予算" in report["message"]
    assert report["cumulative_usage"]["attempts"] == 0


def test_live_and_manual_probe_share_extraction_payload():
    from das.study.live import build_review_params

    config, turns, new_ids, params, _ = probe.build_request(
        Path("study/synthetic_replay.json"), "gpt-5.4-mini", 6000
    )
    live = build_review_params(config, turns, new_ids, "gpt-5.4-mini")
    assert params["messages"] == live["messages"]
    assert params["response_format"] == live["response_format"]

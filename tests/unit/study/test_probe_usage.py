import json

import pytest

from das.llm.cost import CostTracker
from das.study.probe_usage import ProbeLimitError, ProbeUsage, reserve_estimate

MODEL = "gpt-5.4-mini"


def test_completed_usage_uses_existing_cost_tracker_and_survives_restart(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = ProbeUsage(path)
    n = ledger.reserve(MODEL, b"{}", 6000)
    ledger.finish(n, usage={"prompt_tokens": 100, "completion_tokens": 200}, status="completed")
    expected = CostTracker()
    expected.record(MODEL, 100, 200)
    snap = ProbeUsage(path).snapshot()
    assert snap["reported_usage"] == expected.snapshot()
    assert snap["held_estimate_usd"] == 0
    assert snap["unknown_usage_attempts"] == 0
    assert snap["remaining_calls"] == 2


def test_legacy_reservation_remains_counted_and_has_unknown_usage(tmp_path):
    path = tmp_path / "ledger.jsonl"
    old = {"attempt": 1, "model": MODEL, "request_bytes": 4198, "max_output_tokens": 6000}
    original = json.dumps(old) + "\n"
    path.write_text(original)
    ledger = ProbeUsage(path)
    snap = ledger.snapshot()
    assert snap["attempts"] == snap["unknown_usage_attempts"] == 1
    assert snap["held_estimate_usd"] == reserve_estimate(MODEL, 4198, 6000)
    assert ledger.reserve(MODEL, b"{}", 6000) == 2
    assert path.read_text().startswith(original)


def test_failed_and_interrupted_calls_keep_their_reservation(tmp_path):
    ledger = ProbeUsage(tmp_path / "ledger.jsonl")
    n = ledger.reserve(MODEL, b"{}", 6000)
    ledger.finish(n, usage=None, status="failed", http_status=401)
    ledger.reserve(MODEL, b"{}", 6000)  # second process crashes before finish
    snap = ledger.snapshot()
    assert snap["attempts"] == snap["unknown_usage_attempts"] == 2
    assert snap["held_estimate_usd"] > 0
    assert snap["reported_usage"]["n_calls"] == 0


def test_budget_rejects_before_request_or_reservation(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger = ProbeUsage(path, budget_usd=0.001)
    with pytest.raises(ProbeLimitError, match="予算"):
        ledger.reserve(MODEL, b"{}", 6000)
    assert ledger.snapshot()["attempts"] == 0


def test_unknown_model_cannot_use_cheap_fallback_for_gate(tmp_path):
    ledger = ProbeUsage(tmp_path / "ledger.jsonl")
    with pytest.raises(ProbeLimitError, match="料金表"):
        ledger.reserve("unpriced-expensive-model", b"{}", 100)
    assert ledger.snapshot()["attempts"] == 0


def test_invalid_usage_is_not_treated_as_free(tmp_path):
    ledger = ProbeUsage(tmp_path / "ledger.jsonl")
    n = ledger.reserve(MODEL, b"{}", 100)
    ledger.finish(n, usage={"prompt_tokens": -1, "completion_tokens": 10}, status="failed")
    assert ledger.snapshot()["unknown_usage_attempts"] == 1


def test_duplicate_result_cannot_double_count(tmp_path):
    ledger = ProbeUsage(tmp_path / "ledger.jsonl")
    n = ledger.reserve(MODEL, b"{}", 100)
    ledger.finish(n, usage={"prompt_tokens": 10, "completion_tokens": 10}, status="completed")
    with pytest.raises(ValueError, match="二重"):
        ledger.finish(n, usage={"prompt_tokens": 10, "completion_tokens": 10}, status="completed")
    assert ledger.snapshot()["reported_usage"]["n_calls"] == 1


@pytest.mark.parametrize("budget", [float("nan"), float("inf"), -1, 0])
def test_invalid_budget_is_rejected(tmp_path, budget):
    with pytest.raises(ValueError):
        ProbeUsage(tmp_path / "ledger.jsonl", budget_usd=budget)


def test_budget_cannot_silently_change_after_first_reservation(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ProbeUsage(path).reserve(MODEL, b"{}", 100)
    with pytest.raises(ProbeLimitError, match="保存済みの予算"):
        ProbeUsage(path, budget_usd=10).reserve(MODEL, b"{}", 100)

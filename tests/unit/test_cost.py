"""CostTracker / budget enforcement のユニットテスト。"""

from __future__ import annotations

import asyncio

import pytest

from das.llm.cost import (
    PRICING,
    BudgetExceeded,
    CostTracker,
    ModelPricing,
    resolve_pricing,
)


# --- ModelPricing -----------------------------------------------------


def test_model_pricing_cost_basic() -> None:
    p = ModelPricing(input_per_token=1e-6, output_per_token=2e-6)
    assert p.cost(100, 50) == pytest.approx(100e-6 + 100e-6)


def test_pricing_table_all_positive() -> None:
    for name, p in PRICING.items():
        assert p.input_per_token >= 0, f"{name} input price must be >= 0"
        assert p.output_per_token >= 0, f"{name} output price must be >= 0"


# --- resolve_pricing -------------------------------------------------


def test_resolve_pricing_exact_match() -> None:
    p = resolve_pricing("gpt-5-mini")
    assert p == PRICING["gpt-5-mini"]


def test_resolve_pricing_prefix_match_specific() -> None:
    # 「gpt-5-mini-2025-08-07」は「gpt-5-mini」にマッチすべき (= gpt-5 ではない)
    p = resolve_pricing("gpt-5-mini-2025-08-07")
    assert p == PRICING["gpt-5-mini"]


def test_resolve_pricing_falls_back_to_default() -> None:
    # 未知モデルは保守的に gpt-5-mini と同じ価格
    p = resolve_pricing("totally-unknown-model")
    assert p.input_per_token == PRICING["gpt-5-mini"].input_per_token
    assert p.output_per_token == PRICING["gpt-5-mini"].output_per_token


# --- CostTracker (集計) ---------------------------------------------


def test_tracker_records_cost_correctly() -> None:
    t = CostTracker()  # budget なし
    cumulative = t.record("gpt-5-mini", 1000, 500)
    expected = (1000 * 0.40e-6) + (500 * 1.60e-6)
    assert cumulative == pytest.approx(expected)
    assert t.total_usd == pytest.approx(expected)
    assert t.n_calls == 1


def test_tracker_accumulates_multiple_calls() -> None:
    t = CostTracker()
    t.record("gpt-5-mini", 1000, 500)
    t.record("gpt-5-mini", 1000, 500)
    expected = 2 * ((1000 * 0.40e-6) + (500 * 1.60e-6))
    assert t.total_usd == pytest.approx(expected)
    assert t.n_calls == 2


def test_tracker_separates_by_model() -> None:
    t = CostTracker()
    t.record("gpt-5-mini", 1000, 100)
    t.record("text-embedding-3-small", 5000, 0)
    assert "gpt-5-mini" in t.by_model
    assert "text-embedding-3-small" in t.by_model
    assert t.by_model["gpt-5-mini"].n_calls == 1
    assert t.by_model["text-embedding-3-small"].output_tokens == 0


# --- CostTracker (budget) -------------------------------------------


def test_tracker_raises_when_over_budget() -> None:
    t = CostTracker(budget_usd=1e-3)  # 約 $0.001
    # gpt-5-mini で 1000 input + 500 output = $0.0012 → 上限超過
    with pytest.raises(BudgetExceeded):
        t.record("gpt-5-mini", 1000, 500)


def test_tracker_no_raise_when_under_budget() -> None:
    t = CostTracker(budget_usd=1.0)
    t.record("gpt-5-mini", 100, 50)  # ≈ $0.00012、はるかに下
    assert not t.is_over_budget()


def test_tracker_check_before_call_raises_if_already_over() -> None:
    t = CostTracker(budget_usd=1e-3)
    with pytest.raises(BudgetExceeded):
        t.record("gpt-5-mini", 10000, 10000)
    # その後の check_before_call も raise する
    with pytest.raises(BudgetExceeded):
        t.check_before_call()


def test_tracker_check_before_call_ok_when_under_budget() -> None:
    t = CostTracker(budget_usd=1.0)
    t.record("gpt-5-mini", 100, 50)
    # 余裕があるので raise しない
    t.check_before_call()


def test_tracker_invalid_budget() -> None:
    with pytest.raises(ValueError):
        CostTracker(budget_usd=0)
    with pytest.raises(ValueError):
        CostTracker(budget_usd=-1.0)


# --- snapshot --------------------------------------------------------


def test_tracker_snapshot_returns_serializable() -> None:
    import json

    t = CostTracker(budget_usd=0.5)
    t.record("gpt-5-mini", 1000, 500)
    t.record("text-embedding-3-small", 2000, 0)
    snap = t.snapshot()
    # JSON 化できる
    json.dumps(snap)
    assert snap["total_usd"] > 0
    assert snap["budget_usd"] == 0.5
    assert snap["remaining_usd"] is not None
    assert snap["n_calls"] == 2
    assert "gpt-5-mini" in snap["by_model"]


def test_tracker_remaining_usd_when_no_budget() -> None:
    t = CostTracker()
    assert t.remaining_usd() is None


def test_tracker_format_status() -> None:
    t = CostTracker(budget_usd=1.0)
    t.record("gpt-5-mini", 1000, 500)
    s = t.format_status()
    assert "$" in s
    assert "calls" in s


# --- 並列 (asyncio) で record しても整合性が保たれる ---------------


async def test_tracker_safe_under_concurrent_record() -> None:
    """asyncio は単一スレッドなので、record の中に await が無ければ
    並列 task からの呼び出しでも壊れない。"""

    t = CostTracker()
    n_tasks = 50

    async def _bump() -> None:
        t.record("gpt-5-mini", 100, 50)

    await asyncio.gather(*[_bump() for _ in range(n_tasks)])
    assert t.n_calls == n_tasks
    expected = n_tasks * ((100 * 0.40e-6) + (50 * 1.60e-6))
    assert t.total_usd == pytest.approx(expected)


async def test_tracker_concurrent_budget_exceeded() -> None:
    """並列 record で 1 つでも budget を超えたら BudgetExceeded が伝播する。"""

    # 1 call で $0.00012 程度。budget $0.0005 なら 4-5 回で超える
    t = CostTracker(budget_usd=5e-4)
    n_tasks = 50

    async def _bump() -> None:
        try:
            t.record("gpt-5-mini", 100, 50)
        except BudgetExceeded:
            pass  # 最初に超えたタスク以降は止まる

    await asyncio.gather(*[_bump() for _ in range(n_tasks)])
    # 全 task が実行されるが、budget は最終的に超過状態
    assert t.is_over_budget()


# --- warning ---------------------------------------------------------


def test_tracker_warning_threshold_default_80pct() -> None:
    """warning は budget の 80% 到達時に 1 度だけ出る。
    本テストでは warning ログを直接観測せず、内部フラグで確認する。"""

    t = CostTracker(budget_usd=0.001, warn_at_pct=0.5)
    # 50% 未満
    t.record("gpt-5-mini", 100, 50)  # ≈ $0.00012 = 12% of 0.001
    assert not t._warning_emitted  # type: ignore[attr-defined]
    # 50% 超え (ただし budget 内)
    t.record("gpt-5-mini", 400, 200)  # +$0.00048 → 累積 ≈ $0.0006 = 60%
    assert t._warning_emitted  # type: ignore[attr-defined]

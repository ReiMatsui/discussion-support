"""CostTracker / budget enforcement のユニットテスト。"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from das.llm.cost import (
    PRICING,
    BudgetExceededError,
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


def test_resolve_pricing_gpt54_series() -> None:
    """GPT-5.4 mini/nano の料金が解決でき、旧 gpt-5 とは別行になる (model_update)。"""

    mini = resolve_pricing("gpt-5.4-mini")
    assert mini.input_per_token == pytest.approx(0.75e-6)
    assert mini.output_per_token == pytest.approx(4.50e-6)
    nano = resolve_pricing("gpt-5.4-nano")
    assert nano.input_per_token == pytest.approx(0.20e-6)
    # 日付サフィックス付きでも 5.4-mini にマッチ (gpt-5 にフォールバックしない)
    dated = resolve_pricing("gpt-5.4-mini-2026-03-01")
    assert dated.input_per_token == pytest.approx(0.75e-6)
    # 旧 gpt-5-mini 行は rescore 用に残っている
    assert resolve_pricing("gpt-5-mini").input_per_token == pytest.approx(0.40e-6)


# --- G5: 集計が全呼び出しを拾う / soft 超過ログの到達性 ------------------


def test_record_aggregates_all_calls() -> None:
    """record した全呼び出しが n_calls / total / by_model に反映される。"""

    tracker = CostTracker()
    tracker.record("gpt-5-mini", 100, 50)
    tracker.record("gpt-5-mini", 200, 100)
    tracker.record("gpt-5-nano", 10, 5)

    assert tracker.n_calls == 3
    by_model = tracker.by_model
    assert by_model["gpt-5-mini"].n_calls == 2
    assert by_model["gpt-5-nano"].n_calls == 1
    # total は各モデル price × tokens の合計
    expected = (
        resolve_pricing("gpt-5-mini").cost(300, 150)
        + resolve_pricing("gpt-5-nano").cost(10, 5)
    )
    assert tracker.total_usd == pytest.approx(expected)


def test_soft_budget_exceeded_log_reachable_after_warning() -> None:
    """M-5: 80% warning 後でも soft budget 超過ログが到達する (専用フラグ)。

    旧実装は soft 超過ログの条件が `not self._warning_emitted` で、80% warning が先に
    フラグを立てるため到達不能だった。専用フラグ導入で超過を跨いだ最初の record で
    到達することを固定する。
    """

    # 1 呼び出しあたりのコストが分かるよう gpt-5-mini を使う
    unit = resolve_pricing("gpt-5-mini").cost(1000, 0)
    tracker = CostTracker(budget_usd=unit * 2)  # 2 呼び出し分を予算に

    # 1 回目: 50% (warning 閾値 80% 未満)
    tracker.record("gpt-5-mini", 1000, 0)
    assert tracker._warning_emitted is False
    assert tracker._soft_exceeded_emitted is False

    # 2 回目: 100% (>= 80% で warning が立つ。まだ超過はしていない)
    tracker.record("gpt-5-mini", 1000, 0)
    assert tracker._warning_emitted is True
    assert tracker._soft_exceeded_emitted is False

    # 3 回目: 予算超過 → soft 超過ログが到達 (専用フラグが立つ)
    tracker.record("gpt-5-mini", 1000, 0)
    assert tracker.is_over_budget() is True
    assert tracker._soft_exceeded_emitted is True


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


def test_tracker_soft_budget_does_not_raise_but_flags_skip() -> None:
    """soft budget 単独: 超過しても raise しない。should_skip_new_run が True になる。"""

    t = CostTracker(budget_usd=1e-3)
    # gpt-5-mini で 1000+500 = $0.0012 → soft budget 超過
    t.record("gpt-5-mini", 1000, 500)  # raise しない
    assert t.is_over_budget()
    assert t.should_skip_new_run()


def test_tracker_hard_budget_raises_when_exceeded() -> None:
    """hard budget 超過時のみ BudgetExceededError を raise。"""

    t = CostTracker(hard_budget_usd=1e-3)
    with pytest.raises(BudgetExceededError):
        t.record("gpt-5-mini", 1000, 500)


def test_tracker_soft_and_hard_combined() -> None:
    """soft で gate、hard で raise の組み合わせ。"""

    t = CostTracker(budget_usd=1e-3, hard_budget_usd=2e-3)
    # 1 回目: $0.0012 → soft 超過するが hard は超えない (gate のみ)
    t.record("gpt-5-mini", 1000, 500)
    assert t.is_over_budget()
    assert t.should_skip_new_run()
    assert not t.is_over_hard_budget()
    # 2 回目: 追加 $0.0012 → 累計 $0.0024 で hard 超過、raise する
    with pytest.raises(BudgetExceededError):
        t.record("gpt-5-mini", 1000, 500)


def test_tracker_no_raise_when_under_budget() -> None:
    t = CostTracker(budget_usd=1.0)
    t.record("gpt-5-mini", 100, 50)  # ≈ $0.00012、はるかに下
    assert not t.is_over_budget()
    assert not t.should_skip_new_run()


def test_tracker_check_before_call_only_raises_for_hard() -> None:
    """check_before_call は hard budget のみで raise。"""

    # soft だけ → 超過していても raise しない
    t_soft = CostTracker(budget_usd=1e-3)
    t_soft.record("gpt-5-mini", 1000, 500)  # soft 超過
    t_soft.check_before_call()  # raise しない
    # hard だけ → 超過すると raise
    t_hard = CostTracker(hard_budget_usd=1e-3)
    with pytest.raises(BudgetExceededError):
        t_hard.record("gpt-5-mini", 1000, 500)
    with pytest.raises(BudgetExceededError):
        t_hard.check_before_call()


def test_tracker_check_before_call_ok_when_under_budget() -> None:
    t = CostTracker(budget_usd=1.0, hard_budget_usd=2.0)
    t.record("gpt-5-mini", 100, 50)
    t.check_before_call()  # 余裕があるので raise しない


def test_tracker_invalid_budget() -> None:
    with pytest.raises(ValueError):
        CostTracker(budget_usd=0)
    with pytest.raises(ValueError):
        CostTracker(budget_usd=-1.0)
    with pytest.raises(ValueError):
        CostTracker(hard_budget_usd=0)
    with pytest.raises(ValueError):
        # hard < soft はエラー
        CostTracker(budget_usd=1.0, hard_budget_usd=0.5)


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


async def test_tracker_concurrent_hard_budget_exceeded() -> None:
    """並列 record で hard budget を超えたら BudgetExceededError が伝播する。"""

    # 1 call で $0.00012 程度。hard $0.0005 なら 4-5 回で超える
    t = CostTracker(hard_budget_usd=5e-4)
    n_tasks = 50

    async def _bump() -> None:
        with contextlib.suppress(BudgetExceededError):
            t.record("gpt-5-mini", 100, 50)

    await asyncio.gather(*[_bump() for _ in range(n_tasks)])
    # 全 task が実行されるが、hard budget は最終的に超過状態
    assert t.is_over_hard_budget()


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

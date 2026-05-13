"""OpenAI 使用量と推定コストの集計 / 予算上限の enforcement。

設計上の決め事:
  - **asyncio 単一スレッド前提**: ``CostTracker.record`` の中に ``await`` が無いため、
    複数 coroutine が並列に呼んでも互いに割り込まない (= 排他制御不要)
  - **逐次表示**: ``CostTracker.snapshot()`` を呼ぶと現時点の累積を取得できる
  - **上限超過は例外で停止**: ``budget_usd`` 設定時、超過したら ``BudgetExceeded``
    を raise する。呼び出し側 (CLI / run_eval) は catch して部分結果を保存して exit する設計
  - **料金は静的辞書**: 公開料金を埋め込み (2026-05 時点)。未知モデルは保守的に
    gpt-5-mini と同じ料金で扱う

Usage:
    tracker = CostTracker(budget_usd=1.0)
    llm = OpenAIClient(cost_tracker=tracker)
    # ... eval を回す ...
    # BudgetExceeded が raise されたら、tracker.snapshot() を保存して exit

Reference:
  - OpenAI 公開料金 https://openai.com/api/pricing/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from das.logging import get_logger


@dataclass(frozen=True)
class ModelPricing:
    """1 トークンあたりの USD 料金。

    Attributes:
      input_per_token: 入力 (prompt) トークンの単価
      output_per_token: 出力 (completion) トークンの単価。embedding は 0
    """

    input_per_token: float
    output_per_token: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens * self.input_per_token + output_tokens * self.output_per_token


# OpenAI 公開料金 (USD/M tokens を /token に換算)。2026-05 時点。
# 必要に応じて環境変数経由で上書き可能にする拡張は将来案。
PRICING: dict[str, ModelPricing] = {
    # GPT-5 series
    "gpt-5-mini": ModelPricing(0.40e-6, 1.60e-6),
    "gpt-5": ModelPricing(1.25e-6, 10.00e-6),
    "gpt-5-nano": ModelPricing(0.05e-6, 0.40e-6),
    # GPT-4o series (legacy)
    "gpt-4o-mini": ModelPricing(0.15e-6, 0.60e-6),
    "gpt-4o": ModelPricing(2.50e-6, 10.00e-6),
    # Reasoning (o-series)
    "o1-mini": ModelPricing(3.00e-6, 12.00e-6),
    "o1": ModelPricing(15.00e-6, 60.00e-6),
    "o3-mini": ModelPricing(1.10e-6, 4.40e-6),
    # Embeddings
    "text-embedding-3-small": ModelPricing(0.02e-6, 0.0),
    "text-embedding-3-large": ModelPricing(0.13e-6, 0.0),
    # Legacy
    "gpt-3.5-turbo": ModelPricing(0.50e-6, 1.50e-6),
}

# 不明モデルにはこの保守的な料金を当てる (= gpt-5-mini と同じ)。
# 不明モデルで cost が「過小」評価されないようにする
_DEFAULT_PRICING = ModelPricing(0.40e-6, 1.60e-6)


class BudgetExceeded(RuntimeError):
    """累積コストが上限を超過した。呼び出し側で catch して部分結果を保存して exit する。"""


@dataclass
class ModelUsage:
    """1 モデル分の累積使用量。"""

    n_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def resolve_pricing(model: str) -> ModelPricing:
    """モデル名から料金を解決する。

    解決ロジック:
      1. 完全一致
      2. 長い prefix から順に startswith 一致 (例: "gpt-5-mini-2025-08-07" は
         "gpt-5-mini" にマッチ。"gpt-5" より先に "gpt-5-mini" を試す)
      3. それでも見つからなければ ``_DEFAULT_PRICING``
    """

    if model in PRICING:
        return PRICING[model]
    for known in sorted(PRICING.keys(), key=len, reverse=True):
        if model.startswith(known):
            return PRICING[known]
    return _DEFAULT_PRICING


class CostTracker:
    """OpenAI 使用量と推定コストの集計 + 予算 enforcement。

    Args:
      budget_usd: 上限 USD。``None`` なら enforcement なし (集計のみ)。
      warn_at_pct: 残り何 % で warning ログを出すか (0-1)。例: 0.8 で 80% 到達時
    """

    def __init__(
        self,
        budget_usd: float | None = None,
        *,
        warn_at_pct: float = 0.8,
    ) -> None:
        if budget_usd is not None and budget_usd <= 0:
            raise ValueError("budget_usd must be > 0 or None")
        if not (0.0 <= warn_at_pct <= 1.0):
            raise ValueError("warn_at_pct must be in [0, 1]")
        self._budget = budget_usd
        self._warn_at_pct = warn_at_pct
        self._total = 0.0
        self._by_model: dict[str, ModelUsage] = {}
        self._warning_emitted = False
        self._log = get_logger("das.llm.cost")

    # --- 読み出し ----------------------------------------------------

    @property
    def total_usd(self) -> float:
        return self._total

    @property
    def budget_usd(self) -> float | None:
        return self._budget

    @property
    def n_calls(self) -> int:
        return sum(u.n_calls for u in self._by_model.values())

    @property
    def by_model(self) -> dict[str, ModelUsage]:
        """読み取り専用ビュー (mutate 禁止)。"""

        return dict(self._by_model)

    def is_over_budget(self) -> bool:
        return self._budget is not None and self._total > self._budget

    def remaining_usd(self) -> float | None:
        if self._budget is None:
            return None
        return max(0.0, self._budget - self._total)

    # --- 記録と enforcement ----------------------------------------

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """1 つの API 呼び出しを記録し、累積 cost (USD) を返す。

        budget が設定済みで超過したら ``BudgetExceeded`` を raise する。
        """

        pricing = resolve_pricing(model)
        cost = pricing.cost(input_tokens, output_tokens)
        self._total += cost
        usage = self._by_model.setdefault(model, ModelUsage())
        usage.n_calls += 1
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens
        usage.cost_usd += cost

        self._log.debug(
            "cost.record",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 6),
            cumulative_usd=round(self._total, 6),
        )

        # warning 閾値到達 (1 度だけ出す)
        if (
            self._budget is not None
            and not self._warning_emitted
            and self._total >= self._budget * self._warn_at_pct
            and self._total <= self._budget
        ):
            self._warning_emitted = True
            self._log.warning(
                "cost.budget_warning",
                cumulative_usd=round(self._total, 4),
                budget_usd=round(self._budget, 4),
                pct=round(100.0 * self._total / self._budget, 1),
            )

        if self.is_over_budget():
            self._log.error(
                "cost.budget_exceeded",
                cumulative_usd=round(self._total, 4),
                budget_usd=round(self._budget, 4) if self._budget else None,
                n_calls=self.n_calls,
            )
            raise BudgetExceeded(
                f"Budget exceeded: ${self._total:.4f} > ${self._budget:.4f} "
                f"after {self.n_calls} API calls"
            )

        return self._total

    def check_before_call(self) -> None:
        """新しい API 呼び出しの直前チェック。すでに超過していたら raise する。

        並列で走る他 task が record で raise してから停止が伝播する間に
        新しい呼び出しが入ってしまうのを防ぐ第二防衛線。
        """

        if self.is_over_budget():
            raise BudgetExceeded(
                f"Budget already exceeded: ${self._total:.4f} > "
                f"${self._budget:.4f} (call refused before issue)"
            )

    # --- スナップショット ------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """JSON に dump しやすい dict を返す。"""

        return {
            "total_usd": round(self._total, 6),
            "budget_usd": self._budget,
            "remaining_usd": (
                round(self.remaining_usd(), 6) if self._budget is not None else None
            ),
            "n_calls": self.n_calls,
            "over_budget": self.is_over_budget(),
            "by_model": {
                name: {
                    "n_calls": u.n_calls,
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "cost_usd": round(u.cost_usd, 6),
                }
                for name, u in self._by_model.items()
            },
        }

    def format_status(self) -> str:
        """1 行サマリ (CLI 表示用)。"""

        budget_str = ""
        if self._budget is not None:
            pct = 100.0 * self._total / self._budget
            budget_str = f" / ${self._budget:.2f} ({pct:.1f}%)"
        return f"${self._total:.4f}{budget_str}  [{self.n_calls} calls]"


__all__ = [
    "BudgetExceeded",
    "CostTracker",
    "ModelPricing",
    "ModelUsage",
    "PRICING",
    "resolve_pricing",
]

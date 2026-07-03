"""OpenAI 使用量と推定コストの集計 / 予算上限の enforcement。

設計上の決め事:
  - **asyncio 単一スレッド前提**: ``CostTracker.record`` の中に ``await`` が無いため、
    複数 coroutine が並列に呼んでも互いに割り込まない (= 排他制御不要)
  - **逐次表示**: ``CostTracker.snapshot()`` を呼ぶと現時点の累積を取得できる
  - **上限超過は例外で停止**: ``budget_usd`` 設定時、超過したら ``BudgetExceededError``
    を raise する。呼び出し側 (CLI / run_eval) は catch して部分結果を保存して exit する設計
  - **料金は静的辞書**: 公開料金を埋め込み (2026-05 時点)。未知モデルは保守的に
    gpt-5-mini と同じ料金で扱う

Usage:
    tracker = CostTracker(budget_usd=1.0)
    llm = OpenAIClient(cost_tracker=tracker)
    # ... eval を回す ...
    # BudgetExceededError が raise されたら、tracker.snapshot() を保存して exit

Reference:
  - OpenAI 公開料金 https://openai.com/api/pricing/
"""

from __future__ import annotations

from dataclasses import dataclass
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
    # GPT-5.4 series (2026-03〜, 現行の既定。05章 要対応1)
    "gpt-5.4-mini": ModelPricing(0.75e-6, 4.50e-6),
    "gpt-5.4-nano": ModelPricing(0.20e-6, 1.25e-6),
    # GPT-5 series (旧・過去 run の rescore 用に残す)
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


class BudgetExceededError(RuntimeError):
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

    2 段の予算ゲート:
      - **soft budget** (``budget_usd``): 超過すると ``should_skip_new_run()`` が True
        を返す。**進行中の API 呼び出しは止めない**。run_eval は新規 run の開始だけを
        gate するため、in-flight run は最後まで完走する。
      - **hard budget** (``hard_budget_usd``): 超過すると ``record()`` / ``check_before_call()``
        が ``BudgetExceededError`` を即 raise する。暴走防止用の絶対上限。

    Args:
      budget_usd: soft 上限 USD。``None`` なら gate なし。
      hard_budget_usd: hard 絶対上限 USD。``None`` なら hard cap なし。
      warn_at_pct: soft budget の何 % で warning ログを出すか (0-1)。

    Notes:
      ``hard_budget_usd`` だけ指定して ``budget_usd`` を省略すると、
      従来挙動 (= 上限超過で即停止) と同じ動きになる。
    """

    def __init__(
        self,
        budget_usd: float | None = None,
        *,
        hard_budget_usd: float | None = None,
        warn_at_pct: float = 0.8,
    ) -> None:
        if budget_usd is not None and budget_usd <= 0:
            raise ValueError("budget_usd must be > 0 or None")
        if hard_budget_usd is not None and hard_budget_usd <= 0:
            raise ValueError("hard_budget_usd must be > 0 or None")
        if (
            budget_usd is not None
            and hard_budget_usd is not None
            and hard_budget_usd < budget_usd
        ):
            raise ValueError("hard_budget_usd must be >= budget_usd")
        if not (0.0 <= warn_at_pct <= 1.0):
            raise ValueError("warn_at_pct must be in [0, 1]")
        self._budget = budget_usd
        self._hard_budget = hard_budget_usd
        self._warn_at_pct = warn_at_pct
        self._total = 0.0
        self._by_model: dict[str, ModelUsage] = {}
        self._warning_emitted = False
        # soft budget 超過ログ専用フラグ (M-5: 80% warning フラグと共用すると到達不能だった)
        self._soft_exceeded_emitted = False
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

    @property
    def hard_budget_usd(self) -> float | None:
        return self._hard_budget

    def is_over_budget(self) -> bool:
        """soft budget を超過しているか (新規 run 開始の gate 判定用)。"""

        return self._budget is not None and self._total > self._budget

    def is_over_hard_budget(self) -> bool:
        """hard budget を超過しているか (record/check で raise する判定用)。"""

        return self._hard_budget is not None and self._total > self._hard_budget

    def should_skip_new_run(self) -> bool:
        """**新規 run を開始すべきでない**か (= soft budget 超過)。

        進行中の run は止めない。run_eval が job 入口でこれをチェックする。
        """

        return self.is_over_budget()

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

        budget が設定済みで超過したら ``BudgetExceededError`` を raise する。
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

        # soft budget 超過時はログのみ (新規 run は run_eval 側で gate)。
        # M-5: 80% warning が _warning_emitted を立てるため、旧実装の
        # `not self._warning_emitted` 条件はこの info ログをほぼ到達不能にしていた。
        # 専用フラグ _soft_exceeded_emitted で「超過を跨いだ最初の 1 回」に出す。
        if self.is_over_budget() and not self._soft_exceeded_emitted:
            self._soft_exceeded_emitted = True
            self._log.info(
                "cost.soft_budget_exceeded",
                cumulative_usd=round(self._total, 4),
                budget_usd=round(self._budget, 4) if self._budget else None,
                note="new runs will be skipped; in-flight runs continue",
            )

        # hard budget 超過時のみ即停止 (絶対上限の暴走防止)
        if self.is_over_hard_budget():
            self._log.error(
                "cost.hard_budget_exceeded",
                cumulative_usd=round(self._total, 4),
                hard_budget_usd=round(self._hard_budget, 4) if self._hard_budget else None,
                n_calls=self.n_calls,
            )
            raise BudgetExceededError(
                f"Hard budget exceeded: ${self._total:.4f} > "
                f"${self._hard_budget:.4f} after {self.n_calls} API calls"
            )

        return self._total

    def check_before_call(self) -> None:
        """新しい API 呼び出しの直前チェック。hard budget 超過時のみ raise。

        soft budget は run_eval 側で「新規 run の gate」として扱うため、
        in-flight な API 呼び出しはここでは止めない。
        """

        if self.is_over_hard_budget():
            raise BudgetExceededError(
                f"Hard budget already exceeded: ${self._total:.4f} > "
                f"${self._hard_budget:.4f} (call refused before issue)"
            )

    # --- スナップショット ------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """JSON に dump しやすい dict を返す。"""

        return {
            "total_usd": round(self._total, 6),
            "budget_usd": self._budget,
            "hard_budget_usd": self._hard_budget,
            "remaining_usd": (
                round(self.remaining_usd(), 6) if self._budget is not None else None
            ),
            "n_calls": self.n_calls,
            "over_budget": self.is_over_budget(),
            "over_hard_budget": self.is_over_hard_budget(),
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
    "PRICING",
    "BudgetExceededError",
    "CostTracker",
    "ModelPricing",
    "ModelUsage",
    "resolve_pricing",
]

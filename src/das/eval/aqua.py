"""AQuA — Additive deliberation Quality score with Adapters (LLM-judge 自前実装版).

Behrendt et al. "AQuA – Combining Experts' and Non-Experts' Views To Assess
Deliberation Quality in Online Discussions Using LLMs" (DELITE @ LREC-COLING 2024)
の 20 deliberation indicator を、公開アダプタ (ドイツ語学習) ではなく
**das の OpenAI client で LLM-as-judge** として再現する。

設計上の選択:
  - 20 indicator は **原論文 Table 1** の名前・定義・重み (correlation coefficient) をそのまま採用
  - スコアスケール {0, 1, 2, 3} も原論文に揃える
  - 集計式 (s_min ≈ -1.6693, s_max ≈ 4.9893, 0-5 にリスケール) も原論文 §3.4 に揃える
  - 採点は **per-utterance** (= per-comment)。議論全体のスコアは「発話別 AQuA の平均」
  - 1 発話あたり LLM 呼び出し 1 回 (20 indicator をまとめて構造化出力)
  - 文脈 (直前数発話) を渡すが、採点は対象発話のみを見る

論文と完全互換ではない (公開ドイツ語アダプタを使っていない) ので、論文には
"AQuA-inspired 20 indicators (LLM-judge re-implementation)" と書く。

Reference:
  Behrendt, M., Wagner, S.S., Ziegele, M., Wilms, L., Stoll, A., Heinbach, D.,
  Harmeling, S. (2024). AQuA – Combining Experts' and Non-Experts' Views To
  Assess Deliberation Quality in Online Discussions Using LLMs. In Proceedings
  of the First Workshop on Language-driven Deliberation Technology (DELITE) @
  LREC-COLING 2024, pp. 1-12.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, pstdev
from typing import Literal

from pydantic import BaseModel, Field

from das.agents.base import BaseAgent
from das.llm import OpenAIClient
from das.types import Utterance

_PROMPTS_DIR = Path(__file__).parent / "prompts"

Dimension = Literal["rationality", "reciprocity", "civility"]


@dataclass(frozen=True)
class AQuAIndicator:
    """1 つの deliberation indicator のメタ情報。"""

    name: str
    dimension: Dimension
    description: str
    weight: float
    """Behrendt et al. Table 1 の correlation coefficient (= aggregate 時の重み)。
    正の重みは「高品質コメントに positive」、負の重みは「negative trait」。"""


# 原論文 Table 1 をそのまま転記 (順序も保持)
INDICATORS: tuple[AQuAIndicator, ...] = (
    # --- Rationality ---
    AQuAIndicator(
        "relevance",
        "rationality",
        "Does the comment have a relevance for the discussed topic?",
        0.20908452,
    ),
    AQuAIndicator(
        "fact",
        "rationality",
        "Is there at least one fact claiming statement in the comment?",
        0.18285757,
    ),
    AQuAIndicator(
        "opinion",
        "rationality",
        "Is there a subjective statement made in the comment?",
        -0.11069402,
    ),
    AQuAIndicator(
        "justification",
        "rationality",
        "Is at least one statement justified in the comment?",
        0.29000763,
    ),
    AQuAIndicator(
        "solution_proposals",
        "rationality",
        "Does the comment contain a proposal how an issue could be solved?",
        0.39535126,
    ),
    AQuAIndicator(
        "additional_knowledge",
        "rationality",
        "Does the comment contain additional knowledge?",
        0.14655912,
    ),
    AQuAIndicator(
        "question",
        "rationality",
        "Does the comment include a true, i.e., non-rhetoric question?",
        -0.07331445,
    ),
    # --- Reciprocity ---
    AQuAIndicator(
        "referencing_users",
        "reciprocity",
        "Does the comment refer to at least one other user or to all users?",
        -0.03768367,
    ),
    AQuAIndicator(
        "referencing_medium",
        "reciprocity",
        "Does the comment refer to the medium, editorial team, or moderation?",
        0.07019062,
    ),
    AQuAIndicator(
        "referencing_contents",
        "reciprocity",
        "Does the comment refer to content, arguments, or positions in other comments?",
        -0.02847408,
    ),
    AQuAIndicator(
        "referencing_personal",
        "reciprocity",
        "Does the comment refer to the person or personal characteristics of other users?",
        0.21126469,
    ),
    AQuAIndicator(
        "referencing_format",
        "reciprocity",
        "Does the comment refer to the tone, language, spelling, or formal criteria?",
        -0.02674237,
    ),
    # --- Civility ---
    AQuAIndicator(
        "polite_form_of_address",
        "civility",
        "Does the comment contain welcome or farewell phrases?",
        0.01482095,
    ),
    AQuAIndicator(
        "respect",
        "civility",
        "Does the comment contain expressions of respect or thankfulness?",
        0.00732909,
    ),
    AQuAIndicator(
        "screaming",
        "civility",
        "Does the comment contain clusters of punctuation/caps intended to imply screaming?",
        -0.01900971,
    ),
    AQuAIndicator(
        "vulgar",
        "civility",
        "Does the comment contain language that is inappropriate for civil discourse?",
        -0.04995486,
    ),
    AQuAIndicator(
        "insult",
        "civility",
        "Does the comment contain insults towards one or more people?",
        -0.05884586,
    ),
    AQuAIndicator(
        "sarcasm",
        "civility",
        "Does the comment contain biting mockery aimed at devaluing the reference object?",
        -0.15170863,
    ),
    AQuAIndicator(
        "discrimination",
        "civility",
        "Does the comment explicitly or implicitly contain unfair treatment of groups or individuals?",
        0.02934227,
    ),
    AQuAIndicator(
        "storytelling",
        "civility",
        "Does the commenter include personal stories or personal experiences?",
        0.10628146,
    ),
)

assert len(INDICATORS) == 20, "AQuA は厳密に 20 indicator"

# 原論文 §3.4 の正規化定数。3 × Σ(w | w≥0) と 3 × Σ(w | w≤0)。
_S_MAX = sum(3.0 * ind.weight for ind in INDICATORS if ind.weight >= 0)
_S_MIN = sum(3.0 * ind.weight for ind in INDICATORS if ind.weight <= 0)
# 原論文の値: s_max ≈ 4.9893, s_min ≈ -1.6693


class AQuAScores(BaseModel):
    """LLM が返す 1 発話分の 20 indicator スコア (0-3 整数)。

    Field 名は ``INDICATORS`` の name と同一。Pydantic がそのまま受ける。
    """

    relevance: int = Field(ge=0, le=3)
    fact: int = Field(ge=0, le=3)
    opinion: int = Field(ge=0, le=3)
    justification: int = Field(ge=0, le=3)
    solution_proposals: int = Field(ge=0, le=3)
    additional_knowledge: int = Field(ge=0, le=3)
    question: int = Field(ge=0, le=3)
    referencing_users: int = Field(ge=0, le=3)
    referencing_medium: int = Field(ge=0, le=3)
    referencing_contents: int = Field(ge=0, le=3)
    referencing_personal: int = Field(ge=0, le=3)
    referencing_format: int = Field(ge=0, le=3)
    polite_form_of_address: int = Field(ge=0, le=3)
    respect: int = Field(ge=0, le=3)
    screaming: int = Field(ge=0, le=3)
    vulgar: int = Field(ge=0, le=3)
    insult: int = Field(ge=0, le=3)
    sarcasm: int = Field(ge=0, le=3)
    discrimination: int = Field(ge=0, le=3)
    storytelling: int = Field(ge=0, le=3)

    def aqua_value(self) -> float:
        """20 indicator スコアから AQuA 集計値 (0-5 リスケール後) を計算する。"""

        raw = sum(getattr(self, ind.name) * ind.weight for ind in INDICATORS)
        return 5.0 * (raw - _S_MIN) / (_S_MAX - _S_MIN)

    def as_dict(self) -> dict[str, int]:
        return {ind.name: getattr(self, ind.name) for ind in INDICATORS}


@dataclass(frozen=True)
class AQuAUtteranceReport:
    """1 発話分の AQuA 採点結果。"""

    turn_id: int
    speaker: str
    scores: AQuAScores
    aqua_value: float
    """0-5 にリスケール済みの AQuA スコア。"""


@dataclass(frozen=True)
class AQuADiscussionReport:
    """1 議論 (transcript) 分の AQuA 集計。"""

    n_utterances: int
    mean: float
    """発話別 AQuA の単純平均 (= 議論全体のスコア)。"""

    std: float
    per_utterance: list[AQuAUtteranceReport] = field(default_factory=list)
    per_indicator_mean: dict[str, float] = field(default_factory=dict)
    """20 indicator それぞれの 0-3 スコアの発話別平均 (= どの指標が効いているか)。"""

    def to_dict(self) -> dict:
        return {
            "n_utterances": self.n_utterances,
            "mean": self.mean,
            "std": self.std,
            "per_indicator_mean": self.per_indicator_mean,
            "per_utterance": [
                {
                    "turn_id": r.turn_id,
                    "speaker": r.speaker,
                    "aqua_value": r.aqua_value,
                    "scores": r.scores.as_dict(),
                }
                for r in self.per_utterance
            ],
        }


def _load_system_prompt() -> str:
    return (_PROMPTS_DIR / "aqua.md").read_text(encoding="utf-8")


def _format_context(history: list[Utterance], target: Utterance, *, n_context: int = 3) -> str:
    """target 発話直前 n_context 発話を context として整形する。"""

    idx = next((i for i, u in enumerate(history) if u.turn_id == target.turn_id), None)
    if idx is None:
        ctx_block = "(文脈情報なし)"
    else:
        start = max(0, idx - n_context)
        ctx_lines = [
            f"[t{u.turn_id}] {u.speaker}: {u.text}" for u in history[start:idx]
        ]
        ctx_block = "\n".join(ctx_lines) if ctx_lines else "(直前発話なし)"
    return ctx_block


class AQuAAgent(BaseAgent):
    """LLM-judge による AQuA スコア計算エージェント。"""

    name = "aqua"

    def __init__(self, llm: OpenAIClient | None = None) -> None:
        super().__init__(llm=llm)
        self._system_prompt = _load_system_prompt()

    async def score_utterance(
        self,
        target: Utterance,
        *,
        history: list[Utterance] | None = None,
        n_context: int = 3,
        model: str | None = None,
    ) -> AQuAUtteranceReport:
        """1 発話を採点する。"""

        history = history or []
        ctx = _format_context(history, target, n_context=n_context)
        user_content = (
            f"# 直前の文脈 (採点対象ではない、判定の補助にだけ使う)\n{ctx}\n\n"
            f"# 採点対象の発話\n[t{target.turn_id}] {target.speaker}: {target.text}\n\n"
            "# 指示\n"
            "上記の 1 発話を 20 indicator それぞれ 0-3 で採点してください。"
            "判定に迷ったら 1 か 2 の保守的な側に倒すこと (3 は「明らかに」と言える時のみ)。"
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]
        scores = await self.llm.chat_structured(
            messages,  # type: ignore[arg-type]
            response_format=AQuAScores,
            model=model,
        )
        report = AQuAUtteranceReport(
            turn_id=target.turn_id,
            speaker=target.speaker,
            scores=scores,
            aqua_value=scores.aqua_value(),
        )
        self.log.info(
            "aqua.utterance_scored",
            turn_id=target.turn_id,
            speaker=target.speaker,
            aqua_value=round(report.aqua_value, 3),
        )
        return report

    async def score_discussion(
        self,
        transcript: list[Utterance],
        *,
        n_context: int = 3,
        concurrency: int = 5,
        model: str | None = None,
    ) -> AQuADiscussionReport:
        """transcript 全体を採点し、議論レベルの AQuA 集計を返す。"""

        if not transcript:
            return AQuADiscussionReport(
                n_utterances=0, mean=0.0, std=0.0,
                per_utterance=[], per_indicator_mean={},
            )

        sem = asyncio.Semaphore(concurrency)

        async def _one(u: Utterance) -> AQuAUtteranceReport:
            async with sem:
                return await self.score_utterance(
                    u, history=transcript, n_context=n_context, model=model,
                )

        # 並列採点 (各発話は独立)
        reports = await asyncio.gather(*[_one(u) for u in transcript])
        # turn_id 順にソート (gather 結果の並びを保証)
        reports = sorted(reports, key=lambda r: r.turn_id)

        values = [r.aqua_value for r in reports]
        mean_val = fmean(values) if values else 0.0
        std_val = pstdev(values) if len(values) > 1 else 0.0

        per_indicator_mean: dict[str, float] = {}
        for ind in INDICATORS:
            scores = [getattr(r.scores, ind.name) for r in reports]
            per_indicator_mean[ind.name] = fmean(scores) if scores else 0.0

        self.log.info(
            "aqua.discussion_scored",
            n_utterances=len(reports),
            mean=round(mean_val, 3),
            std=round(std_val, 3),
        )
        return AQuADiscussionReport(
            n_utterances=len(reports),
            mean=mean_val,
            std=std_val,
            per_utterance=reports,
            per_indicator_mean=per_indicator_mean,
        )


def aggregate_aqua_reports(reports: list[AQuADiscussionReport]) -> dict:
    """複数ラン分の AQuA を condition 集計形式に揃える。"""

    if not reports:
        return {}
    means = [r.mean for r in reports]
    agg: dict = {
        "n_runs": len(reports),
        "mean": fmean(means),
        "std": pstdev(means) if len(means) > 1 else 0.0,
        "per_indicator_mean": {},
    }
    for ind in INDICATORS:
        per_run = [r.per_indicator_mean.get(ind.name, 0.0) for r in reports]
        agg["per_indicator_mean"][ind.name] = fmean(per_run) if per_run else 0.0
    return agg


__all__ = [
    "INDICATORS",
    "AQuAAgent",
    "AQuADiscussionReport",
    "AQuAIndicator",
    "AQuAScores",
    "AQuAUtteranceReport",
    "aggregate_aqua_reports",
]

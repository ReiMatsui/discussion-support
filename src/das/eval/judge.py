"""LLM-as-judge による主観指標の代理計測 (研究計画書 §5.1 対応)。

各ペルソナになりきった LLM が、議論終了後に以下を 7 段階で評価する:
  - overall_satisfaction (1-7)
  - information_usefulness (1-7)
  - opposition_understanding (1-7)
  - confidence_change (-3 〜 +3)
  - intervention_transparency (1-7)

これは段階B シミュレーション評価で RQ1〜RQ3 に答えるための代理測定で、
段階C 対面実験のアンケートと同形式に揃える設計。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev

from pydantic import BaseModel, Field

from das.agents.base import BaseAgent
from das.eval.conditions import InterventionLogEntry, relation_label
from das.eval.persona import PersonaSpec
from das.llm import OpenAIClient
from das.types import Utterance

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_STANCE_DESCRIPTION: dict[str, str] = {
    "pro": "提案や変更に賛成の立場 (現状を変える側を後押しする)",
    "con": "提案や変更に反対の立場 (現状維持またはコスト・リスクを重視する)",
    "neutral": "中立。両方の論点に耳を傾け、論拠の強さで判断する",
}


class JudgeScores(BaseModel):
    """LLM が返す 1 ペルソナぶんのスコア。

    各指標に **per-metric の根拠 (reason)** を持たせ、ジャッジが
    なぜそのスコアを付けたかが UI / ログから直接読める設計。
    """

    overall_satisfaction: int = Field(ge=1, le=7)
    overall_satisfaction_reason: str = Field(default="", description="この点数を付けた具体的な根拠")
    information_usefulness: int = Field(ge=1, le=7)
    information_usefulness_reason: str = Field(default="")
    opposition_understanding: int = Field(ge=1, le=7)
    opposition_understanding_reason: str = Field(default="")
    confidence_change: int = Field(ge=-3, le=3)
    confidence_change_reason: str = Field(default="")
    intervention_transparency: int = Field(ge=1, le=7)
    intervention_transparency_reason: str = Field(default="")
    rationale: str = Field(default="", description="議論全体への総評")


@dataclass(frozen=True)
class JudgeReport:
    """1 ペルソナぶんの評価結果 (条件・トピック付き)。"""

    persona_name: str
    condition_name: str
    topic: str
    scores: JudgeScores


def _load_system_prompt() -> str:
    return (_PROMPTS_DIR / "judge.md").read_text(encoding="utf-8")


def _format_transcript(transcript: list[Utterance]) -> str:
    return "\n".join(f"[t{u.turn_id}] {u.speaker}: {u.text}" for u in transcript)


def _format_info_log_for_persona(
    info_log: list[InterventionLogEntry], persona_name: str
) -> str:
    rows = [e for e in info_log if e.persona_name == persona_name]
    if not rows:
        return "(あなた向けの参考情報は提示されませんでした)"
    lines: list[str] = []
    for entry in rows:
        lines.append(f"--- ターン {entry.turn_id} ---")
        if not entry.items:
            lines.append("  (項目なし)")
            continue
        for item in entry.items:
            tag = relation_label(str(item.get("relation") or ""))
            lines.append(f"  [{tag}] {item.get('source_text', '')}")
    return "\n".join(lines)


class JudgeAgent(BaseAgent):
    """ペルソナになりきって 7 段階評価を返す LLM ジャッジ。"""

    name = "judge"

    def __init__(
        self,
        llm: OpenAIClient | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        super().__init__(llm=llm)
        self._system_prompt = _load_system_prompt()
        self._model = model
        self._temperature = temperature

    def _build_messages(
        self,
        persona: PersonaSpec,
        topic: str,
        transcript: list[Utterance],
        condition_name: str,
        info_log: list[InterventionLogEntry] | None,
    ) -> list[dict]:
        # NOTE (レビュー C-1): judge の入力から条件識別情報を排除する (盲検化)。
        # condition_name は JudgeReport のメタデータとしてのみ保持し、
        # プロンプトには一切埋め込まない。None 条件でも「条件」という語や
        # 期待スコアを示唆する文言を出さず、提示なしと同じ中立表現に揃える。
        system = self._system_prompt.format(
            name=persona.name,
            stance_description=_STANCE_DESCRIPTION.get(persona.stance, persona.stance),
            focus=persona.focus,
            personality=persona.personality,
            topic=topic,
        )
        info_block = (
            _format_info_log_for_persona(info_log, persona.name)
            if info_log is not None
            else "(あなた向けの参考情報は提示されませんでした)"
        )
        user = (
            f"## 議論ログ\n{_format_transcript(transcript)}\n\n"
            f"## あなたが提示された参考情報\n{info_block}\n\n"
            f"## 出力\nJSON で各スコアと rationale:"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    async def evaluate_for(
        self,
        persona: PersonaSpec,
        topic: str,
        transcript: list[Utterance],
        condition_name: str,
        info_log: list[InterventionLogEntry] | None = None,
    ) -> JudgeReport:
        """1 ペルソナぶんの評価を返す。"""

        messages = self._build_messages(
            persona, topic, transcript, condition_name, info_log
        )
        scores = await self.llm.chat_structured(
            messages,  # type: ignore[arg-type]
            response_format=JudgeScores,
            model=self._model or self.llm.smart_model,
            temperature=self._temperature,
        )
        self.log.info(
            "judge.evaluated",
            persona=persona.name,
            condition=condition_name,
            satisfaction=scores.overall_satisfaction,
        )
        return JudgeReport(
            persona_name=persona.name,
            condition_name=condition_name,
            topic=topic,
            scores=scores,
        )

    async def evaluate_session(
        self,
        personas: list[PersonaSpec],
        topic: str,
        transcript: list[Utterance],
        condition_name: str,
        info_log: list[InterventionLogEntry] | None = None,
    ) -> list[JudgeReport]:
        """全ペルソナぶんの評価をまとめて返す。"""

        reports: list[JudgeReport] = []
        for persona in personas:
            report = await self.evaluate_for(
                persona, topic, transcript, condition_name, info_log
            )
            reports.append(report)
        return reports


# --- 集計 -----------------------------------------------------------


@dataclass(frozen=True)
class AggregatedScores:
    """複数のジャッジ結果を集計したもの。"""

    n: int
    overall_satisfaction_mean: float
    information_usefulness_mean: float
    opposition_understanding_mean: float
    confidence_change_mean: float
    intervention_transparency_mean: float
    overall_satisfaction_std: float = 0.0
    information_usefulness_std: float = 0.0
    opposition_understanding_std: float = 0.0
    confidence_change_std: float = 0.0
    intervention_transparency_std: float = 0.0


_SCORE_ATTRS: tuple[str, ...] = (
    "overall_satisfaction",
    "information_usefulness",
    "opposition_understanding",
    "confidence_change",
    "intervention_transparency",
)


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def _std(values: list[float]) -> float:
    return pstdev(values) if len(values) >= 2 else 0.0


def _aggregated_from_series(n: int, series: dict[str, list[float]]) -> AggregatedScores:
    return AggregatedScores(
        n=n,
        overall_satisfaction_mean=_mean(series["overall_satisfaction"]),
        information_usefulness_mean=_mean(series["information_usefulness"]),
        opposition_understanding_mean=_mean(series["opposition_understanding"]),
        confidence_change_mean=_mean(series["confidence_change"]),
        intervention_transparency_mean=_mean(series["intervention_transparency"]),
        overall_satisfaction_std=_std(series["overall_satisfaction"]),
        information_usefulness_std=_std(series["information_usefulness"]),
        opposition_understanding_std=_std(series["opposition_understanding"]),
        confidence_change_std=_std(series["confidence_change"]),
        intervention_transparency_std=_std(series["intervention_transparency"]),
    )


def aggregate_reports(reports: list[JudgeReport]) -> AggregatedScores:
    """複数ペルソナ x 複数ランの ``JudgeReport`` を平均と標準偏差にまとめる (flat pool)。

    NOTE (レビュー H-5): これはペルソナ×ランを独立サンプルとして pool する集計で、
    実効サンプルサイズを過大申告する。条件間比較には
    :func:`aggregate_reports_by_run` (ラン単位の2段集計) を使うこと。
    本関数は単一トランスクリプトのペルソナ間集計や参考値算出のために残す。
    """

    if not reports:
        return AggregatedScores(
            n=0,
            overall_satisfaction_mean=0.0,
            information_usefulness_mean=0.0,
            opposition_understanding_mean=0.0,
            confidence_change_mean=0.0,
            intervention_transparency_mean=0.0,
        )

    series = {attr: [float(getattr(r.scores, attr)) for r in reports] for attr in _SCORE_ATTRS}
    return _aggregated_from_series(len(reports), series)


def aggregate_reports_by_run(runs_reports: list[list[JudgeReport]]) -> AggregatedScores:
    """ラン単位の2段集計 (レビュー H-5 の疑似反復対策)。

    各ランの主観指標を「ラン内でペルソナ平均」してから、「ラン間で平均±SD」を取る。
    ``n`` はレポートを持つ **ラン数** (= クラスタ数)。同一ラン内の複数ペルソナは
    同じ transcript を評価しており独立でないため、pool せずまずラン内で平均する。

    :param runs_reports: ランごとの ``JudgeReport`` のリスト
        (``[[run1のペルソナ別report...], [run2の...], ...]``)。
    """

    # 空ランや report を持たないランは除外
    non_empty = [reps for reps in runs_reports if reps]
    if not non_empty:
        return AggregatedScores(
            n=0,
            overall_satisfaction_mean=0.0,
            information_usefulness_mean=0.0,
            opposition_understanding_mean=0.0,
            confidence_change_mean=0.0,
            intervention_transparency_mean=0.0,
        )

    # 第1段: ラン内でペルソナ平均 → ランごとに1値
    run_means: dict[str, list[float]] = {attr: [] for attr in _SCORE_ATTRS}
    for reps in non_empty:
        for attr in _SCORE_ATTRS:
            run_means[attr].append(_mean([float(getattr(r.scores, attr)) for r in reps]))

    # 第2段: ラン間で平均±SD
    return _aggregated_from_series(len(non_empty), run_means)


def aggregate_reports_by_persona(reports: list[JudgeReport]) -> dict[str, AggregatedScores]:
    """ペルソナ名ごとに (ラン横断で) 集計する。参考値として summary に残すための内訳。"""

    by_name: dict[str, list[JudgeReport]] = {}
    for rep in reports:
        by_name.setdefault(rep.persona_name, []).append(rep)
    return {name: aggregate_reports(reps) for name, reps in by_name.items()}


__all__ = [
    "AggregatedScores",
    "JudgeAgent",
    "JudgeReport",
    "JudgeScores",
    "aggregate_reports",
    "aggregate_reports_by_persona",
    "aggregate_reports_by_run",
]

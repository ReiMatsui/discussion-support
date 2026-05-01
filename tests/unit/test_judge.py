"""LLM-as-judge のユニットテスト。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from das.eval.conditions import InterventionLogEntry
from das.eval.judge import (
    JudgeAgent,
    JudgeReport,
    JudgeScores,
    aggregate_reports,
)
from das.eval.persona import build_persona
from das.llm import OpenAIClient
from das.types import Utterance


def _fake_llm() -> OpenAIClient:
    return OpenAIClient(client=MagicMock())


def _scores(**overrides: int) -> JudgeScores:
    base = {
        "overall_satisfaction": 5,
        "information_usefulness": 4,
        "opposition_understanding": 4,
        "confidence_change": 0,
        "intervention_transparency": 3,
        "rationale": "default",
    }
    base.update(overrides)  # type: ignore[arg-type]
    return JudgeScores(**base)  # type: ignore[arg-type]


# --- evaluate_for ----------------------------------------------------


async def test_judge_evaluate_for_returns_report() -> None:
    llm = _fake_llm()
    expected = _scores(overall_satisfaction=6, intervention_transparency=5)
    captured = AsyncMock(return_value=expected)
    llm.chat_structured = captured  # type: ignore[method-assign]

    judge = JudgeAgent(llm=llm)
    persona = build_persona(name="A", stance="pro", focus="環境")
    transcript = [Utterance(turn_id=1, speaker="A", text="主張")]
    report = await judge.evaluate_for(
        persona, "プラ容器", transcript, condition_name="full_proposal"
    )

    assert isinstance(report, JudgeReport)
    assert report.persona_name == "A"
    assert report.condition_name == "full_proposal"
    assert report.scores.overall_satisfaction == 6


async def test_judge_uses_smart_model_by_default() -> None:
    llm = _fake_llm()
    captured = AsyncMock(return_value=_scores())
    llm.chat_structured = captured  # type: ignore[method-assign]

    judge = JudgeAgent(llm=llm)
    persona = build_persona(name="A")
    await judge.evaluate_for(persona, "topic", [], "none")

    kwargs = captured.await_args.kwargs
    assert kwargs["model"] == llm.smart_model
    assert kwargs["temperature"] == 0.0


async def test_judge_includes_info_log_for_persona() -> None:
    llm = _fake_llm()
    captured = AsyncMock(return_value=_scores())
    llm.chat_structured = captured  # type: ignore[method-assign]

    judge = JudgeAgent(llm=llm)
    persona = build_persona(name="A")
    info_log = [
        InterventionLogEntry(
            turn_id=1,
            persona_name="A",
            timestamp="2026-05-01T00:00:00Z",
            items=[
                {"relation": "support", "source_text": "X 大学事例"},
                {"relation": "attack", "source_text": "コスト懸念"},
            ],
        ),
        InterventionLogEntry(
            turn_id=2,
            persona_name="B",  # 別ペルソナ向け、A の評価には含めない
            timestamp="2026-05-01T00:00:01Z",
            items=[],
        ),
    ]
    await judge.evaluate_for(persona, "topic", [], "full_proposal", info_log=info_log)

    user_msg = captured.await_args.args[0][1]["content"]
    assert "X 大学事例" in user_msg
    assert "[支持]" in user_msg
    assert "[反論]" in user_msg
    # B 向けのターン 2 は A の評価には出ない
    assert "ターン 2" not in user_msg


async def test_judge_evaluate_session_iterates_personas() -> None:
    llm = _fake_llm()
    llm.chat_structured = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _scores(overall_satisfaction=6),
            _scores(overall_satisfaction=4),
            _scores(overall_satisfaction=5),
        ]
    )
    judge = JudgeAgent(llm=llm)
    personas = [
        build_persona(name="A", stance="pro"),
        build_persona(name="B", stance="con"),
        build_persona(name="C", stance="neutral"),
    ]
    reports = await judge.evaluate_session(
        personas, "topic", [], condition_name="none"
    )
    assert len(reports) == 3
    assert [r.persona_name for r in reports] == ["A", "B", "C"]
    assert reports[0].scores.overall_satisfaction == 6


# --- aggregate_reports ----------------------------------------------


def test_aggregate_empty_reports() -> None:
    a = aggregate_reports([])
    assert a.n == 0
    assert a.overall_satisfaction_mean == 0.0


def test_aggregate_mean_and_std() -> None:
    reports = [
        JudgeReport(
            persona_name="A",
            condition_name="none",
            topic="t",
            scores=_scores(overall_satisfaction=4),
        ),
        JudgeReport(
            persona_name="B",
            condition_name="none",
            topic="t",
            scores=_scores(overall_satisfaction=6),
        ),
        JudgeReport(
            persona_name="C",
            condition_name="none",
            topic="t",
            scores=_scores(overall_satisfaction=5),
        ),
    ]
    a = aggregate_reports(reports)
    assert a.n == 3
    assert a.overall_satisfaction_mean == pytest.approx(5.0)
    assert a.overall_satisfaction_std > 0.0


def test_judge_scores_validation() -> None:
    """範囲外の値は pydantic で弾かれる。"""

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        JudgeScores(
            overall_satisfaction=10,  # 7 を超えている
            information_usefulness=4,
            opposition_understanding=4,
            confidence_change=0,
            intervention_transparency=3,
        )
    with pytest.raises(ValidationError):
        JudgeScores(
            overall_satisfaction=4,
            information_usefulness=4,
            opposition_understanding=4,
            confidence_change=10,  # +3 を超えている
            intervention_transparency=3,
        )

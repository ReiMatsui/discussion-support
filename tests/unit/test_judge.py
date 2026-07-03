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
    aggregate_reports_by_persona,
    aggregate_reports_by_run,
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


async def test_judge_prompt_is_blind_to_condition_name() -> None:
    """レビュー C-1: judge 入力に条件識別情報を一切埋め込まない (盲検化)。"""

    llm = _fake_llm()
    captured = AsyncMock(return_value=_scores())
    llm.chat_structured = captured  # type: ignore[method-assign]

    judge = JudgeAgent(llm=llm)
    persona = build_persona(name="A")
    transcript = [Utterance(turn_id=1, speaker="A", text="主張")]
    await judge.evaluate_for(
        persona, "topic", transcript, condition_name="full_proposal"
    )
    joined = "\n".join(m["content"] for m in captured.await_args.args[0])
    assert "full_proposal" not in joined
    assert "## 条件" not in joined

    # None 条件でも「条件」という語で条件を示唆しない
    await judge.evaluate_for(
        persona, "topic", transcript, condition_name="none", info_log=None
    )
    joined_none = "\n".join(m["content"] for m in captured.await_args.args[0])
    assert "情報提供なし条件" not in joined_none
    assert "none" not in joined_none


async def test_judge_info_log_unlabeled_item_is_neutral() -> None:
    """レビュー H-3: relation 未指定 (FlatRAG 等) の提示は中立の [参考] にする。"""

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
            items=[{"relation": "", "source_text": "関係ラベルなしの提示"}],
        )
    ]
    await judge.evaluate_for(persona, "topic", [], "flat_rag", info_log=info_log)
    user_msg = captured.await_args.args[0][1]["content"]
    assert "[参考] 関係ラベルなしの提示" in user_msg
    assert "[反論]" not in user_msg


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


def _rep(name: str, sat: int) -> JudgeReport:
    return JudgeReport(
        persona_name=name,
        condition_name="none",
        topic="t",
        scores=_scores(overall_satisfaction=sat),
    )


def test_aggregate_by_run_two_stage_matches_hand_calc() -> None:
    """2ラン×3ペルソナ固定データで、ラン単位2段集計が手計算と一致する (E2)。

    run1: 満足度 (3,6,6) → ラン平均 5.0
    run2: 満足度 (4,4,7) → ラン平均 5.0
    ラン間: 平均 5.0, pstdev([5.0, 5.0]) = 0.0
    n = ラン数 = 2 (pool した 6 ではない)。
    """

    run1 = [_rep("A", 3), _rep("B", 6), _rep("C", 6)]
    run2 = [_rep("A", 4), _rep("B", 4), _rep("C", 7)]
    a = aggregate_reports_by_run([run1, run2])
    assert a.n == 2
    assert a.overall_satisfaction_mean == pytest.approx(5.0)
    assert a.overall_satisfaction_std == pytest.approx(0.0)


def test_aggregate_by_run_std_is_across_runs() -> None:
    """ラン平均が異なるとき、SD はラン間で計算される。

    run1: (6,6,6) → 6.0, run2: (4,4,4) → 4.0
    ラン間平均 5.0, pstdev([6.0, 4.0]) = 1.0
    """

    run1 = [_rep("A", 6), _rep("B", 6), _rep("C", 6)]
    run2 = [_rep("A", 4), _rep("B", 4), _rep("C", 4)]
    a = aggregate_reports_by_run([run1, run2])
    assert a.n == 2
    assert a.overall_satisfaction_mean == pytest.approx(5.0)
    assert a.overall_satisfaction_std == pytest.approx(1.0)


def test_aggregate_by_run_skips_empty_runs() -> None:
    a = aggregate_reports_by_run([[], [_rep("A", 5)], []])
    assert a.n == 1
    assert a.overall_satisfaction_mean == pytest.approx(5.0)


def test_aggregate_by_persona_breakdown() -> None:
    reports = [_rep("A", 4), _rep("A", 6), _rep("B", 5)]
    by_persona = aggregate_reports_by_persona(reports)
    assert set(by_persona) == {"A", "B"}
    assert by_persona["A"].n == 2
    assert by_persona["A"].overall_satisfaction_mean == pytest.approx(5.0)
    assert by_persona["B"].n == 1


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

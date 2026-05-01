"""run_eval (多数回ラン executor) のユニットテスト。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from das.eval.conditions import ConditionNone
from das.eval.judge import JudgeAgent, JudgeScores
from das.eval.persona import build_persona
from das.eval.run_eval import (
    EvalResult,
    SingleRunResult,
    run_eval,
)
from das.llm import OpenAIClient


def _fake_llm(reply: str = "発言") -> OpenAIClient:
    client = OpenAIClient(client=MagicMock())
    client.chat = AsyncMock(return_value=reply)  # type: ignore[method-assign]
    return client


def _fake_judge_scores() -> JudgeScores:
    return JudgeScores(
        overall_satisfaction=5,
        information_usefulness=4,
        opposition_understanding=4,
        confidence_change=0,
        intervention_transparency=3,
        rationale="auto",
    )


# --- 基本動作 -------------------------------------------------------


async def test_run_eval_minimum_one_run() -> None:
    """ConditionNone 1 run x1 persona で発話ターン分の transcript が出来る。"""

    llm = _fake_llm("発言")
    personas = [build_persona(name="A", stance="pro")]
    result = await run_eval(
        topic="トピック",
        personas=personas,
        condition_factories={"none": ConditionNone},
        n_runs=1,
        max_turns=3,
        llm=llm,
    )
    assert isinstance(result, EvalResult)
    assert len(result.runs) == 1
    assert result.runs[0].condition_name == "none"
    assert len(result.runs[0].transcript) == 3


async def test_run_eval_multiple_runs_and_conditions() -> None:
    llm = _fake_llm("u")
    personas = [build_persona(name="A"), build_persona(name="B", stance="con")]
    factories = {"none": ConditionNone, "none_again": ConditionNone}
    result = await run_eval(
        topic="t",
        personas=personas,
        condition_factories=factories,
        n_runs=2,
        max_turns=2,
        llm=llm,
    )
    assert len(result.runs) == 4
    grouped = result.by_condition()
    assert set(grouped.keys()) == {"none", "none_again"}
    assert len(grouped["none"]) == 2


async def test_run_eval_invalid_args() -> None:
    llm = _fake_llm()
    with pytest.raises(ValueError):
        await run_eval(
            topic="t",
            personas=[build_persona(name="A")],
            condition_factories={},
            n_runs=1,
            llm=llm,
        )
    with pytest.raises(ValueError):
        await run_eval(
            topic="t",
            personas=[build_persona(name="A")],
            condition_factories={"none": ConditionNone},
            n_runs=0,
            llm=llm,
        )


# --- judge 統合 ------------------------------------------------------


async def test_run_eval_with_judge_aggregates() -> None:
    llm = _fake_llm("u")
    judge = JudgeAgent(llm=llm)
    judge.evaluate_session = AsyncMock(  # type: ignore[method-assign]
        side_effect=lambda personas, topic, transcript, condition_name, info_log=None: [
            __import__("das").eval.JudgeReport(
                persona_name=p.name,
                condition_name=condition_name,
                topic=topic,
                scores=_fake_judge_scores(),
            )
            for p in personas
        ]
    )

    personas = [build_persona(name="A"), build_persona(name="B")]
    result = await run_eval(
        topic="t",
        personas=personas,
        condition_factories={"none": ConditionNone},
        n_runs=2,
        max_turns=2,
        llm=llm,
        judge=judge,
    )
    # 2 ペルソナ x2 ラン = 4 件のレポート
    all_reports = [rep for r in result.runs for rep in r.judge_reports]
    assert len(all_reports) == 4

    aggregated = result.aggregate()
    assert "none" in aggregated
    assert aggregated["none"].n == 4
    assert aggregated["none"].overall_satisfaction_mean == 5.0


# --- ファイル出力 ----------------------------------------------------


async def test_run_eval_writes_to_eval_dir(tmp_path: Path) -> None:
    llm = _fake_llm("u")
    personas = [build_persona(name="A")]
    await run_eval(
        topic="t",
        personas=personas,
        condition_factories={"none": ConditionNone},
        n_runs=1,
        max_turns=2,
        llm=llm,
        eval_dir=tmp_path,
        eval_id="my-eval",
    )

    base = tmp_path / "my-eval"
    assert base.exists()
    assert (base / "meta.json").exists()
    assert (base / "summary.json").exists()
    run_dir = base / "none" / "run_001"
    assert run_dir.exists()
    transcript_lines = (run_dir / "transcript.jsonl").read_text().strip().split("\n")
    assert len(transcript_lines) == 2

    # meta.json は JSON として読める
    meta = json.loads((base / "meta.json").read_text())
    assert meta["topic"] == "t"
    assert meta["n_runs_per_condition"] == 1


# --- progress callback -----------------------------------------------


async def test_run_eval_progress_callback() -> None:
    llm = _fake_llm("u")
    personas = [build_persona(name="A")]
    seen: list[tuple[str, int, int]] = []

    def progress(cond: str, done: int, total: int) -> None:
        seen.append((cond, done, total))

    await run_eval(
        topic="t",
        personas=personas,
        condition_factories={"none": ConditionNone, "none2": ConditionNone},
        n_runs=2,
        max_turns=2,
        llm=llm,
        progress=progress,
    )
    assert len(seen) == 4
    assert seen[-1] == (seen[-1][0], 4, 4)


# --- 戻り値構造 -----------------------------------------------------


def test_single_run_result_fields() -> None:
    """SingleRunResult に必要なフィールドがあることを確認。"""

    sr = SingleRunResult(
        run_id="r1",
        condition_name="none",
        topic="t",
        transcript=[],
        transcript_metrics_=__import__(
            "das.eval.metrics", fromlist=["transcript_metrics"]
        ).transcript_metrics([]),
    )
    assert sr.condition_name == "none"
    assert sr.judge_reports == []
    assert sr.intervention_log is None

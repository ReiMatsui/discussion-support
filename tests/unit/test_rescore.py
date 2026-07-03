"""eval-rescore (実行と採点の分離) のユニットテスト。

受け入れ基準 (設計 E1):
  - 同一 run の rescore が決定的部分 (構造指標・citation) で同一結果を返す
  - judge を差し替えて rescore すると judge スコアだけが変わる
  - 旧形式 run を読んだら明確なエラー (黙って誤計算しない)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from das.eval.conditions import ConditionNone
from das.eval.judge import JudgeAgent, JudgeReport, JudgeScores
from das.eval.persona import build_persona
from das.eval.rescore import RescoreError, rescore_eval_dir, rescore_run
from das.eval.run_eval import run_eval
from das.llm import OpenAIClient


def _fake_llm(reply: str = "発言") -> OpenAIClient:
    client = OpenAIClient(client=MagicMock())
    client.chat = AsyncMock(return_value=reply)  # type: ignore[method-assign]
    return client


def _judge_with_scores(llm: OpenAIClient, sat: int) -> JudgeAgent:
    judge = JudgeAgent(llm=llm)
    judge.evaluate_session = AsyncMock(  # type: ignore[method-assign]
        side_effect=lambda personas, topic, transcript, condition_name, info_log=None: [
            JudgeReport(
                persona_name=p.name,
                condition_name=condition_name,
                topic=topic,
                scores=JudgeScores(
                    overall_satisfaction=sat,
                    information_usefulness=4,
                    opposition_understanding=4,
                    confidence_change=0,
                    intervention_transparency=3,
                    rationale="auto",
                ),
            )
            for p in personas
        ]
    )
    return judge


async def _make_eval_dir(tmp_path: Path, *, sat: int = 5) -> Path:
    llm = _fake_llm("u")
    judge = _judge_with_scores(llm, sat)
    await run_eval(
        topic="t",
        personas=[build_persona(name="A"), build_persona(name="B", stance="con")],
        condition_factories={"none": ConditionNone},
        n_runs=2,
        max_turns=2,
        llm=llm,
        judge=judge,
        eval_dir=tmp_path,
        eval_id="ev",
    )
    return tmp_path / "ev"


def _summary(eval_dir: Path) -> dict:
    return json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))


async def test_rescore_deterministic_structural_and_citation(tmp_path: Path) -> None:
    """構造指標・citation は rescore しても不変 (決定的計算)。"""
    eval_dir = await _make_eval_dir(tmp_path, sat=5)
    before = _summary(eval_dir)["by_condition"]["none"]

    llm = _fake_llm("u")
    await rescore_eval_dir(eval_dir, llm=llm, judge=_judge_with_scores(llm, 5))
    after = _summary(eval_dir)["by_condition"]["none"]

    assert after["structural"] == before["structural"]
    assert after["citation"] == before["citation"]


async def test_rescore_judge_change_only_moves_judge(tmp_path: Path) -> None:
    """judge を差し替えると judge スコアだけ変わり、構造/citation は不変。"""
    eval_dir = await _make_eval_dir(tmp_path, sat=5)
    before = _summary(eval_dir)["by_condition"]["none"]
    assert before["overall_satisfaction"][0] == pytest.approx(5.0)

    llm = _fake_llm("u")
    # judge を満足度=7 に差し替えて rescore
    await rescore_eval_dir(eval_dir, llm=llm, judge=_judge_with_scores(llm, 7))
    after = _summary(eval_dir)["by_condition"]["none"]

    assert after["overall_satisfaction"][0] == pytest.approx(7.0)
    assert after["structural"] == before["structural"]
    assert after["citation"] == before["citation"]


async def test_rescore_no_judge_keeps_structural(tmp_path: Path) -> None:
    """--no-judge 相当 (judge=None) でも構造指標は再計算される。"""
    eval_dir = await _make_eval_dir(tmp_path, sat=5)
    before = _summary(eval_dir)["by_condition"]["none"]

    llm = _fake_llm("u")
    result = await rescore_eval_dir(eval_dir, llm=llm, judge=None)
    after = _summary(eval_dir)["by_condition"]["none"]

    assert after["structural"] == before["structural"]
    # judge 無しなので主観スコアは 0 件集計 (n_runs_scored=0)
    assert after["n_runs_scored"] == 0
    assert len(result.runs) == 2


async def test_rescore_missing_meta_is_clear_error(tmp_path: Path) -> None:
    """meta.json が無い eval_dir は RescoreError。"""
    eval_dir = tmp_path / "broken"
    (eval_dir / "none" / "run_001").mkdir(parents=True)
    (eval_dir / "none" / "run_001" / "transcript.jsonl").write_text("", encoding="utf-8")

    llm = _fake_llm()
    with pytest.raises(RescoreError, match=r"meta\.json"):
        await rescore_eval_dir(eval_dir, llm=llm, judge=None)


async def test_rescore_meta_without_personas_is_clear_error(tmp_path: Path) -> None:
    """personas / topic を欠く旧形式 meta は RescoreError。"""
    eval_dir = tmp_path / "old"
    eval_dir.mkdir()
    (eval_dir / "meta.json").write_text(
        json.dumps({"eval_id": "old", "topic": "t"}), encoding="utf-8"
    )
    llm = _fake_llm()
    with pytest.raises(RescoreError, match="personas"):
        await rescore_eval_dir(eval_dir, llm=llm, judge=None)


async def test_rescore_run_without_transcript_is_clear_error(tmp_path: Path) -> None:
    """transcript.jsonl の無い run を直接 rescore すると RescoreError。"""
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    llm = _fake_llm()
    with pytest.raises(RescoreError, match="transcript"):
        await rescore_run(
            run_dir,
            condition_name="none",
            topic="t",
            personas=[build_persona(name="A")],
            consensus_kwargs={},
            llm=llm,
            judge=None,
        )

"""AQuA (Behrendt et al. DELITE 2024) 自前実装のユニットテスト。

LLM 部分は AsyncMock でフェイクする (実 API は呼ばない)。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from das.eval.aqua import (
    _S_MAX,
    _S_MIN,
    INDICATORS,
    AQuAAgent,
    AQuAScores,
    aggregate_aqua_reports,
)
from das.llm import OpenAIClient
from das.types import Utterance


def _fake_llm() -> OpenAIClient:
    return OpenAIClient(client=MagicMock())


def _scores(**overrides: int) -> AQuAScores:
    """全 indicator を 0 にして、上書きで一部を変える test helper。"""

    base = {ind.name: 0 for ind in INDICATORS}
    base.update(overrides)
    return AQuAScores(**base)  # type: ignore[arg-type]


# --- 集計式の正しさ -------------------------------------------------


def test_aqua_has_exactly_20_indicators() -> None:
    assert len(INDICATORS) == 20
    # 3 dimensions
    by_dim = {"rationality": 0, "reciprocity": 0, "civility": 0}
    for ind in INDICATORS:
        by_dim[ind.dimension] += 1
    assert by_dim == {"rationality": 7, "reciprocity": 5, "civility": 8}


def test_aqua_normalization_constants_match_paper() -> None:
    # Behrendt et al. §3.4: s_max ≈ 4.9893, s_min ≈ -1.6693
    assert pytest.approx(4.9893, abs=1e-3) == _S_MAX
    assert pytest.approx(-1.6693, abs=1e-3) == _S_MIN


def test_aqua_value_min_when_only_negatives_max() -> None:
    """負の重み indicator を 3、正の重み indicator を 0 にすると AQuA = 0."""

    s = _scores(**{ind.name: 3 for ind in INDICATORS if ind.weight < 0})
    assert s.aqua_value() == pytest.approx(0.0, abs=1e-6)


def test_aqua_value_max_when_only_positives_max() -> None:
    """正の重み indicator を 3、負の重み indicator を 0 にすると AQuA = 5."""

    s = _scores(**{ind.name: 3 for ind in INDICATORS if ind.weight > 0})
    assert s.aqua_value() == pytest.approx(5.0, abs=1e-6)


def test_aqua_value_all_zero_lies_in_range() -> None:
    s = _scores()
    v = s.aqua_value()
    # raw=0 -> normalized = 5 * (0 - s_min) / (s_max - s_min) ≈ 1.25
    assert 0.0 < v < 5.0
    assert v == pytest.approx(5.0 * (-_S_MIN) / (_S_MAX - _S_MIN), abs=1e-6)


def test_aqua_value_realistic_good_utterance() -> None:
    """relevance=3, justification=3, solution=2 程度の「良い」発話は 2.5 以上になる."""

    s = _scores(relevance=3, justification=3, solution_proposals=2, fact=2)
    assert s.aqua_value() > 2.5


# --- AQuAAgent (LLM 連携) -------------------------------------------


async def test_aqua_agent_score_utterance_returns_report() -> None:
    llm = _fake_llm()
    expected = _scores(relevance=3, justification=2)
    llm.chat_structured = AsyncMock(return_value=expected)  # type: ignore[method-assign]

    agent = AQuAAgent(llm=llm)
    utt = Utterance(turn_id=1, speaker="A", text="プラ容器は環境負荷が高い")
    report = await agent.score_utterance(utt, history=[utt])

    assert report.turn_id == 1
    assert report.speaker == "A"
    assert report.scores.relevance == 3
    assert report.scores.justification == 2
    assert report.aqua_value == pytest.approx(expected.aqua_value())


async def test_aqua_agent_score_discussion_aggregates() -> None:
    llm = _fake_llm()
    # 各発話に同じスコアを返すフェイク
    fixed = _scores(relevance=3, justification=2)
    llm.chat_structured = AsyncMock(return_value=fixed)  # type: ignore[method-assign]

    agent = AQuAAgent(llm=llm)
    transcript = [
        Utterance(turn_id=1, speaker="A", text="主張1"),
        Utterance(turn_id=2, speaker="B", text="反論1"),
        Utterance(turn_id=3, speaker="A", text="再主張"),
    ]
    report = await agent.score_discussion(transcript)

    assert report.n_utterances == 3
    # 全 utterance が同じスコアなら mean = それぞれの aqua_value
    assert report.mean == pytest.approx(fixed.aqua_value())
    assert report.std == pytest.approx(0.0)
    assert len(report.per_utterance) == 3
    # 並び順 (turn_id 昇順) が保たれる
    assert [r.turn_id for r in report.per_utterance] == [1, 2, 3]


async def test_aqua_agent_score_discussion_empty() -> None:
    llm = _fake_llm()
    llm.chat_structured = AsyncMock(return_value=_scores())  # type: ignore[method-assign]
    agent = AQuAAgent(llm=llm)
    report = await agent.score_discussion([])
    assert report.n_utterances == 0
    assert report.mean == 0.0


async def test_aqua_agent_per_indicator_mean() -> None:
    """per_indicator_mean が「発話別 0-3 スコアの平均」になっている."""

    llm = _fake_llm()
    # 各呼び出しで違うスコアを返す
    seq = [
        _scores(relevance=3, justification=0),
        _scores(relevance=2, justification=2),
        _scores(relevance=1, justification=3),
    ]
    llm.chat_structured = AsyncMock(side_effect=seq)  # type: ignore[method-assign]
    agent = AQuAAgent(llm=llm)
    transcript = [
        Utterance(turn_id=i, speaker="A", text=f"u{i}") for i in (1, 2, 3)
    ]
    report = await agent.score_discussion(transcript, concurrency=1)
    # AsyncMock の side_effect は呼び出し順に消費される
    # concurrency=1 で順番が保たれる
    assert report.per_indicator_mean["relevance"] == pytest.approx(2.0)
    assert report.per_indicator_mean["justification"] == pytest.approx(5 / 3)


# --- aggregate_aqua_reports ------------------------------------------


def test_aggregate_aqua_reports_empty() -> None:
    assert aggregate_aqua_reports([]) == {}


async def test_aggregate_aqua_reports_combines_runs() -> None:
    llm = _fake_llm()
    seq = [
        _scores(relevance=3, justification=3),  # good run
        _scores(relevance=1, justification=1),  # mid run
        _scores(relevance=0, justification=0),  # bad run
    ]
    llm.chat_structured = AsyncMock(side_effect=seq)  # type: ignore[method-assign]
    agent = AQuAAgent(llm=llm)

    # 3 ラン (各 1 発話) を別々に採点
    reports = []
    for i, _ in enumerate(seq, start=1):
        u = Utterance(turn_id=i, speaker="A", text=f"u{i}")
        reports.append(await agent.score_discussion([u]))

    agg = aggregate_aqua_reports(reports)
    assert agg["n_runs"] == 3
    # 各 run の mean を平均
    assert agg["mean"] == pytest.approx(
        sum(r.mean for r in reports) / 3
    )
    # std > 0 (run 間でばらついているはず)
    assert agg["std"] > 0
    assert "relevance" in agg["per_indicator_mean"]
    assert agg["per_indicator_mean"]["relevance"] == pytest.approx(
        (3 + 1 + 0) / 3
    )

"""ConsensusAgent (LLM-judge による合意検出) のユニットテスト。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from das.agents.consensus_agent import (
    ConsensusAgent,
    ConsensusJudgement,
    StanceJudgement,
)
from das.eval.consensus import detect_consensus_with_llm
from das.eval.persona import build_persona
from das.graph.schema import Edge, Node
from das.graph.store import NetworkXGraphStore
from das.llm import OpenAIClient
from das.types import Utterance


def _utts(speakers: list[str], texts: list[str]) -> list[Utterance]:
    return [
        Utterance(turn_id=i + 1, speaker=s, text=t)
        for i, (s, t) in enumerate(zip(speakers, texts, strict=True))
    ]


def _fake_llm() -> OpenAIClient:
    return OpenAIClient(client=MagicMock())


def _stub_judgement(*, reached: bool, confidence: float = 0.85) -> ConsensusJudgement:
    return ConsensusJudgement(
        consensus_reached=reached,
        consensus_position="段階的廃止 + 学生支援" if reached else "",
        n_agreeing=3 if reached else 1,
        n_total=3,
        confidence=confidence,
        rationale="全員が条件付きで段階導入に賛同",
        stances=[
            StanceJudgement(speaker="A", position="廃止", polarity="pro", confidence=0.9),
            StanceJudgement(speaker="B", position="段階導入なら", polarity="partial_pro", confidence=0.8),
            StanceJudgement(speaker="C", position="折衷案", polarity="partial_pro", confidence=0.85),
        ],
    )


# --- 基本動作 -----------------------------------------------------------


async def test_consensus_agent_returns_structured_judgement() -> None:
    llm = _fake_llm()
    expected = _stub_judgement(reached=True)
    llm.chat_structured = AsyncMock(return_value=expected)  # type: ignore[method-assign]

    agent = ConsensusAgent(llm=llm)
    transcript = _utts(["A", "B", "C"], ["x", "y", "z"])
    personas = [
        build_persona(name="A", stance="pro"),
        build_persona(name="B", stance="con"),
        build_persona(name="C", stance="neutral"),
    ]
    result = await agent.judge(topic="T", transcript=transcript, personas=personas)
    assert result.consensus_reached is True
    assert result.n_agreeing == 3
    llm.chat_structured.assert_awaited_once()


# --- 二段ハイブリッド (detect_consensus_with_llm) ---------------------


async def test_with_llm_skips_call_when_no_signals() -> None:
    """構造シグナルが立っていない序盤では LLM を呼ばずに早期 return。"""

    llm = _fake_llm()
    judge_mock = AsyncMock()
    agent = ConsensusAgent(llm=llm)
    agent.judge = judge_mock  # type: ignore[method-assign]

    transcript = _utts(["A", "B"], ["主張1", "反論1"])  # 短すぎる
    report = await detect_consensus_with_llm(
        transcript,
        topic="T",
        personas=[build_persona(name="A")],
        agent=agent,
    )
    assert report.consensus_reached is False
    judge_mock.assert_not_awaited()  # LLM は呼ばれない


async def test_with_llm_invokes_when_structural_signal_present() -> None:
    """構造的静止が立つと LLM-judge を呼んで最終判定する。"""

    # 古いノードのみある (新 claim/attack ゼロ → no_new_attacks シグナル)
    store = NetworkXGraphStore()
    n1 = Node(text="古い主張", node_type="claim", source="utterance", author="A")
    n2 = Node(text="古い反論", node_type="claim", source="utterance", author="B")
    store.add_node(n1)
    store.add_node(n2)
    store.add_edge(Edge(src_id=n2.id, dst_id=n1.id, relation="attack", confidence=0.8))

    llm = _fake_llm()
    agent = ConsensusAgent(llm=llm)
    expected = _stub_judgement(reached=True, confidence=0.85)
    agent.judge = AsyncMock(return_value=expected)  # type: ignore[method-assign]

    transcript = _utts(
        ["A", "B", "C", "A", "B", "C"],
        ["主張1", "反論1", "中立", "整理", "賛同", "結論"],
    )
    personas = [
        build_persona(name="A"),
        build_persona(name="B"),
        build_persona(name="C"),
    ]
    report = await detect_consensus_with_llm(
        transcript,
        topic="T",
        personas=personas,
        agent=agent,
        store=store,
    )
    assert report.consensus_reached is True
    assert report.confidence == pytest.approx(0.85)
    assert report.llm_judgement is not None
    assert report.llm_judgement["n_agreeing"] == 3
    agent.judge.assert_awaited_once()


async def test_with_llm_rejects_low_confidence_judgement() -> None:
    """LLM が合意と言っても confidence < min_judge_confidence なら却下。"""

    store = NetworkXGraphStore()
    n1 = Node(text="x", node_type="claim", source="utterance", author="A")
    n2 = Node(text="y", node_type="claim", source="utterance", author="B")
    store.add_node(n1)
    store.add_node(n2)
    store.add_edge(Edge(src_id=n2.id, dst_id=n1.id, relation="attack", confidence=0.7))

    agent = ConsensusAgent(llm=_fake_llm())
    expected = _stub_judgement(reached=True, confidence=0.4)  # 低信頼度
    agent.judge = AsyncMock(return_value=expected)  # type: ignore[method-assign]

    transcript = _utts(["A"] * 6, ["x"] * 6)
    report = await detect_consensus_with_llm(
        transcript,
        topic="T",
        personas=[build_persona(name="A")],
        agent=agent,
        store=store,
        min_judge_confidence=0.7,
    )
    assert report.consensus_reached is False  # confidence 不足で却下
    # ただし llm_judgement は記録される (透明性)
    assert report.llm_judgement is not None

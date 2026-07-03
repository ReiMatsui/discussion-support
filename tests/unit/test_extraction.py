"""ExtractionAgent のユニットテスト。

OpenAI 呼び出しは ``OpenAIClient.chat_structured`` を AsyncMock で差し替える。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from das.agents.extraction import (
    ExtractionAgent,
    _ExtractedUnit,
    _ExtractionResult,
    _IntraEdge,
)
from das.llm import OpenAIClient
from das.types import Utterance


def _fake_llm() -> OpenAIClient:
    """``AsyncOpenAI`` 部分は使わないので MagicMock で代用したラッパを返す。"""

    return OpenAIClient(client=MagicMock())


@pytest.fixture
def utterance() -> Utterance:
    return Utterance(
        turn_id=5,
        speaker="A",
        text="プラ容器は年間 2 トンのゴミを出している。だから廃止すべき。",
    )


async def test_extract_decomposes_utterance(utterance: Utterance) -> None:
    llm = _fake_llm()
    llm.chat_structured = AsyncMock(  # type: ignore[method-assign]
        return_value=_ExtractionResult(
            units=[
                _ExtractedUnit(
                    text="プラ容器は年間 2 トンのゴミを出している",
                    node_type="premise",
                ),
                _ExtractedUnit(text="プラ容器を廃止すべき", node_type="claim"),
            ]
        )
    )

    agent = ExtractionAgent(llm=llm)
    result = await agent.extract(utterance)
    nodes = result.nodes

    assert len(nodes) == 2
    premise, claim = nodes
    assert premise.text == "プラ容器は年間 2 トンのゴミを出している"
    assert premise.node_type == "premise"
    assert premise.source == "utterance"
    assert premise.author == "A"
    assert premise.metadata["turn_id"] == 5
    # turn_index はアクティブ窓判定用に発話連番 (= turn_id) を持つ (logic_review A5)
    assert premise.turn_index == 5
    assert claim.turn_index == 5
    assert premise.timestamp == utterance.timestamp
    assert claim.node_type == "claim"


async def test_extract_skips_empty_or_whitespace(utterance: Utterance) -> None:
    llm = _fake_llm()
    llm.chat_structured = AsyncMock(  # type: ignore[method-assign]
        return_value=_ExtractionResult(
            units=[
                _ExtractedUnit(text="", node_type="claim"),
                _ExtractedUnit(text="   ", node_type="premise"),
                _ExtractedUnit(text="本物の主張", node_type="claim"),
            ]
        )
    )
    agent = ExtractionAgent(llm=llm)
    result = await agent.extract(utterance)

    assert [n.text for n in result.nodes] == ["本物の主張"]


async def test_extract_returns_empty_when_no_units(utterance: Utterance) -> None:
    llm = _fake_llm()
    llm.chat_structured = AsyncMock(  # type: ignore[method-assign]
        return_value=_ExtractionResult(units=[])
    )
    agent = ExtractionAgent(llm=llm)
    result = await agent.extract(utterance)
    assert result.nodes == []
    assert result.edges == []


async def test_extract_passes_speaker_and_turn_in_user_prompt(utterance: Utterance) -> None:
    """LLM に渡される user メッセージに speaker と turn_id が含まれることを確認。"""

    llm = _fake_llm()
    captured = AsyncMock(return_value=_ExtractionResult(units=[]))
    llm.chat_structured = captured  # type: ignore[method-assign]

    agent = ExtractionAgent(llm=llm)
    await agent.extract(utterance)

    captured.assert_awaited_once()
    messages = captured.await_args.args[0]
    user_msg = messages[1]
    assert user_msg["role"] == "user"
    assert "話者: A" in user_msg["content"]
    assert "発話番号: 5" in user_msg["content"]
    assert "プラ容器" in user_msg["content"]


async def test_extract_includes_context_for_reference_resolution(
    utterance: Utterance,
) -> None:
    """context を渡すと user メッセージに参照文脈 (話者名付き) が入る (G2)。"""

    llm = _fake_llm()
    captured = AsyncMock(return_value=_ExtractionResult(units=[]))
    llm.chat_structured = captured  # type: ignore[method-assign]

    agent = ExtractionAgent(llm=llm)
    context = [Utterance(turn_id=4, speaker="B", text="紙容器はコストが3倍だ")]
    await agent.extract(utterance, context=context)

    content = captured.await_args.args[0][1]["content"]
    assert "参照文脈" in content
    assert "B: 紙容器はコストが3倍だ" in content
    # 判定対象は元発話
    assert "発話番号: 5" in content


async def test_extract_builds_intra_edges(utterance: Utterance) -> None:
    """intra_edges が created_by=extraction のエッジになる (G2, H-1)。"""

    llm = _fake_llm()
    llm.chat_structured = AsyncMock(  # type: ignore[method-assign]
        return_value=_ExtractionResult(
            units=[
                _ExtractedUnit(text="根拠", node_type="premise"),
                _ExtractedUnit(text="主張", node_type="claim"),
            ],
            intra_edges=[_IntraEdge(src=0, dst=1, relation="support")],
        )
    )
    agent = ExtractionAgent(llm=llm)
    result = await agent.extract(utterance)

    assert len(result.nodes) == 2
    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge.src_id == result.nodes[0].id  # premise
    assert edge.dst_id == result.nodes[1].id  # claim
    assert edge.relation == "support"
    assert edge.created_by == "extraction"
    assert edge.confidence == 1.0


async def test_extract_intra_edges_default_empty(utterance: Utterance) -> None:
    """intra_edges を返さない (旧スキーマ互換) 場合はエッジ 0 件。"""

    llm = _fake_llm()
    llm.chat_structured = AsyncMock(  # type: ignore[method-assign]
        return_value=_ExtractionResult(units=[_ExtractedUnit(text="主張", node_type="claim")])
    )
    agent = ExtractionAgent(llm=llm)
    result = await agent.extract(utterance)
    assert len(result.nodes) == 1
    assert result.edges == []

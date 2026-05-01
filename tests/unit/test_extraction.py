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
    nodes = await agent.extract(utterance)

    assert len(nodes) == 2
    premise, claim = nodes
    assert premise.text == "プラ容器は年間 2 トンのゴミを出している"
    assert premise.node_type == "premise"
    assert premise.source == "utterance"
    assert premise.author == "A"
    assert premise.metadata["turn_id"] == 5
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
    nodes = await agent.extract(utterance)

    assert [n.text for n in nodes] == ["本物の主張"]


async def test_extract_returns_empty_when_no_units(utterance: Utterance) -> None:
    llm = _fake_llm()
    llm.chat_structured = AsyncMock(  # type: ignore[method-assign]
        return_value=_ExtractionResult(units=[])
    )
    agent = ExtractionAgent(llm=llm)
    assert await agent.extract(utterance) == []


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

"""Orchestrator のユニットテスト。

OpenAI 呼び出しは AsyncMock でフェイク化する。本物の API は呼ばない。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from das.agents.extraction import _ExtractedUnit, _ExtractionResult
from das.agents.linking import _LinkJudgment
from das.graph.schema import Node
from das.graph.store import NetworkXGraphStore
from das.llm import OpenAIClient
from das.runtime import Orchestrator
from das.types import Utterance


def _fake_llm() -> OpenAIClient:
    return OpenAIClient(client=MagicMock())


@pytest.fixture
def transcript() -> list[Utterance]:
    return [
        Utterance(turn_id=1, speaker="A", text="プラ容器を廃止すべき"),
        Utterance(
            turn_id=2,
            speaker="B",
            text="紙容器はコスト 3 倍で値上げにつながる",
        ),
    ]


def _extraction_for(turn_id: int) -> _ExtractionResult:
    """ターンに応じた抽出結果。"""

    if turn_id == 1:
        return _ExtractionResult(
            units=[_ExtractedUnit(text="プラ容器を廃止すべき", node_type="claim")]
        )
    if turn_id == 2:
        return _ExtractionResult(
            units=[
                _ExtractedUnit(text="紙容器はコストが 3 倍で値上げにつながる", node_type="claim")
            ]
        )
    return _ExtractionResult(units=[])


def _make_orchestrator() -> tuple[Orchestrator, OpenAIClient]:
    llm = _fake_llm()
    store = NetworkXGraphStore()
    orch = Orchestrator.assemble(llm=llm, store=store, threshold=0.6, top_k=3)
    return orch, llm


async def test_run_session_creates_nodes_for_each_utterance(
    transcript: list[Utterance],
) -> None:
    orch, llm = _make_orchestrator()

    extraction_results = {1: _extraction_for(1), 2: _extraction_for(2)}
    judgments = [_LinkJudgment(relation="none", confidence=0.9, rationale="-")]

    async def fake_chat_structured(messages: list[dict], response_format, **kwargs):
        # extraction か linking かを response_format で判定
        if response_format is _ExtractionResult:
            user = messages[1]["content"]
            if "発話番号: 1" in user:
                return extraction_results[1]
            if "発話番号: 2" in user:
                return extraction_results[2]
            return _ExtractionResult(units=[])
        if response_format is _LinkJudgment:
            return judgments[0]
        raise AssertionError("unexpected schema")

    llm.chat_structured = AsyncMock(side_effect=fake_chat_structured)  # type: ignore[method-assign]
    llm.embed_one = AsyncMock(return_value=[1.0, 0.0])  # type: ignore[method-assign]
    llm.embed = AsyncMock(return_value=[[0.5, 0.5]])  # type: ignore[method-assign]

    store = await orch.run_session(transcript)

    # 各 utterance から 1 ノード = 計 2 ノードがストアに入る
    nodes = list(store.nodes())
    assert len(nodes) == 2
    speakers = {n.author for n in nodes}
    assert speakers == {"A", "B"}


async def test_run_session_invokes_linking_per_added_node(
    transcript: list[Utterance],
) -> None:
    """発話ノードごとに LinkingAgent.link_node が呼ばれることを確認。"""

    orch, llm = _make_orchestrator()
    link_calls: list[Node] = []

    async def fake_link(target: Node, store) -> list:
        link_calls.append(target)
        return []

    orch.linking.link_node = AsyncMock(side_effect=fake_link)  # type: ignore[method-assign]

    async def fake_extract(messages: list[dict], response_format, **kwargs):
        user = messages[1]["content"]
        if "発話番号: 1" in user:
            return _extraction_for(1)
        if "発話番号: 2" in user:
            return _extraction_for(2)
        return _ExtractionResult(units=[])

    llm.chat_structured = AsyncMock(side_effect=fake_extract)  # type: ignore[method-assign]

    await orch.run_session(transcript)

    assert len(link_calls) == 2
    assert {n.text for n in link_calls} == {
        "プラ容器を廃止すべき",
        "紙容器はコストが 3 倍で値上げにつながる",
    }


async def test_run_session_with_extraction_returning_no_units_does_not_link(
    transcript: list[Utterance],
) -> None:
    orch, llm = _make_orchestrator()
    orch.linking.link_node = AsyncMock(return_value=[])  # type: ignore[method-assign]
    llm.chat_structured = AsyncMock(  # type: ignore[method-assign]
        return_value=_ExtractionResult(units=[])
    )

    store = await orch.run_session(transcript)
    assert list(store.nodes()) == []
    orch.linking.link_node.assert_not_awaited()


async def test_ingest_documents_does_not_trigger_linking(tmp_path) -> None:
    """ドキュメント取り込みでは LinkingAgent が起動しないことを確認 (バス未経由)。"""

    orch, llm = _make_orchestrator()
    orch.linking.link_node = AsyncMock(return_value=[])  # type: ignore[method-assign]

    (tmp_path / "doc.md").write_text("本文", encoding="utf-8")

    from das.agents.document import _DocumentExtraction, _DocumentUnit

    llm.chat_structured = AsyncMock(  # type: ignore[method-assign]
        return_value=_DocumentExtraction(
            units=[_DocumentUnit(text="文書クレーム", node_type="claim")]
        )
    )

    nodes = await orch.ingest_documents(tmp_path)
    assert len(nodes) == 1
    orch.linking.link_node.assert_not_awaited()

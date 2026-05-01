"""3 条件 (None / FlatRAG / FullProposal) のユニットテスト。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from das.eval.conditions import (
    ConditionFlatRAG,
    ConditionFullProposal,
    ConditionNone,
)
from das.eval.persona import build_persona
from das.llm import OpenAIClient
from das.types import Utterance


def _fake_llm() -> OpenAIClient:
    return OpenAIClient(client=MagicMock())


# --- ConditionNone ------------------------------------------------------


async def test_condition_none_always_returns_none() -> None:
    cond = ConditionNone()
    await cond.setup()
    persona = build_persona(name="X")
    history = [Utterance(turn_id=1, speaker="X", text="t")]
    assert await cond.info_provider(history, persona) is None
    assert await cond.info_provider([], persona) is None


# --- ConditionFlatRAG ---------------------------------------------------


async def test_flat_rag_returns_none_without_setup() -> None:
    llm = _fake_llm()
    cond = ConditionFlatRAG(llm=llm, top_k=3)
    persona = build_persona(name="X")
    history = [Utterance(turn_id=1, speaker="X", text="プラ容器")]
    assert await cond.info_provider(history, persona) is None


async def test_flat_rag_returns_none_for_empty_history(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text("段落 A\n\n段落 B", encoding="utf-8")
    llm = _fake_llm()
    llm.embed = AsyncMock(return_value=[[1.0, 0.0], [0.0, 1.0]])  # type: ignore[method-assign]
    cond = ConditionFlatRAG(llm=llm)
    await cond.setup(docs_dir=tmp_path)

    persona = build_persona(name="X")
    assert await cond.info_provider([], persona) is None


async def test_flat_rag_picks_top_k(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text(
        "コストの話\n\n環境負荷の話\n\n別の論点", encoding="utf-8"
    )
    llm = _fake_llm()
    # 3 段落の embedding を順に [1,0], [0,1], [0,0] に固定
    llm.embed = AsyncMock(return_value=[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # type: ignore[method-assign]
    cond = ConditionFlatRAG(llm=llm, top_k=2)
    await cond.setup(docs_dir=tmp_path)

    # クエリ vector は [1,0] に設定 → 「コストの話」が最も近い
    llm.embed_one = AsyncMock(return_value=[1.0, 0.0])  # type: ignore[method-assign]

    persona = build_persona(name="X")
    history = [Utterance(turn_id=1, speaker="X", text="コスト懸念")]
    info = await cond.info_provider(history, persona)
    assert info is not None
    assert "コストの話" in info
    # top_k=2 で 2 件返る
    assert info.count("[doc]") == 2


async def test_flat_rag_setup_with_missing_dir() -> None:
    cond = ConditionFlatRAG(llm=_fake_llm())
    await cond.setup(docs_dir=None)
    await cond.setup(docs_dir=Path("/nonexistent/path"))
    # エラーなく終わる、後続の info_provider は None を返す
    assert (
        await cond.info_provider(
            [Utterance(turn_id=1, speaker="X", text="x")],
            build_persona(name="X"),
        )
        is None
    )


# --- ConditionFullProposal -------------------------------------------


async def test_full_proposal_returns_none_when_no_history() -> None:
    llm = _fake_llm()
    cond = ConditionFullProposal(llm=llm)
    await cond.setup()
    persona = build_persona(name="X")
    assert await cond.info_provider([], persona) is None


async def test_full_proposal_returns_none_before_setup() -> None:
    llm = _fake_llm()
    cond = ConditionFullProposal(llm=llm)
    persona = build_persona(name="X")
    history = [Utterance(turn_id=1, speaker="X", text="t")]
    assert await cond.info_provider(history, persona) is None


async def test_full_proposal_returns_support_attack_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Orchestrator をフェイク化して、最新発話に張られた edge を返すことを確認。"""

    from uuid import uuid4

    from das.graph.schema import Edge, Node
    from das.graph.store import NetworkXGraphStore

    llm = _fake_llm()
    cond = ConditionFullProposal(llm=llm)

    # setup の中で Orchestrator.assemble が呼ばれる。これを差し替える。
    fake_store = NetworkXGraphStore()
    fake_orch = MagicMock()
    fake_orch.bus = MagicMock()
    fake_orch.bus.publish = AsyncMock()
    fake_orch.bus.drain = AsyncMock()
    fake_orch.ingest_documents = AsyncMock(return_value=[])
    fake_orch.store = fake_store

    def fake_assemble(*args: object, **kwargs: object) -> object:
        return fake_orch

    monkeypatch.setattr("das.eval.conditions.Orchestrator.assemble", fake_assemble)

    await cond.setup()

    # turn_id=1 に対応するノードを 1 件、それを支持する文書ノードと攻撃する発話ノードを用意
    target = Node(
        id=uuid4(),
        text="プラ容器を廃止すべき",
        node_type="claim",
        source="utterance",
        author="A",
        metadata={"turn_id": 1},
    )
    doc_supporter = Node(
        text="統計データ", node_type="premise", source="document", author="d1"
    )
    attacker = Node(
        text="コスト懸念",
        node_type="claim",
        source="utterance",
        author="B",
        metadata={"turn_id": 0},
    )
    fake_store.add_node(target)
    fake_store.add_node(doc_supporter)
    fake_store.add_node(attacker)
    fake_store.add_edge(
        Edge(
            src_id=doc_supporter.id,
            dst_id=target.id,
            relation="support",
            confidence=0.9,
        )
    )
    fake_store.add_edge(
        Edge(
            src_id=attacker.id,
            dst_id=target.id,
            relation="attack",
            confidence=0.8,
        )
    )

    persona = build_persona(name="A")
    history = [Utterance(turn_id=1, speaker="A", text="プラ容器を廃止すべき")]
    info = await cond.info_provider(history, persona)

    assert info is not None
    assert "[支持] 統計データ" in info
    assert "[反論] コスト懸念" in info
    fake_orch.bus.publish.assert_awaited_once()


async def test_full_proposal_does_not_reprocess_same_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _fake_llm()
    cond = ConditionFullProposal(llm=llm)

    fake_orch = MagicMock()
    fake_orch.bus = MagicMock()
    fake_orch.bus.publish = AsyncMock()
    fake_orch.bus.drain = AsyncMock()
    fake_orch.ingest_documents = AsyncMock(return_value=[])
    from das.graph.store import NetworkXGraphStore

    fake_orch.store = NetworkXGraphStore()
    monkeypatch.setattr(
        "das.eval.conditions.Orchestrator.assemble", lambda *a, **k: fake_orch
    )
    await cond.setup()

    persona = build_persona(name="A")
    history1 = [Utterance(turn_id=1, speaker="A", text="t1")]
    history2 = [
        Utterance(turn_id=1, speaker="A", text="t1"),
        Utterance(turn_id=2, speaker="B", text="t2"),
    ]

    await cond.info_provider(history1, persona)
    await cond.info_provider(history2, persona)

    # publish は turn_id=1, 2 で計 2 回 (1 は重複しない)
    assert fake_orch.bus.publish.await_count == 2

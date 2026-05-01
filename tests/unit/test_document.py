"""DocumentAgent のユニットテスト。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from das.agents.document import (
    DocumentAgent,
    _DocumentExtraction,
    _DocumentUnit,
)
from das.graph.schema import Node
from das.graph.store import NetworkXGraphStore
from das.llm import OpenAIClient


def _fake_llm() -> OpenAIClient:
    return OpenAIClient(client=MagicMock())


def _result(*units: tuple[str, str]) -> _DocumentExtraction:
    return _DocumentExtraction(
        units=[_DocumentUnit(text=t, node_type=nt) for t, nt in units]  # type: ignore[arg-type]
    )


@pytest.fixture
def store() -> NetworkXGraphStore:
    return NetworkXGraphStore()


# --- ingest_text --------------------------------------------------------


async def test_ingest_text_creates_document_nodes(store: NetworkXGraphStore) -> None:
    llm = _fake_llm()
    llm.chat_structured = AsyncMock(  # type: ignore[method-assign]
        return_value=_result(
            ("X 大学では紙容器導入初年度に容器コストが約 3 倍になった", "premise"),
            ("紙容器への移行は長期的に持続可能である", "claim"),
        )
    )
    agent = DocumentAgent(llm=llm)

    nodes = await agent.ingest_text(
        "X 大学のカフェテリアは...",
        doc_id="x_univ_case",
        store=store,
        source_path="data/docs/x_univ_case.md",
    )

    assert len(nodes) == 2
    premise = nodes[0]
    assert premise.source == "document"
    assert premise.author == "x_univ_case"
    assert premise.metadata["doc_id"] == "x_univ_case"
    assert premise.metadata["source_path"] == "data/docs/x_univ_case.md"
    assert {n.node_type for n in nodes} == {"premise", "claim"}
    assert {n.id for n in store.nodes()} == {n.id for n in nodes}


async def test_ingest_text_skips_blank_units(store: NetworkXGraphStore) -> None:
    llm = _fake_llm()
    llm.chat_structured = AsyncMock(  # type: ignore[method-assign]
        return_value=_result(
            ("", "claim"),
            ("   ", "premise"),
            ("有効な前提", "premise"),
        )
    )
    agent = DocumentAgent(llm=llm)
    nodes = await agent.ingest_text("...", doc_id="d1", store=store)
    assert [n.text for n in nodes] == ["有効な前提"]


# --- ingest_directory ----------------------------------------------------


async def test_ingest_directory_reads_md_and_txt(tmp_path: Path, store: NetworkXGraphStore) -> None:
    (tmp_path / "alpha.md").write_text("alpha 本文", encoding="utf-8")
    (tmp_path / "beta.txt").write_text("beta 本文", encoding="utf-8")
    (tmp_path / "ignore.json").write_text("{}", encoding="utf-8")

    llm = _fake_llm()
    # 各 ingest_text 呼び出しで 1 件ずつ返す副作用
    llm.chat_structured = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _result(("a-claim", "claim")),
            _result(("b-premise", "premise")),
        ]
    )
    agent = DocumentAgent(llm=llm)

    nodes = await agent.ingest_directory(tmp_path, store=store)

    assert {n.text for n in nodes} == {"a-claim", "b-premise"}
    assert {n.author for n in nodes} == {"alpha", "beta"}
    # json は読まれない
    assert llm.chat_structured.await_count == 2


async def test_ingest_directory_passes_source_path(
    tmp_path: Path, store: NetworkXGraphStore
) -> None:
    file = tmp_path / "doc.md"
    file.write_text("body", encoding="utf-8")
    llm = _fake_llm()
    llm.chat_structured = AsyncMock(  # type: ignore[method-assign]
        return_value=_result(("c", "claim"))
    )
    agent = DocumentAgent(llm=llm)

    nodes = await agent.ingest_directory(tmp_path, store=store)

    assert nodes[0].metadata["source_path"] == str(file)


# --- retrieve -----------------------------------------------------------


def test_retrieve_returns_only_document_nodes(store: NetworkXGraphStore) -> None:
    doc_node = Node(text="doc", node_type="claim", source="document", author="d1")
    utt_node = Node(text="utt", node_type="claim", source="utterance", author="A")
    web_node = Node(text="web", node_type="premise", source="web", author="example.com")
    for n in (doc_node, utt_node, web_node):
        store.add_node(n)

    agent = DocumentAgent(llm=OpenAIClient(client=MagicMock()))
    target = Node(text="target", node_type="claim", source="utterance", author="A")
    result = agent.retrieve(target, store)
    assert {n.id for n in result} == {doc_node.id}


def test_retrieve_respects_limit(store: NetworkXGraphStore) -> None:
    for i in range(5):
        store.add_node(Node(text=f"d{i}", node_type="premise", source="document", author=f"d{i}"))
    agent = DocumentAgent(llm=OpenAIClient(client=MagicMock()))
    target = Node(text="target", node_type="claim", source="utterance", author="A")
    result = agent.retrieve(target, store, limit=2)
    assert len(result) == 2


def test_retrieve_real_sample_docs_directory_exists() -> None:
    """``data/docs/`` 配下に M1 用のサンプル文書が置かれていることを確認 (整合性チェック)。"""

    repo_root = Path(__file__).resolve().parents[2]
    docs_dir = repo_root / "data" / "docs"
    md_files = list(docs_dir.glob("*.md"))
    assert len(md_files) >= 3, f"expected sample docs in {docs_dir}, found: {md_files}"

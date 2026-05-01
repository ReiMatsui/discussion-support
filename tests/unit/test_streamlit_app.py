"""Streamlit ビューアの軽量テスト。

Streamlit のフル UI テストは行わず、ヘルパ関数のみ単体で検証する。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from das.graph.schema import Edge, Node
from das.graph.store import NetworkXGraphStore

streamlit_app = pytest.importorskip("das.ui.streamlit_app")


@pytest.fixture
def populated() -> tuple[list[Node], list[Edge]]:
    n1 = Node(text="A の主張", node_type="claim", source="utterance", author="A")
    n2 = Node(text="B の反論", node_type="claim", source="utterance", author="B")
    n3 = Node(
        text="文献からの根拠",
        node_type="premise",
        source="document",
        author="d1",
    )
    edges = [
        Edge(src_id=n2.id, dst_id=n1.id, relation="attack", confidence=0.9),
        Edge(src_id=n3.id, dst_id=n2.id, relation="attack", confidence=0.8),
    ]
    return [n1, n2, n3], edges


def test_stats_counts(populated: tuple[list[Node], list[Edge]]) -> None:
    nodes, edges = populated
    stats = streamlit_app._stats(nodes, edges)
    assert stats["nodes"] == 3
    assert stats["edges"] == 2
    assert stats["utterance"] == 2
    assert stats["document"] == 1
    assert stats["web"] == 0
    assert stats["support"] == 0
    assert stats["attack"] == 2


def test_node_dataframe_columns(populated: tuple[list[Node], list[Edge]]) -> None:
    nodes, _ = populated
    df = streamlit_app._node_dataframe(nodes)
    assert list(df.columns) == ["id", "type", "source", "author", "text"]
    assert len(df) == 3


def test_edge_dataframe_resolves_text(populated: tuple[list[Node], list[Edge]]) -> None:
    nodes, edges = populated
    df = streamlit_app._edge_dataframe(edges, nodes)
    assert list(df.columns) == [
        "src",
        "→",
        "dst",
        "relation",
        "confidence",
        "rationale",
    ]
    # ノードテキストが解決されている
    assert "A の主張" in df["dst"].tolist()
    assert "B の反論" in df["src"].tolist() or "B の反論" in df["dst"].tolist()


def test_list_run_directories_skips_files(tmp_path: Path) -> None:
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_b").mkdir()
    (tmp_path / "stray.txt").write_text("ignore me")
    result = streamlit_app._list_run_directories(tmp_path)
    assert {p.name for p in result} == {"run_a", "run_b"}


def test_list_run_directories_returns_empty_for_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    assert streamlit_app._list_run_directories(missing) == []


def test_render_graph_html_contains_node_ids(
    populated: tuple[list[Node], list[Edge]],
) -> None:
    nodes, edges = populated
    store = NetworkXGraphStore()
    for n in nodes:
        store.add_node(n)
    for e in edges:
        store.add_edge(e)
    html = streamlit_app._render_graph_html(store)
    for n in nodes:
        assert str(n.id) in html

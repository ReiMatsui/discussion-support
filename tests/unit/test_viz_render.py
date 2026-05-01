"""viz.render のユニットテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest

from das.graph.schema import Edge, Node
from das.graph.store import NetworkXGraphStore
from das.viz import dump_snapshot, load_snapshot, render_html


@pytest.fixture
def populated_store() -> NetworkXGraphStore:
    store = NetworkXGraphStore()
    n1 = Node(
        text="プラ容器を廃止すべき",
        node_type="claim",
        source="utterance",
        author="A",
    )
    n2 = Node(
        text="紙容器はコストが 3 倍で値上げにつながる",
        node_type="claim",
        source="utterance",
        author="B",
    )
    n3 = Node(
        text="X 大学では紙容器導入 2 年目にコスト増が解消した",
        node_type="premise",
        source="document",
        author="x_univ_case",
    )
    n4 = Node(
        text="バイオプラは +40% コストで生分解可能",
        node_type="premise",
        source="web",
        author="example.com",
    )
    for n in (n1, n2, n3, n4):
        store.add_node(n)
    store.add_edge(
        Edge(
            src_id=n2.id,
            dst_id=n1.id,
            relation="attack",
            confidence=0.9,
            rationale="コスト懸念は廃止論を弱める",
        )
    )
    store.add_edge(
        Edge(
            src_id=n3.id,
            dst_id=n2.id,
            relation="attack",
            confidence=0.85,
            rationale="先行事例がコスト増の永続性を否定",
        )
    )
    store.add_edge(
        Edge(
            src_id=n4.id,
            dst_id=n1.id,
            relation="support",
            confidence=0.7,
            rationale="第 3 の選択肢の存在",
        )
    )
    return store


def test_render_html_writes_file(
    populated_store: NetworkXGraphStore, tmp_path: Path
) -> None:
    out = render_html(populated_store, tmp_path / "graph.html")
    assert out.exists()
    html = out.read_text(encoding="utf-8")

    # 全ノードの UUID が HTML 内に書き出されている (ASCII なのでエスケープに左右されない)
    for node in populated_store.nodes():
        assert str(node.id) in html

    # エッジの relation 名と arrows 設定も含まれる
    assert "support" in html
    assert "attack" in html
    assert "to" in html  # arrows="to" の埋め込み


def test_render_html_creates_parent_directory(
    populated_store: NetworkXGraphStore, tmp_path: Path
) -> None:
    deep = tmp_path / "a" / "b" / "graph.html"
    out = render_html(populated_store, deep)
    assert out.exists()


def test_render_html_with_empty_store(tmp_path: Path) -> None:
    store = NetworkXGraphStore()
    out = render_html(store, tmp_path / "empty.html")
    assert out.exists()


# --- JSON 入出力 -------------------------------------------------------


def test_snapshot_roundtrip_via_json(
    populated_store: NetworkXGraphStore, tmp_path: Path
) -> None:
    path = tmp_path / "snap.json"
    dump_snapshot(populated_store, path)
    assert path.exists()

    loaded = load_snapshot(path)
    assert {n.id for n in loaded.nodes()} == {
        n.id for n in populated_store.nodes()
    }
    assert {e.id for e in loaded.edges()} == {
        e.id for e in populated_store.edges()
    }


def test_load_snapshot_into_existing_store(
    populated_store: NetworkXGraphStore, tmp_path: Path
) -> None:
    """既存ストアに重ね読みできることを確認。"""

    path = tmp_path / "snap.json"
    dump_snapshot(populated_store, path)

    other = NetworkXGraphStore()
    extra = Node(
        text="別の発話", node_type="claim", source="utterance", author="Z"
    )
    other.add_node(extra)

    load_snapshot(path, other)
    assert extra.id in {n.id for n in other.nodes()}
    assert {n.id for n in populated_store.nodes()}.issubset(
        {n.id for n in other.nodes()}
    )

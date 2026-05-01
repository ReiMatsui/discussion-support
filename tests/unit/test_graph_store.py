"""NetworkXGraphStore のユニットテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest

from das.graph.ops import (
    knowledge_nodes,
    support_attack_balance,
    unanswered_attacks,
    utterance_nodes,
)
from das.graph.schema import Edge, Node
from das.graph.store import GraphStore, NetworkXGraphStore


@pytest.fixture
def store() -> NetworkXGraphStore:
    return NetworkXGraphStore()


@pytest.fixture
def cafeteria_nodes() -> dict[str, Node]:
    """発表資料の例 (p7) を再現したノード集。"""

    a1 = Node(
        text="プラ容器は年間 2 トンのゴミを出しており、廃止すべき",
        node_type="claim",
        source="utterance",
        author="A",
    )
    a2 = Node(
        text="紙容器はコストが 3 倍で、学食の値上げにつながる",
        node_type="claim",
        source="utterance",
        author="B",
    )
    a3 = Node(
        text="X 大学の事例では紙容器導入後、コスト増は初年度のみで 2 年目に回収",
        node_type="premise",
        source="document",
        author="X-univ-case-study",
    )
    a4 = Node(
        text="最新のバイオプラ容器は従来比 +40% のコストで生分解可能",
        node_type="premise",
        source="web",
        author="example.com",
    )
    return {"a1": a1, "a2": a2, "a3": a3, "a4": a4}


def test_protocol_compliance(store: NetworkXGraphStore) -> None:
    assert isinstance(store, GraphStore)


def test_add_and_get_node(store: NetworkXGraphStore, cafeteria_nodes: dict[str, Node]) -> None:
    a1 = cafeteria_nodes["a1"]
    store.add_node(a1)
    assert store.get_node(a1.id) == a1
    assert list(store.nodes()) == [a1]


def test_idempotent_add(store: NetworkXGraphStore, cafeteria_nodes: dict[str, Node]) -> None:
    a1 = cafeteria_nodes["a1"]
    store.add_node(a1)
    store.add_node(a1)  # 二度目は無視される
    assert len(list(store.nodes())) == 1


def test_edge_requires_both_endpoints(
    store: NetworkXGraphStore, cafeteria_nodes: dict[str, Node]
) -> None:
    a1 = cafeteria_nodes["a1"]
    a2 = cafeteria_nodes["a2"]
    store.add_node(a1)
    edge = Edge(src_id=a2.id, dst_id=a1.id, relation="attack", confidence=0.9)
    with pytest.raises(ValueError):
        store.add_edge(edge)


def test_neighbors_direction(store: NetworkXGraphStore, cafeteria_nodes: dict[str, Node]) -> None:
    nodes = cafeteria_nodes
    for n in nodes.values():
        store.add_node(n)
    e_a2_attacks_a1 = Edge(
        src_id=nodes["a2"].id, dst_id=nodes["a1"].id, relation="attack", confidence=0.9
    )
    e_a3_attacks_a2 = Edge(
        src_id=nodes["a3"].id, dst_id=nodes["a2"].id, relation="attack", confidence=0.8
    )
    e_a4_supports_a1 = Edge(
        src_id=nodes["a4"].id, dst_id=nodes["a1"].id, relation="support", confidence=0.7
    )
    for e in (e_a2_attacks_a1, e_a3_attacks_a2, e_a4_supports_a1):
        store.add_edge(e)

    out_a2 = list(store.neighbors(nodes["a2"].id, direction="out"))
    assert {e.id for e in out_a2} == {e_a2_attacks_a1.id}

    in_a1 = list(store.neighbors(nodes["a1"].id, direction="in"))
    assert {e.id for e in in_a1} == {e_a2_attacks_a1.id, e_a4_supports_a1.id}


def test_snapshot_roundtrip(store: NetworkXGraphStore, cafeteria_nodes: dict[str, Node]) -> None:
    nodes = cafeteria_nodes
    for n in nodes.values():
        store.add_node(n)
    edge = Edge(src_id=nodes["a3"].id, dst_id=nodes["a2"].id, relation="attack", confidence=0.8)
    store.add_edge(edge)

    payload = store.snapshot()
    other = NetworkXGraphStore()
    other.load_snapshot(payload)

    assert {n.id for n in other.nodes()} == {n.id for n in nodes.values()}
    assert {e.id for e in other.edges()} == {edge.id}


def test_persistent_replay(tmp_path: Path, cafeteria_nodes: dict[str, Node]) -> None:
    db_path = tmp_path / "graph.sqlite"
    nodes = cafeteria_nodes
    store_a = NetworkXGraphStore(db_path=db_path)
    for n in nodes.values():
        store_a.add_node(n)
    edge = Edge(src_id=nodes["a3"].id, dst_id=nodes["a2"].id, relation="attack", confidence=0.8)
    store_a.add_edge(edge)
    store_a.close()

    store_b = NetworkXGraphStore(db_path=db_path)
    assert {n.id for n in store_b.nodes()} == {n.id for n in nodes.values()}
    assert {e.id for e in store_b.edges()} == {edge.id}
    store_b.close()


def test_ops_helpers(store: NetworkXGraphStore, cafeteria_nodes: dict[str, Node]) -> None:
    nodes = cafeteria_nodes
    for n in nodes.values():
        store.add_node(n)
    store.add_edge(
        Edge(src_id=nodes["a2"].id, dst_id=nodes["a1"].id, relation="attack", confidence=0.9)
    )
    store.add_edge(
        Edge(src_id=nodes["a3"].id, dst_id=nodes["a2"].id, relation="attack", confidence=0.8)
    )
    store.add_edge(
        Edge(src_id=nodes["a4"].id, dst_id=nodes["a1"].id, relation="support", confidence=0.7)
    )

    assert {n.id for n in utterance_nodes(store)} == {nodes["a1"].id, nodes["a2"].id}
    assert {n.id for n in knowledge_nodes(store)} == {nodes["a3"].id, nodes["a4"].id}

    balance = support_attack_balance(store)
    assert balance == {"support": 1, "attack": 1}

    # a1 は a2 から攻撃を受けて、a1 自身からは攻撃を返していない → 未応答
    unanswered_ids = {n.id for n in unanswered_attacks(store)}
    assert nodes["a1"].id in unanswered_ids

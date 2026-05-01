"""グラフに対する補助操作。

将来、ファシリテーションエージェントが
  - 攻撃が集中しているノード
  - 未応答の反論
  - 偏り (一方に支持が集中)
を検知するための土台になる。M1 では最低限のクエリだけ用意。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from uuid import UUID

from das.graph.schema import Edge, Node, NodeSource
from das.graph.store import GraphStore


def utterance_nodes(store: GraphStore) -> list[Node]:
    """発話由来のノードだけを返す。"""

    return [n for n in store.nodes() if n.source == "utterance"]


def knowledge_nodes(store: GraphStore) -> list[Node]:
    """文献・Web 由来のノードを返す。"""

    return [n for n in store.nodes() if n.source in {"document", "web"}]


def linked_to(store: GraphStore, node_id: UUID) -> list[Edge]:
    """対象ノードを始点・終点とするエッジ。"""

    return list(store.neighbors(node_id, direction="both"))


def unanswered_attacks(store: GraphStore) -> list[Node]:
    """攻撃を 1 件以上受け、それに対する反撃 (=攻撃→攻撃) が無い発話ノード。

    M1 段階の暫定実装。後でしきい値や時間窓を入れる。
    """

    incoming_attacks: dict[UUID, int] = Counter()
    has_response: set[UUID] = set()
    for edge in store.edges():
        if edge.relation != "attack":
            continue
        incoming_attacks[edge.dst_id] += 1
        # 反撃: 攻撃された側 (dst) からさらに攻撃が出ているか
        for back_edge in store.neighbors(edge.dst_id, direction="out"):
            if back_edge.relation == "attack":
                has_response.add(edge.dst_id)

    result: list[Node] = []
    for node_id, count in incoming_attacks.items():
        if count > 0 and node_id not in has_response:
            node = store.get_node(node_id)
            if node is not None and node.source == "utterance":
                result.append(node)
    return result


def support_attack_balance(
    store: GraphStore,
    sources: Iterable[NodeSource] = ("document", "web"),
) -> dict[str, int]:
    """指定ソース由来のノードからの support / attack 件数を集計する。"""

    source_set = set(sources)
    counts = {"support": 0, "attack": 0}
    for edge in store.edges():
        src = store.get_node(edge.src_id)
        if src is None or src.source not in source_set:
            continue
        counts[edge.relation] += 1
    return counts


__all__ = [
    "knowledge_nodes",
    "linked_to",
    "support_attack_balance",
    "unanswered_attacks",
    "utterance_nodes",
]

"""``das.eval.metrics`` のユニットテスト。"""

from __future__ import annotations

import pytest

from das.eval.metrics import (
    gini_coefficient,
    graph_metrics,
    transcript_metrics,
)
from das.graph.schema import Edge, Node
from das.graph.store import NetworkXGraphStore
from das.types import Utterance

# --- gini ----------------------------------------------------------------


def test_gini_empty() -> None:
    assert gini_coefficient([]) == 0.0


def test_gini_single() -> None:
    assert gini_coefficient([5]) == 0.0


def test_gini_uniform_is_zero() -> None:
    assert gini_coefficient([3, 3, 3]) == pytest.approx(0.0)


def test_gini_total_zero() -> None:
    assert gini_coefficient([0, 0, 0]) == 0.0


def test_gini_max_inequality() -> None:
    # 1 人だけが全部発言 → 1 - 1/n に近づく
    g = gini_coefficient([0, 0, 0, 100])
    assert 0.7 < g < 0.8  # n=4 の場合の理論最大は 0.75


# --- transcript_metrics --------------------------------------------------


def test_transcript_metrics_empty() -> None:
    m = transcript_metrics([])
    assert m.n_turns == 0
    assert m.n_chars_total == 0
    assert m.avg_chars_per_turn == 0.0
    assert m.speaker_turn_counts == {}
    assert m.gini_speaker_balance == 0.0


def test_transcript_metrics_basic() -> None:
    transcript = [
        Utterance(turn_id=1, speaker="A", text="一二三"),
        Utterance(turn_id=2, speaker="B", text="abcd"),
        Utterance(turn_id=3, speaker="A", text="xy"),
    ]
    m = transcript_metrics(transcript)
    assert m.n_turns == 3
    assert m.n_chars_total == 3 + 4 + 2
    assert m.avg_chars_per_turn == pytest.approx(9 / 3)
    assert m.speaker_turn_counts == {"A": 2, "B": 1}


# --- graph_metrics --------------------------------------------------------


def test_graph_metrics_empty_store() -> None:
    store = NetworkXGraphStore()
    g = graph_metrics(store)
    assert g.n_nodes == 0
    assert g.n_edges == 0
    assert g.support_attack_ratio is None


def test_graph_metrics_counts() -> None:
    store = NetworkXGraphStore()
    a = Node(text="a", node_type="claim", source="utterance", author="A")
    b = Node(text="b", node_type="claim", source="utterance", author="B")
    d = Node(text="d", node_type="premise", source="document", author="d1")
    w = Node(text="w", node_type="premise", source="web", author="example.com")
    for n in (a, b, d, w):
        store.add_node(n)
    store.add_edge(Edge(src_id=d.id, dst_id=a.id, relation="support", confidence=0.9))
    store.add_edge(Edge(src_id=b.id, dst_id=a.id, relation="attack", confidence=0.8))
    store.add_edge(Edge(src_id=w.id, dst_id=a.id, relation="support", confidence=0.7))

    g = graph_metrics(store)
    assert g.n_nodes == 4
    assert g.n_edges == 3
    assert g.n_utterance_nodes == 2
    assert g.n_document_nodes == 1
    assert g.n_web_nodes == 1
    assert g.n_support_edges == 2
    assert g.n_attack_edges == 1
    assert g.support_attack_ratio == pytest.approx(2 / 3)

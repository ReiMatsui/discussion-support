"""DiscussionStructuralMetrics のユニットテスト。"""

from __future__ import annotations

import pytest

from das.eval.structural_metrics import (
    aggregate_structural_metrics,
    compute_structural_metrics,
)
from das.graph.schema import Edge, Node
from das.graph.store import NetworkXGraphStore
from das.types import Utterance


def _utts(speakers: list[str]) -> list[Utterance]:
    return [
        Utterance(turn_id=i + 1, speaker=s, text=f"u{i + 1}")
        for i, s in enumerate(speakers)
    ]


def test_empty_inputs() -> None:
    m = compute_structural_metrics([], None)
    assert m.n_utterances == 0
    assert m.participation_gini == 0.0
    assert m.speaker_share == {}


def test_perfectly_equal_participation() -> None:
    transcript = _utts(["A", "B", "C", "A", "B", "C"])
    m = compute_structural_metrics(transcript, None)
    assert m.n_speakers == 3
    assert m.speaker_share["A"] == pytest.approx(1 / 3)
    assert m.speaker_share["B"] == pytest.approx(1 / 3)
    assert m.participation_gini == pytest.approx(0.0, abs=0.01)


def test_extremely_unequal_participation() -> None:
    transcript = _utts(["A"] * 9 + ["B"])  # A: 9, B: 1
    m = compute_structural_metrics(transcript, None)
    # 偏在度合いが非常に高い
    assert m.participation_gini >= 0.4


def test_unsupported_claim_pct() -> None:
    """発話 claim 2 件、片方だけ premise を持つ → pct_unsupported = 0.5。"""

    store = NetworkXGraphStore()
    c1 = Node(text="主張1", node_type="claim", source="utterance", author="A",
              metadata={"turn_id": 1})
    c2 = Node(text="主張2", node_type="claim", source="utterance", author="B",
              metadata={"turn_id": 2})
    p = Node(text="根拠", node_type="premise", source="document", author="d1")
    store.add_node(c1)
    store.add_node(c2)
    store.add_node(p)
    store.add_edge(Edge(src_id=p.id, dst_id=c1.id, relation="support", confidence=0.9))

    transcript = [
        Utterance(turn_id=1, speaker="A", text="x"),
        Utterance(turn_id=2, speaker="B", text="y"),
    ]
    m = compute_structural_metrics(transcript, store)
    assert m.n_utterance_claims == 2
    assert m.avg_premises_per_claim == pytest.approx(0.5)
    assert m.pct_unsupported_claims == pytest.approx(0.5)


def test_response_rate_when_each_utterance_responds_to_prior() -> None:
    """すべての発話が以前の発話に attack/support を張っていれば response_rate = 1.0。"""

    store = NetworkXGraphStore()
    n1 = Node(text="主張", node_type="claim", source="utterance", author="A",
              metadata={"turn_id": 1})
    n2 = Node(text="反論", node_type="claim", source="utterance", author="B",
              metadata={"turn_id": 2})
    store.add_node(n1)
    store.add_node(n2)
    # n2 (turn 2) が n1 (turn 1) を attack
    store.add_edge(Edge(src_id=n2.id, dst_id=n1.id, relation="attack", confidence=0.8))

    transcript = [
        Utterance(turn_id=1, speaker="A", text="x", timestamp=n1.timestamp),
        Utterance(turn_id=2, speaker="B", text="y", timestamp=n2.timestamp),
    ]
    m = compute_structural_metrics(transcript, store)
    # 2 ターン中 1 ターン (=t2) が応答 → response_rate = 0.5
    assert m.response_rate == pytest.approx(0.5)


def test_n_total_edges_propagated() -> None:
    store = NetworkXGraphStore()
    n1 = Node(text="x", node_type="claim", source="utterance", author="A")
    n2 = Node(text="y", node_type="claim", source="utterance", author="B")
    store.add_node(n1)
    store.add_node(n2)
    store.add_edge(Edge(src_id=n2.id, dst_id=n1.id, relation="attack", confidence=0.7))
    store.add_edge(Edge(src_id=n1.id, dst_id=n2.id, relation="support", confidence=0.7))

    transcript = [Utterance(turn_id=1, speaker="A", text="x")]
    m = compute_structural_metrics(transcript, store)
    assert m.n_total_edges == 2
    assert m.n_attack_edges == 1
    assert m.n_support_edges == 1


def test_aggregate_handles_empty_list() -> None:
    assert aggregate_structural_metrics([]) == {}


def test_aggregate_returns_means() -> None:
    runs = [
        compute_structural_metrics(_utts(["A", "B"]), None),
        compute_structural_metrics(_utts(["A", "A"]), None),
    ]
    agg = aggregate_structural_metrics(runs)
    assert agg["n_runs"] == 2
    # gini はそれぞれ 0.0 と (極端) のはず
    assert "participation_gini_mean" in agg

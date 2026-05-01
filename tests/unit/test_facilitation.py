"""FacilitationAgent のユニットテスト。"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from das.agents.facilitation import (
    BiasReport,
    FacilitationAgent,
    StageReport,
)
from das.graph.schema import Edge, Node
from das.graph.store import NetworkXGraphStore
from das.llm import OpenAIClient
from das.types import Utterance


def _fake_llm() -> OpenAIClient:
    return OpenAIClient(client=MagicMock())


@pytest.fixture
def cafeteria_store() -> tuple[NetworkXGraphStore, dict[str, Node]]:
    store = NetworkXGraphStore()
    a1 = Node(text="プラ容器を廃止すべき", node_type="claim", source="utterance", author="A")
    a2 = Node(
        text="紙容器はコスト 3 倍で値上げにつながる",
        node_type="claim",
        source="utterance",
        author="B",
    )
    a3 = Node(
        text="X 大学では紙容器導入 2 年目にコスト解消",
        node_type="premise",
        source="document",
        author="x_univ_case",
    )
    a4 = Node(
        text="バイオプラは +40% コストで生分解可能",
        node_type="premise",
        source="web",
        author="example.com",
    )
    for n in (a1, a2, a3, a4):
        store.add_node(n)
    return store, {"a1": a1, "a2": a2, "a3": a3, "a4": a4}


# --- detect_bias --------------------------------------------------------


def test_bias_balanced_when_no_edges() -> None:
    store = NetworkXGraphStore()
    agent = FacilitationAgent(llm=_fake_llm())
    bias = agent.detect_bias(store)
    assert bias.dominant_side == "balanced"
    assert bias.imbalance_ratio == 0.0


def test_bias_dominant_attack(
    cafeteria_store: tuple[NetworkXGraphStore, dict[str, Node]],
) -> None:
    store, n = cafeteria_store
    store.add_edge(Edge(src_id=n["a2"].id, dst_id=n["a1"].id, relation="attack", confidence=0.9))
    store.add_edge(Edge(src_id=n["a3"].id, dst_id=n["a1"].id, relation="attack", confidence=0.8))
    store.add_edge(Edge(src_id=n["a4"].id, dst_id=n["a1"].id, relation="support", confidence=0.7))

    agent = FacilitationAgent(llm=_fake_llm())
    bias = agent.detect_bias(store)
    assert bias.n_attack == 2
    assert bias.n_support == 1
    assert bias.dominant_side == "attack"
    assert bias.imbalance_ratio == pytest.approx(1 / 3)


def test_bias_weak_claims_detected(
    cafeteria_store: tuple[NetworkXGraphStore, dict[str, Node]],
) -> None:
    """発話 claim が 2 件以上 attack を受け support 0 のとき weak_claims に挙げられる。"""

    store, n = cafeteria_store
    store.add_edge(Edge(src_id=n["a2"].id, dst_id=n["a1"].id, relation="attack", confidence=0.9))
    store.add_edge(Edge(src_id=n["a3"].id, dst_id=n["a1"].id, relation="attack", confidence=0.8))
    agent = FacilitationAgent(llm=_fake_llm())
    bias = agent.detect_bias(store)
    assert n["a1"] in bias.weak_claims


def test_bias_over_supported_detected(
    cafeteria_store: tuple[NetworkXGraphStore, dict[str, Node]],
) -> None:
    store, n = cafeteria_store
    store.add_edge(Edge(src_id=n["a3"].id, dst_id=n["a1"].id, relation="support", confidence=0.9))
    store.add_edge(Edge(src_id=n["a4"].id, dst_id=n["a1"].id, relation="support", confidence=0.8))
    agent = FacilitationAgent(llm=_fake_llm())
    bias = agent.detect_bias(store)
    assert n["a1"] in bias.over_supported_claims


# --- detect_stage -------------------------------------------------------


def test_stage_diverge_with_diverse_speakers() -> None:
    transcript = [
        Utterance(turn_id=1, speaker="A", text="プラ容器の廃止は環境問題として重要"),
        Utterance(turn_id=2, speaker="B", text="しかし学食のコスト負担が増えるリスクがある"),
        Utterance(turn_id=3, speaker="C", text="先行事例だと長期的にはコストが回収できる模様"),
        Utterance(turn_id=4, speaker="D", text="バイオプラなど別の選択肢も検討すべき"),
    ]
    agent = FacilitationAgent(llm=_fake_llm())
    stage = agent.detect_stage(transcript)
    assert stage.speaker_diversity == 1.0
    assert stage.stage == "diverge"


def test_stage_stalled_with_repetitions() -> None:
    transcript = [
        Utterance(turn_id=i, speaker="A", text="プラ容器を廃止すべきだ。コスト軽視できる。")
        for i in range(1, 7)
    ]
    agent = FacilitationAgent(llm=_fake_llm())
    stage = agent.detect_stage(transcript)
    assert stage.stage == "stalled"
    assert stage.repetition_rate > 0.5


def test_stage_empty_transcript() -> None:
    agent = FacilitationAgent(llm=_fake_llm())
    stage = agent.detect_stage([])
    assert stage.stage == "diverge"
    assert stage.n_recent_turns == 0


# --- select_for_target --------------------------------------------------


def test_select_returns_empty_when_no_neighbors(
    cafeteria_store: tuple[NetworkXGraphStore, dict[str, Node]],
) -> None:
    store, n = cafeteria_store
    agent = FacilitationAgent(llm=_fake_llm())
    items = agent.select_for_target(n["a1"], store, [])
    assert items == []


def test_select_basic_adjacent(
    cafeteria_store: tuple[NetworkXGraphStore, dict[str, Node]],
) -> None:
    store, n = cafeteria_store
    store.add_edge(
        Edge(
            src_id=n["a2"].id,
            dst_id=n["a1"].id,
            relation="attack",
            confidence=0.9,
            rationale="コスト懸念",
        )
    )
    agent = FacilitationAgent(llm=_fake_llm())
    items = agent.select_for_target(n["a1"], store, [])
    assert len(items) == 1
    assert items[0].relation == "attack"
    assert items[0].source_text == n["a2"].text
    assert items[0].source_kind == "utterance"
    assert items[0].rationale == "コスト懸念"
    # 偏り無し → adjacent
    assert items[0].reason == "adjacent"


def test_select_balance_correction_lifts_minority_side(
    cafeteria_store: tuple[NetworkXGraphStore, dict[str, Node]],
) -> None:
    """全体が attack 優勢のとき、target への support の優先度が引き上げられる。"""

    store, n = cafeteria_store
    # 全体: 4 attack vs 1 support (attack 優勢、imbalance = 3/5 = 0.6)
    store.add_edge(Edge(src_id=n["a2"].id, dst_id=n["a3"].id, relation="attack", confidence=0.8))
    store.add_edge(Edge(src_id=n["a3"].id, dst_id=n["a2"].id, relation="attack", confidence=0.8))
    store.add_edge(Edge(src_id=n["a4"].id, dst_id=n["a3"].id, relation="attack", confidence=0.7))
    # target=a1 への支持 1 件 + 攻撃 1 件
    store.add_edge(Edge(src_id=n["a4"].id, dst_id=n["a1"].id, relation="support", confidence=0.6))
    store.add_edge(Edge(src_id=n["a2"].id, dst_id=n["a1"].id, relation="attack", confidence=0.6))

    agent = FacilitationAgent(llm=_fake_llm())
    items = agent.select_for_target(n["a1"], store, [])
    by_relation = {it.relation: it for it in items}
    # support が引き上げられて attack より優先される
    assert by_relation["support"].priority > by_relation["attack"].priority
    assert by_relation["support"].reason == "balance_correction"


def test_select_stage_alignment_in_stalled(
    cafeteria_store: tuple[NetworkXGraphStore, dict[str, Node]],
) -> None:
    """停滞時には attack の優先度が上がる (bias を中立化した状態で検証)。"""

    store, n = cafeteria_store
    # bias を中立にするため support 1 + attack 1
    store.add_edge(Edge(src_id=n["a3"].id, dst_id=n["a1"].id, relation="support", confidence=0.6))
    store.add_edge(Edge(src_id=n["a2"].id, dst_id=n["a1"].id, relation="attack", confidence=0.6))
    transcript = [
        Utterance(turn_id=i, speaker="A", text="プラ容器を廃止すべきだ。") for i in range(1, 7)
    ]
    agent = FacilitationAgent(llm=_fake_llm(), max_items=5)
    items = agent.select_for_target(n["a1"], store, transcript)
    by_relation = {it.relation: it for it in items}
    # stage=stalled で attack が 1.2 倍され、support は変化なし
    assert by_relation["attack"].priority > by_relation["support"].priority
    assert by_relation["attack"].priority > 0.6
    assert by_relation["attack"].reason == "stage_alignment"


def test_select_max_items_caps(
    cafeteria_store: tuple[NetworkXGraphStore, dict[str, Node]],
) -> None:
    store, n = cafeteria_store
    # a1 に多数の support / attack を集中
    extras = []
    for i in range(6):
        extra = Node(
            text=f"追加根拠 {i}",
            node_type="premise",
            source="document",
            author=f"d{i}",
        )
        store.add_node(extra)
        extras.append(extra)
        store.add_edge(
            Edge(
                src_id=extra.id,
                dst_id=n["a1"].id,
                relation="support" if i % 2 == 0 else "attack",
                confidence=0.6 + i * 0.05,
            )
        )
    agent = FacilitationAgent(llm=_fake_llm(), max_items=3)
    items = agent.select_for_target(n["a1"], store, [])
    assert len(items) == 3
    # 優先度降順
    priorities = [i.priority for i in items]
    assert priorities == sorted(priorities, reverse=True)


# --- BiasReport / StageReport の基礎 ----------------------------------


def test_bias_report_imbalance_ratio_bounds() -> None:
    b = BiasReport(n_support=10, n_attack=0, dominant_side="support")
    assert b.imbalance_ratio == 1.0
    b = BiasReport(n_support=5, n_attack=5, dominant_side="balanced")
    assert b.imbalance_ratio == 0.0


def test_stage_report_fields() -> None:
    s = StageReport(stage="converge", n_recent_turns=4, repetition_rate=0.2, speaker_diversity=0.5)
    assert s.stage == "converge"

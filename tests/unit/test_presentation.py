"""L3 (summary) と L4 (retrospective) のユニットテスト。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from das.graph.schema import Edge, Node
from das.graph.store import NetworkXGraphStore
from das.llm import OpenAIClient
from das.presentation.retrospective import (
    retrospective_for,
    retrospectives_by_speaker,
)
from das.presentation.summary import (
    llm_summary,
    programmatic_summary,
    summarize_session,
)
from das.types import Utterance


@pytest.fixture
def store_with_attacks() -> tuple[NetworkXGraphStore, dict[str, Node]]:
    store = NetworkXGraphStore()
    a1 = Node(text="主張 A", node_type="claim", source="utterance", author="A")
    b1 = Node(text="反論 B", node_type="claim", source="utterance", author="B")
    c1 = Node(text="主張 C", node_type="claim", source="utterance", author="C")
    d1 = Node(text="文書 D", node_type="premise", source="document", author="d1")
    for n in (a1, b1, c1, d1):
        store.add_node(n)
    # B が A を攻撃 → A は応答していない (= 未応答)
    store.add_edge(Edge(src_id=b1.id, dst_id=a1.id, relation="attack", confidence=0.9))
    # 文書 D が A を支持
    store.add_edge(Edge(src_id=d1.id, dst_id=a1.id, relation="support", confidence=0.8))
    # C が B を攻撃、B も C に反撃
    store.add_edge(Edge(src_id=c1.id, dst_id=b1.id, relation="attack", confidence=0.7))
    store.add_edge(Edge(src_id=b1.id, dst_id=c1.id, relation="attack", confidence=0.6))
    return store, {"a1": a1, "b1": b1, "c1": c1, "d1": d1}


@pytest.fixture
def transcript() -> list[Utterance]:
    return [
        Utterance(turn_id=1, speaker="A", text="主張 A を述べます"),
        Utterance(turn_id=2, speaker="B", text="それに反論します"),
        Utterance(turn_id=3, speaker="C", text="さらに主張 C を加えます"),
    ]


# --- programmatic_summary ----------------------------------------------


def test_programmatic_summary_empty_store() -> None:
    s = programmatic_summary(NetworkXGraphStore(), [])
    assert s.n_nodes == 0
    assert "始まっていません" in s.text


def test_programmatic_summary_basic_counts(
    store_with_attacks: tuple[NetworkXGraphStore, dict[str, Node]],
    transcript: list[Utterance],
) -> None:
    store, _ = store_with_attacks
    s = programmatic_summary(store, transcript)
    assert s.n_nodes == 4
    assert s.n_support == 1
    assert s.n_attack == 3
    # A は B から attack を受けたが反撃していない → 未応答
    assert s.unanswered_attacks >= 1
    # 構造ライン
    assert any("発言ターン数" in line for line in s.structural_lines)
    assert "ターン" in s.text


# --- llm_summary -------------------------------------------------------


async def test_llm_summary_calls_smart_model(
    store_with_attacks: tuple[NetworkXGraphStore, dict[str, Node]],
    transcript: list[Utterance],
) -> None:
    store, _ = store_with_attacks
    llm = OpenAIClient(client=MagicMock())
    captured = AsyncMock(return_value="3 文の自然な要約です。")
    llm.chat = captured  # type: ignore[method-assign]

    s = await llm_summary(store, transcript, llm)

    captured.assert_awaited_once()
    kwargs = captured.await_args.kwargs
    assert kwargs.get("model") == llm.smart_model
    assert s.text == "3 文の自然な要約です。"


async def test_summarize_session_falls_back_when_llm_none(
    store_with_attacks: tuple[NetworkXGraphStore, dict[str, Node]],
    transcript: list[Utterance],
) -> None:
    store, _ = store_with_attacks
    s = await summarize_session(store, transcript, llm=None)
    # programmatic_summary と同じ結果
    s2 = programmatic_summary(store, transcript)
    assert s.text == s2.text
    assert s.n_nodes == s2.n_nodes


# --- retrospective_for ------------------------------------------------


def test_retrospective_collects_unanswered_attacks(
    store_with_attacks: tuple[NetworkXGraphStore, dict[str, Node]],
    transcript: list[Utterance],
) -> None:
    store, _ = store_with_attacks
    retro = retrospective_for("A", store, transcript)
    assert len(retro.own_claims) == 1
    # A は B から攻撃を受けたが応答していない
    assert len(retro.unanswered_attacks) == 1
    assert retro.unanswered_attacks[0].attacker.author == "B"
    assert retro.answered_attacks == []
    # A は他者を攻撃していない
    assert retro.outgoing_attacks == []


def test_retrospective_speaker_who_replied(
    store_with_attacks: tuple[NetworkXGraphStore, dict[str, Node]],
    transcript: list[Utterance],
) -> None:
    store, _ = store_with_attacks
    retro = retrospective_for("B", store, transcript)
    # B は C から攻撃を受け、C に反撃した → answered
    assert len(retro.answered_attacks) >= 1
    # B 自身は A と C を攻撃した
    assert len(retro.outgoing_attacks) == 2


def test_retrospective_speaker_with_no_claims() -> None:
    store = NetworkXGraphStore()
    retro = retrospective_for("Z", store, [])
    assert retro.own_claims == []
    assert retro.unanswered_attacks == []
    assert "0 件" in retro.text_summary


def test_retrospectives_by_speaker_keys(
    store_with_attacks: tuple[NetworkXGraphStore, dict[str, Node]],
    transcript: list[Utterance],
) -> None:
    store, _ = store_with_attacks
    result = retrospectives_by_speaker(store, transcript)
    assert set(result.keys()) == {"A", "B", "C"}

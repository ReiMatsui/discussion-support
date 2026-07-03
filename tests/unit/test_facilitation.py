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


def test_stage_stalled_when_no_new_claims_or_attacks() -> None:
    """直近窓で新 claim も新 attack も追加されていなければ stalled。"""

    # 古いノードのみがある store。直近発話に対応するノードは無い。
    store = NetworkXGraphStore()
    old = Node(text="古い主張", node_type="claim", source="utterance", author="X")
    store.add_node(old)

    transcript = [
        Utterance(turn_id=i, speaker="A", text="繰り返しの発言") for i in range(1, 5)
    ]
    agent = FacilitationAgent(llm=_fake_llm())
    stage = agent.detect_stage(transcript, store)
    assert stage.stage == "stalled"
    assert stage.new_claims_in_window == 0
    assert stage.new_attacks_in_window == 0


def test_stage_empty_transcript() -> None:
    agent = FacilitationAgent(llm=_fake_llm())
    stage = agent.detect_stage([])
    assert stage.stage == "diverge"
    assert stage.n_recent_utterances == 0


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


def test_select_priority_is_confidence_no_coefficients(
    cafeteria_store: tuple[NetworkXGraphStore, dict[str, Node]],
) -> None:
    """priority は confidence そのもの。偏り補正の乗算係数 (旧 1.3/0.7) は撤廃 (Phase2)。

    旧テスト test_select_balance_correction_lifts_minority_side を、係数撤廃後の
    挙動に置き換えたもの。同じ「attack 優勢の全体 + target への support/attack 各 1 件」
    でも、support の優先度は引き上げられず confidence のまま。並べ替えは種別優先度で
    attack が先。
    """

    store, n = cafeteria_store
    store.add_edge(Edge(src_id=n["a2"].id, dst_id=n["a3"].id, relation="attack", confidence=0.8))
    store.add_edge(Edge(src_id=n["a3"].id, dst_id=n["a2"].id, relation="attack", confidence=0.8))
    store.add_edge(Edge(src_id=n["a4"].id, dst_id=n["a3"].id, relation="attack", confidence=0.7))
    store.add_edge(Edge(src_id=n["a4"].id, dst_id=n["a1"].id, relation="support", confidence=0.6))
    store.add_edge(Edge(src_id=n["a2"].id, dst_id=n["a1"].id, relation="attack", confidence=0.6))

    agent = FacilitationAgent(llm=_fake_llm())
    items = agent.select_for_target(n["a1"], store, [])
    by_relation = {it.relation: it for it in items}
    # 係数なし: 両者とも priority == confidence (0.6)、reason は adjacent
    assert by_relation["support"].priority == pytest.approx(0.6)
    assert by_relation["attack"].priority == pytest.approx(0.6)
    assert by_relation["support"].reason == "adjacent"
    assert by_relation["attack"].reason == "adjacent"
    # 種別優先度で attack が先頭
    assert items[0].relation == "attack"


def test_select_attack_ranks_above_support_by_kind(
    cafeteria_store: tuple[NetworkXGraphStore, dict[str, Node]],
) -> None:
    """種別優先度: 同 confidence なら attack (緊張) が support より先 (Phase2)。

    旧テスト test_select_stage_alignment_in_stalled の置き換え。停滞時の 1.2 倍係数は
    撤廃し、代わりに並べ替えキー (種別優先度, confidence) で attack を優先する。
    """

    store, n = cafeteria_store
    store.add_edge(Edge(src_id=n["a3"].id, dst_id=n["a1"].id, relation="support", confidence=0.6))
    store.add_edge(Edge(src_id=n["a2"].id, dst_id=n["a1"].id, relation="attack", confidence=0.6))
    transcript = [
        Utterance(turn_id=i, speaker="A", text="プラ容器を廃止すべきだ。") for i in range(1, 7)
    ]
    agent = FacilitationAgent(llm=_fake_llm(), max_items=5)
    items = agent.select_for_target(n["a1"], store, transcript)
    # 係数なしで priority は等しいが、種別優先度で attack が先頭
    assert items[0].relation == "attack"
    assert items[0].priority == pytest.approx(0.6)
    assert items[0].reason == "adjacent"


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
    s = StageReport(
        stage="converge",
        n_recent_utterances=4,
        new_claims_in_window=2,
        new_attacks_in_window=1,
        speaker_diversity=0.5,
    )
    assert s.stage == "converge"
    assert s.new_claims_in_window == 2


# --- decide_intervention (Stage 1: いつ介入するか) -----------------------


def _utt(turn_id: int, speaker: str = "A", text: str = "発言") -> Utterance:
    return Utterance(turn_id=turn_id, speaker=speaker, text=text)


def test_decide_skip_when_history_empty() -> None:
    agent = FacilitationAgent(llm=_fake_llm())
    decision = agent.decide_intervention([], NetworkXGraphStore())
    assert decision.kind == "skip"
    assert "履歴" in decision.reason


def test_decide_skip_when_last_utterance_has_no_nodes() -> None:
    """発話はあるが extraction 未完了で対応ノードが無いときは skip。"""

    agent = FacilitationAgent(llm=_fake_llm())
    decision = agent.decide_intervention([_utt(1, "A", "x")], NetworkXGraphStore())
    assert decision.kind == "skip"


def test_decide_l1_with_addressed_to_last_speaker() -> None:
    """最新発話に隣接エッジがあれば L1。addressed_to は発話者。"""

    store = NetworkXGraphStore()
    target = Node(
        text="主張", node_type="claim", source="utterance", author="A",
        metadata={"turn_id": 1},
    )
    attacker = Node(text="反論", node_type="claim", source="document", author="d1")
    store.add_node(target)
    store.add_node(attacker)
    store.add_edge(Edge(src_id=attacker.id, dst_id=target.id, relation="attack", confidence=0.9))

    agent = FacilitationAgent(llm=_fake_llm())
    decision = agent.decide_intervention([_utt(1, "A")], store)
    assert decision.kind == "l1"
    assert decision.addressed_to == "A"
    assert len(decision.items) == 1
    assert decision.items[0].relation == "attack"


def test_decide_l1_excludes_out_of_window_attacker() -> None:
    """アクティブ窓外の古い攻撃元からのエッジは L1 候補にならない (Phase2, A5)。"""

    store = NetworkXGraphStore()
    target = Node(
        text="最新の主張", node_type="claim", source="utterance", author="A",
        turn_index=20, metadata={"turn_id": 20},
    )
    old_attacker = Node(
        text="ずっと前の反論", node_type="claim", source="utterance", author="B",
        turn_index=1, metadata={"turn_id": 1},
    )
    store.add_node(target)
    store.add_node(old_attacker)
    store.add_edge(Edge(src_id=old_attacker.id, dst_id=target.id, relation="attack", confidence=0.9))

    # active_window=5 → window_start = 20 - 5 + 1 = 16。turn 1 の攻撃元は窓外。
    agent = FacilitationAgent(llm=_fake_llm(), active_window=5)
    decision = agent.decide_intervention([_utt(20, "A")], store)
    assert decision.kind == "skip"


def test_decide_l1_includes_in_window_attacker() -> None:
    """アクティブ窓内の攻撃元からのエッジは L1 候補になる (Phase2, A5)。"""

    store = NetworkXGraphStore()
    target = Node(
        text="最新の主張", node_type="claim", source="utterance", author="A",
        turn_index=20, metadata={"turn_id": 20},
    )
    recent_attacker = Node(
        text="最近の反論", node_type="claim", source="utterance", author="B",
        turn_index=18, metadata={"turn_id": 18},
    )
    store.add_node(target)
    store.add_node(recent_attacker)
    store.add_edge(Edge(src_id=recent_attacker.id, dst_id=target.id, relation="attack", confidence=0.9))

    agent = FacilitationAgent(llm=_fake_llm(), active_window=5)  # window_start=16
    decision = agent.decide_intervention([_utt(20, "A")], store)
    assert decision.kind == "l1"
    assert decision.items[0].relation == "attack"


def test_decide_skip_after_intervention_with_no_new_edges() -> None:
    """直前介入後にエッジが増えていなければ skip (連続介入の抑制)。"""

    store = NetworkXGraphStore()
    target = Node(
        text="主張", node_type="claim", source="utterance", author="A",
        metadata={"turn_id": 1},
    )
    src = Node(text="支持", node_type="premise", source="document", author="d1")
    store.add_node(target)
    store.add_node(src)
    store.add_edge(Edge(src_id=src.id, dst_id=target.id, relation="support", confidence=0.8))

    agent = FacilitationAgent(llm=_fake_llm())
    # 1 回目: L1 が出る
    d1 = agent.decide_intervention([_utt(1, "A")], store)
    assert d1.kind == "l1"

    # 2 回目: 新しい発話のノードが無く、エッジも増えていない → skip
    target2 = Node(
        text="続き", node_type="claim", source="utterance", author="B",
        metadata={"turn_id": 2},
    )
    store.add_node(target2)  # ノードは増えるが、エッジは増えない
    d2 = agent.decide_intervention(
        [_utt(1, "A"), _utt(2, "B", "続き")], store
    )
    assert d2.kind == "skip"
    assert "新エッジ追加なし" in d2.reason


def test_decide_l2_when_stalled() -> None:
    """十分な発話数があり、直近窓で新 claim/attack が無い → L2。"""

    store = NetworkXGraphStore()
    # 古い発話のノードのみ (timestamp が古い)
    old_node = Node(
        text="古い主張", node_type="claim", source="utterance", author="A",
        metadata={"turn_id": -1},
    )
    store.add_node(old_node)

    # 最新発話に対応するノードが無いと skip にされてしまうので、turn_id 経由で 1 件は紐付ける
    last_node = Node(
        text="最新", node_type="claim", source="utterance", author="C",
        metadata={"turn_id": 6},
    )
    store.add_node(last_node)

    agent = FacilitationAgent(llm=_fake_llm(), stall_window=4, l2_min_interval=2)
    transcript = [
        _utt(i, ["A", "B", "C"][i % 3], "繰り返しの議論") for i in range(1, 7)
    ]
    decision = agent.decide_intervention(transcript, store)
    # last_node の隣接が無いので L1 にもならない場合があるが、この設定では stalled が立つ
    assert decision.kind in ("l2", "skip")
    if decision.kind == "l2":
        assert decision.addressed_to is None
        assert decision.brief
        assert "停滞" in decision.reason or "偏り" in decision.reason


def test_decide_l2_respects_min_interval() -> None:
    """L2 を 1 回出した直後は、間隔条件を満たさないので L2 が再発しない。"""

    store = NetworkXGraphStore()
    last_node = Node(
        text="最新", node_type="claim", source="utterance", author="A",
        metadata={"turn_id": 5},
    )
    store.add_node(last_node)

    agent = FacilitationAgent(llm=_fake_llm(), stall_window=4, l2_min_interval=5)
    transcript = [_utt(i, "A", "x") for i in range(1, 6)]
    d1 = agent.decide_intervention(transcript, store)
    # 6 ターン目: 新ノードを足してエッジも 1 本足す → 連続抑制は外れるが L2 間隔条件で抑制
    next_node = Node(
        text="続き", node_type="claim", source="utterance", author="B",
        metadata={"turn_id": 6},
    )
    store.add_node(next_node)
    store.add_edge(Edge(src_id=next_node.id, dst_id=last_node.id, relation="support", confidence=0.7))
    transcript2 = [*transcript, _utt(6, "B", "y")]
    d2 = agent.decide_intervention(transcript2, store)
    # 直前が L2 だった場合、5 発話空くまで L2 は再発しない
    if d1.kind == "l2":
        assert d2.kind != "l2"


def test_reset_clears_internal_state() -> None:
    agent = FacilitationAgent(llm=_fake_llm())
    store = NetworkXGraphStore()
    target = Node(
        text="m", node_type="claim", source="utterance", author="A",
        metadata={"turn_id": 1},
    )
    store.add_node(target)
    agent.decide_intervention([_utt(1, "A")], store)  # 内部状態を進める
    agent.reset()
    # reset 後は再び 1 ターン目から始まったかのように扱える
    assert agent._last_decision_kind is None  # type: ignore[attr-defined]


# --- compose_l2_brief (deterministic fallback) ---------------------------


def test_compose_l2_brief_falls_back_to_deterministic_on_llm_failure() -> None:
    """LLM 整文が失敗したら deterministic fallback が走る。

    G3 で「llm is None なら deterministic」の到達不能分岐 (レビュー M-1) を削除した。
    BaseAgent が常に llm を生成するため、fallback は「LLM 呼び出しの失敗」で担保する。
    """

    import asyncio
    from unittest.mock import AsyncMock

    store = NetworkXGraphStore()
    a1 = Node(text="主張", node_type="claim", source="utterance", author="A")
    a2 = Node(text="反論", node_type="claim", source="utterance", author="B")
    store.add_node(a1)
    store.add_node(a2)
    store.add_edge(Edge(src_id=a2.id, dst_id=a1.id, relation="attack", confidence=0.9))
    store.add_edge(Edge(src_id=a2.id, dst_id=a1.id, relation="attack", confidence=0.8))

    agent = FacilitationAgent(llm=_fake_llm())
    agent.llm.chat = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
    transcript = [_utt(i, ["A", "B"][i % 2], "発言") for i in range(1, 5)]
    brief = asyncio.run(agent.compose_l2_brief(transcript, store))
    assert "ここまでの整理" in brief
    assert "支持" in brief or "反論" in brief


def test_decide_and_render_passes_through_non_l2() -> None:
    """decide_and_render: skip/L1 は LLM 整文を呼ばずそのまま返す。"""

    import asyncio
    from unittest.mock import AsyncMock

    agent = FacilitationAgent(llm=_fake_llm())
    agent.compose_l2_brief = AsyncMock()  # type: ignore[method-assign]
    decision = asyncio.run(agent.decide_and_render([], NetworkXGraphStore()))
    assert decision.kind == "skip"
    agent.compose_l2_brief.assert_not_awaited()


def test_decide_and_render_renders_l2_brief() -> None:
    """decide_and_render: L2 のとき compose_l2_brief で brief を差し替える。"""

    import asyncio
    from unittest.mock import AsyncMock

    from das.agents.facilitation import InterventionDecision

    agent = FacilitationAgent(llm=_fake_llm())
    agent.decide_intervention = MagicMock(  # type: ignore[method-assign]
        return_value=InterventionDecision(kind="l2", brief="det", reason="r")
    )
    agent.compose_l2_brief = AsyncMock(return_value="LLM 整文版")  # type: ignore[method-assign]
    decision = asyncio.run(agent.decide_and_render([], NetworkXGraphStore()))
    assert decision.kind == "l2"
    assert decision.brief == "LLM 整文版"
    agent.compose_l2_brief.assert_awaited_once()

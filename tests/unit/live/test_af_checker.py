"""AF checker + af 候補プラミング (H1 フェーズ4c) のユニットテスト。

AF 無効 (既定) では af 候補が一切出ず、ルールベース挙動が不変であることも確認する。
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from das.asr.live._facilitation import FacilitationController
from das.asr.live._workers import (
    _af_checker_tick,
    _af_l1_presentation,
    _AfEarlyGenGate,
    _build_candidates,
    _controller_normal_decision,
    _PendingInterventions,
)
from das.graph.schema import Edge, Node
from das.graph.store import NetworkXGraphStore


class _FakeState:
    def __init__(self, records, *, af_runtime=None, mode="facilitator"):
        self.state_lock = threading.Lock()
        self.meeting_epoch = 0
        self.records = records
        self.stop = threading.Event()
        self.af_requests: queue.Queue = queue.Queue()
        self.af_runtime = af_runtime
        self.agent = SimpleNamespace(mode=mode, pending_count=0)
        # drain() が直接参照するキュー群
        self.drift_requests: queue.Queue = queue.Queue()
        self.invite_requests: queue.Queue = queue.Queue()
        self.factcheck_requests: queue.Queue = queue.Queue()
        self.manual_call_requests: queue.Queue = queue.Queue()
        self.summarize_requests: queue.Queue = queue.Queue()

    def disp_name(self, key):
        return key


def _fac_agent():
    from das.agents.facilitation import FacilitationAgent

    return FacilitationAgent(llm=None)


def _store_with_unanswered_attack():
    store = NetworkXGraphStore()
    target = Node(text="最新主張", node_type="claim", source="utterance", author="A",
                  turn_index=1, metadata={"turn_id": 1})
    attacker = Node(text="コスト懸念", node_type="claim", source="utterance", author="B",
                    turn_index=1, metadata={"turn_id": 0})
    store.add_node(target)
    store.add_node(attacker)
    store.add_edge(Edge(src_id=attacker.id, dst_id=target.id, relation="attack", confidence=0.9))
    return store


# --- _af_checker_tick ---------------------------------------------------


def test_af_checker_noop_when_disabled():
    """af_runtime が無い (AF 既定 OFF) と何も積まない = ルールベース不変。"""
    state = _FakeState([{"speaker": "A", "text": "最新主張"}], af_runtime=None)
    assert _af_checker_tick(state, _fac_agent(), set()) == 0
    assert state.af_requests.empty()


def test_af_checker_pushes_af_l1_for_unanswered_attack():
    """AF 有効時、未応答攻撃があれば af_l1 候補を積む。"""
    runtime = SimpleNamespace(store=_store_with_unanswered_attack())
    state = _FakeState([{"speaker": "A", "text": "最新主張"}], af_runtime=runtime)
    n = _af_checker_tick(state, _fac_agent(), set())
    assert n == 1
    req = state.af_requests.get_nowait()
    assert req["kind"] == "af_l1"
    assert "コスト懸念" in req["af_text"]
    assert req["target_speaker"] == "A"


def test_af_checker_respects_presented_set():
    """提示済み source_text は価値ゲートで落ち、af 候補が積まれない。"""
    runtime = SimpleNamespace(store=_store_with_unanswered_attack())
    state = _FakeState([{"speaker": "A", "text": "最新主張"}], af_runtime=runtime)
    assert _af_checker_tick(state, _fac_agent(), {"コスト懸念"}) == 0
    assert state.af_requests.empty()


def test_af_checker_skips_conversation_mode():
    runtime = SimpleNamespace(store=_store_with_unanswered_attack())
    state = _FakeState([{"speaker": "A", "text": "x"}], af_runtime=runtime, mode="conversation")
    assert _af_checker_tick(state, _fac_agent(), set()) == 0


def test_af_l2_refire_guard_requires_new_nodes():
    """前回 af_l2 以降に新規発話ノードが4件追加されるまで af_l2 を再発火しない。"""
    from das.agents.facilitation import InterventionDecision

    store = _store_with_unanswered_attack()  # 発話ノード2件
    runtime = SimpleNamespace(store=store)
    state = _FakeState([{"speaker": "A", "text": "x"}], af_runtime=runtime)
    facil = _fac_agent()
    # decide を常に L2 (停滞) にして、ガードだけを検証する
    facil.decide_intervention = lambda transcript, s: InterventionDecision(  # type: ignore[method-assign]
        kind="l2", brief="俯瞰", reason="停滞")
    gate: dict = {}

    assert _af_checker_tick(state, facil, set(), gate) == 1  # 初回は発火
    assert gate["last_l2_node_count"] == 2
    # 新規ノード追加なし → 抑止
    assert _af_checker_tick(state, facil, set(), gate) == 0
    # 発話ノードを3件だけ追加 → まだ不足 (< 4)
    for i in range(3):
        store.add_node(Node(text=f"a{i}", node_type="claim", source="utterance",
                            author="A", turn_index=10 + i))
    assert _af_checker_tick(state, facil, set(), gate) == 0
    # さらに1件追加 → 計4件 → 再発火
    store.add_node(Node(text="a4", node_type="claim", source="utterance",
                        author="A", turn_index=20))
    assert _af_checker_tick(state, facil, set(), gate) == 1
    assert gate["last_l2_node_count"] == 6


# --- 候補プラミング -----------------------------------------------------


def test_pending_drains_af_requests():
    state = _FakeState([])
    state.af_requests.put({"kind": "af_l1", "brief": "b", "af_text": "t", "target_speaker": "A"})
    pending = _PendingInterventions()
    pending.drain(state, now=time.monotonic())
    assert pending.af is not None
    assert pending.af["kind"] == "af_l1"


def test_build_candidates_includes_af():
    now = time.monotonic()
    pending = _PendingInterventions()
    pending.af = {"kind": "af_l1", "brief": "提示", "af_text": "[反論] X",
                  "target_speaker": "A", "created_at": now}
    agent = SimpleNamespace(mode="facilitator", pending_count=0, _pending_intervention=None)
    cands = _build_candidates(pending, agent, now=now)
    af = [c for c in cands if c.kind == "af_l1"]
    assert len(af) == 1
    assert af[0].payload["af_text"] == "[反論] X"
    assert af[0].target_speaker == "A"


def test_no_af_candidate_when_pending_empty():
    """pending.af が無ければ af 候補は出ない (AF 無効時の不変性)。"""
    now = time.monotonic()
    pending = _PendingInterventions()
    agent = SimpleNamespace(mode="facilitator", pending_count=0, _pending_intervention=None)
    cands = _build_candidates(pending, agent, now=now)
    assert not any(c.kind in ("af_l1", "af_l2") for c in cands)


def test_pending_af_l2_suppresses_summarize():
    """保留中 af_l2 がある間は summarize 候補を生成しない (設計88f9a78)。"""
    now = time.monotonic()
    agent = SimpleNamespace(mode="facilitator", pending_count=0, _pending_intervention=None)
    # summarize と af_l2 が両方保留
    pending = _PendingInterventions()
    pending.summarize = {"focus": "整理", "created_at": now}
    pending.af = {"kind": "af_l2", "brief": "俯瞰", "af_text": "t", "created_at": now}
    kinds = {c.kind for c in _build_candidates(pending, agent, now=now)}
    assert "af_l2" in kinds
    assert "summarize" not in kinds  # 抑止された

    # af_l1 保留では summarize は抑止されない (af_l2 のときだけ)
    pending2 = _PendingInterventions()
    pending2.summarize = {"focus": "整理", "created_at": now}
    pending2.af = {"kind": "af_l1", "brief": "提示", "af_text": "t", "created_at": now}
    kinds2 = {c.kind for c in _build_candidates(pending2, agent, now=now)}
    assert "summarize" in kinds2
    assert "af_l1" in kinds2


def test_controller_normal_decision_maps_af():
    now = time.monotonic()
    pending = _PendingInterventions()
    pending.af = {"kind": "af_l1", "brief": "提示", "af_text": "[反論] X",
                  "target_speaker": "A", "created_at": now}
    agent = SimpleNamespace(mode="facilitator", pending_count=0, _pending_intervention=None)
    controller = FacilitationController()
    decision, _ctrl, _cands, _lat = _controller_normal_decision(
        controller, pending=pending, agent=agent, now=now,
        silence_elapsed=3.0, silence_summarize=None, partner_present=False,
        last_intervention_at=0.0, cooldown=8.0, last_invited=None,
        recent_interventions=[], epoch=1)
    assert decision.reason == "af_l1"
    assert decision.af_text == "[反論] X"


def test_drop_stale_af():
    now = time.monotonic()
    pending = _PendingInterventions()
    pending.af = {"kind": "af_l1", "created_at": now - 100.0}  # 45s TTL 超過
    pending.drop_stale_af(now=now)
    assert pending.af is None


# --- _AfEarlyGenGate: 生成先行・再生ゲートの状態機械 (フェーズ6) --------


def _gate_agent():
    return MagicMock(trigger=MagicMock(), release_playback=MagicMock(),
                     cancel_held=MagicMock())


def _af_l1(af_text="[反論] X"):
    return {"kind": "af_l1", "af_text": af_text, "target_speaker": "A"}


def test_gate_early_generates_at_0_3s():
    """沈黙 0.3s〜pause 未満 & agent フリーで hold 付き trigger する。"""
    gate = _AfEarlyGenGate()
    agent = _gate_agent()
    # af_l1 pause=1.5。silence 0.4 → 生成先行
    assert gate.tick(agent=agent, af=_af_l1(), silence=0.4,
                     new_utterance=False, agent_busy=False) == "trigger"
    assert gate.is_holding is True
    _, kwargs = agent.trigger.call_args
    assert kwargs["hold_playback"] is True
    assert kwargs["af_presentation"] == "[反論] X"


def test_gate_no_trigger_below_threshold_or_busy():
    gate = _AfEarlyGenGate()
    agent = _gate_agent()
    assert gate.tick(agent=agent, af=_af_l1(), silence=0.1,
                     new_utterance=False, agent_busy=False) == "none"
    assert gate.tick(agent=agent, af=_af_l1(), silence=0.5,
                     new_utterance=False, agent_busy=True) == "none"
    agent.trigger.assert_not_called()


def test_gate_no_early_trigger_when_pause_already_met():
    """沈黙が既に pause 以上なら生成先行しない (通常 dispatch が扱う)。"""
    gate = _AfEarlyGenGate()
    agent = _gate_agent()
    assert gate.tick(agent=agent, af=_af_l1(), silence=2.0,
                     new_utterance=False, agent_busy=False) == "none"
    assert gate.is_holding is False


def test_gate_releases_at_pause():
    gate = _AfEarlyGenGate()
    agent = _gate_agent()
    gate.tick(agent=agent, af=_af_l1(), silence=0.4, new_utterance=False, agent_busy=False)
    # 沈黙 1.4 (< pause 1.5) → まだ保留
    assert gate.tick(agent=agent, af=_af_l1(), silence=1.4,
                     new_utterance=False, agent_busy=True) == "holding"
    agent.release_playback.assert_not_called()
    # 沈黙 1.6 (>= pause) → フロア成立で再生
    assert gate.tick(agent=agent, af=_af_l1(), silence=1.6,
                     new_utterance=False, agent_busy=True) == "release"
    agent.release_playback.assert_called_once()
    assert gate.is_holding is False


def test_gate_cancels_on_new_utterance():
    gate = _AfEarlyGenGate()
    agent = _gate_agent()
    gate.tick(agent=agent, af=_af_l1(), silence=0.4, new_utterance=False, agent_busy=False)
    # フロア成立前に新規確定発話 → 破棄
    assert gate.tick(agent=agent, af=_af_l1(), silence=0.8,
                     new_utterance=True, agent_busy=True) == "cancel"
    agent.cancel_held.assert_called_once()
    assert gate.is_holding is False


def test_gate_cancels_when_candidate_disappears():
    gate = _AfEarlyGenGate()
    agent = _gate_agent()
    gate.tick(agent=agent, af=_af_l1(), silence=0.4, new_utterance=False, agent_busy=False)
    assert gate.tick(agent=agent, af=None, silence=0.8,
                     new_utterance=False, agent_busy=True) == "cancel"
    agent.cancel_held.assert_called_once()


def test_gate_af_l2_uses_longer_pause():
    """af_l2 は pause 2.0。1.8 では保留、2.1 で release。"""
    gate = _AfEarlyGenGate()
    agent = _gate_agent()
    af2 = {"kind": "af_l2", "af_text": "俯瞰", "target_speaker": None}
    gate.tick(agent=agent, af=af2, silence=0.5, new_utterance=False, agent_busy=False)
    assert gate.tick(agent=agent, af=af2, silence=1.8,
                     new_utterance=False, agent_busy=True) == "holding"
    assert gate.tick(agent=agent, af=af2, silence=2.1,
                     new_utterance=False, agent_busy=True) == "release"


def test_af_l1_presentation_labels():
    from das.agents.facilitation import InfoItem, InterventionDecision

    decision = InterventionDecision(
        kind="l1", addressed_to="A",
        items=[
            InfoItem(relation="attack", target_text="主張", target_speaker="A",
                     source_text="反論根拠", source_kind="utterance", source_author="B",
                     confidence=0.8),
            InfoItem(relation="support", target_text="主張", target_speaker="A",
                     source_text="支持根拠", source_kind="document", source_author="d1",
                     confidence=0.8),
        ])
    text = _af_l1_presentation(decision)
    assert "Aさん" in text
    assert "[反論] 反論根拠" in text
    assert "[支持] 支持根拠" in text

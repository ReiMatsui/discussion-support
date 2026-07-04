"""AF checker + af 候補プラミング (H1 フェーズ4c) のユニットテスト。

AF 無効 (既定) では af 候補が一切出ず、ルールベース挙動が不変であることも確認する。
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

from das.asr.live._facilitation import FacilitationController
from das.asr.live._workers import (
    _af_checker_tick,
    _af_l1_presentation,
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

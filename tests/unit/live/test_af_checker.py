"""AF checker + af 候補プラミング (H1 フェーズ4c) のユニットテスト。

AF 無効 (既定) では af 候補が一切出ず、ルールベース挙動が不変であることも確認する。
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from das.asr.live._facilitation import FacilitationController, InterventionLogEntry
from das.asr.live._workers import (
    _af_checker_tick,
    _af_gate_status,
    _af_l1_presentation,
    _af_l2_reason_type,
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


def _l2_facil(reason="停滞"):
    from das.agents.facilitation import InterventionDecision

    facil = _fac_agent()
    facil.decide_intervention = lambda transcript, s: InterventionDecision(  # type: ignore[method-assign]
        kind="l2", brief="俯瞰", reason=reason)
    return facil


def _add_utt_nodes(store, n, start=100):
    for i in range(n):
        store.add_node(Node(text=f"n{start + i}", node_type="claim", source="utterance",
                            author="A", turn_index=start + i))


def test_af_l2_reason_type_classifies():
    assert _af_l2_reason_type("停滞 (直近 4 発話で新 claim 0 件)") == "stalled"
    assert _af_l2_reason_type("未応答の反論 (窓内 unanswered=2)") == "bias"
    assert _af_l2_reason_type("構造的偏り") == "bias"
    assert _af_l2_reason_type("なにか別の") == "other"


def test_af_l2_same_reason_backoff_doubles():
    """同種理由・未改善のまま再発火が続くと必要新規ノード数が 4→8→16 と倍化する。"""
    store = _store_with_unanswered_attack()  # 発話ノード2件
    runtime = SimpleNamespace(store=store)  # _response_edges 無し → 応答0
    state = _FakeState([{"speaker": "A", "text": "x"}], af_runtime=runtime)
    facil = _l2_facil("停滞")
    gate: dict = {}

    assert _af_checker_tick(state, facil, set(), gate) == 1  # 初回発火 (base=4)
    # 1回目の再発火は base=4 で足りる
    _add_utt_nodes(store, 4, start=10)  # +4 → 計6
    assert _af_checker_tick(state, facil, set(), gate) == 1
    # 次は 8 必要。+4 (計10, 差4) では不足
    _add_utt_nodes(store, 4, start=20)
    assert _af_checker_tick(state, facil, set(), gate) == 0
    _add_utt_nodes(store, 4, start=30)  # +4 → 差8 → 発火
    assert _af_checker_tick(state, facil, set(), gate) == 1
    # 次は 16 必要。+8 では不足
    _add_utt_nodes(store, 8, start=40)
    assert _af_checker_tick(state, facil, set(), gate) == 0
    _add_utt_nodes(store, 8, start=60)  # +8 → 差16 → 発火 (上限)
    assert _af_checker_tick(state, facil, set(), gate) == 1


def test_af_l2_backoff_resets_on_reason_change():
    """理由タイプが変わればバックオフは base=4 に戻る。"""
    store = _store_with_unanswered_attack()
    runtime = SimpleNamespace(store=store)
    state = _FakeState([{"speaker": "A", "text": "x"}], af_runtime=runtime)
    gate: dict = {}

    assert _af_checker_tick(state, _l2_facil("停滞"), set(), gate) == 1
    _add_utt_nodes(store, 4, start=10)
    assert _af_checker_tick(state, _l2_facil("停滞"), set(), gate) == 1  # level→1 (次は8)
    # 理由が bias に変化 → リセット。base=4 で足りる
    _add_utt_nodes(store, 4, start=20)
    assert _af_checker_tick(state, _l2_facil("未応答の反論"), set(), gate) == 1
    assert gate["l2_backoff_level"] == 0


def test_af_l2_backoff_resets_on_response_edge():
    """応答エッジが増えたら (介入が届いた) バックオフを base に戻す。"""
    store = _store_with_unanswered_attack()
    runtime = SimpleNamespace(store=store, _response_edges=[])
    state = _FakeState([{"speaker": "A", "text": "x"}], af_runtime=runtime)
    facil = _l2_facil("停滞")
    gate: dict = {}

    assert _af_checker_tick(state, facil, set(), gate) == 1
    _add_utt_nodes(store, 4, start=10)
    assert _af_checker_tick(state, facil, set(), gate) == 1  # level→1
    # 応答エッジ検出 → リセット。次も base=4 で足りる
    runtime._response_edges.append({"intervention_id": "iv-1"})
    _add_utt_nodes(store, 4, start=20)
    assert _af_checker_tick(state, facil, set(), gate) == 1
    assert gate["l2_backoff_level"] == 0


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


def _silence_candidate(silence_summarize, partner_present):
    now = time.monotonic()
    agent = SimpleNamespace(mode="facilitator", pending_count=3, _pending_intervention=None)
    cands = _build_candidates(_PendingInterventions(), agent, now=now,
                              silence_summarize=silence_summarize,
                              partner_present=partner_present)
    return next((c for c in cands if c.kind == "silence"), None)


def test_silence_threshold_respects_profile_with_partner():
    """T1: Partner 同席でも silence_summarize=None (controlled) なら沈黙候補は出さず、
    有効なら max(profile, debate=15.0) を採る。"""
    # controlled (None): Partner の有無に関わらず沈黙候補なし
    assert _silence_candidate(None, partner_present=True) is None
    assert _silence_candidate(None, partner_present=False) is None
    # active (8.0): Partner ありは max(8,15)=15、Partner なしは 8
    c = _silence_candidate(8.0, partner_present=True)
    assert c is not None and c.payload["pause_required"] == 15.0
    c = _silence_candidate(8.0, partner_present=False)
    assert c is not None and c.payload["pause_required"] == 8.0
    # standard (18.0): Partner ありは max(18,15)=18 (プロファイルの方が長い)
    c = _silence_candidate(18.0, partner_present=True)
    assert c is not None and c.payload["pause_required"] == 18.0


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


def test_gate_early_generates_on_hold_status():
    """status=hold (採択見込み・間待ち) & agent フリー & 沈黙>=0.3 で hold 付き trigger。"""
    gate = _AfEarlyGenGate()
    agent = _gate_agent()
    assert gate.tick(agent=agent, af=_af_l1(), status="hold", silence=0.4,
                     new_utterance=False, agent_busy=False, now=0.0) == "trigger"
    assert gate.is_holding is True
    _, kwargs = agent.trigger.call_args
    assert kwargs["hold_playback"] is True
    assert kwargs["af_presentation"] == "[反論] X"


def test_gate_no_trigger_below_threshold_or_busy_or_status_none():
    gate = _AfEarlyGenGate()
    agent = _gate_agent()
    # 沈黙不足
    assert gate.tick(agent=agent, af=_af_l1(), status="hold", silence=0.1,
                     new_utterance=False, agent_busy=False) == "none"
    # agent busy
    assert gate.tick(agent=agent, af=_af_l1(), status="hold", silence=0.5,
                     new_utterance=False, agent_busy=True) == "none"
    # Controller が採択見込みでない (cooldown 等) → status=none
    assert gate.tick(agent=agent, af=_af_l1(), status="none", silence=0.5,
                     new_utterance=False, agent_busy=False) == "none"
    agent.trigger.assert_not_called()


def test_gate_immediate_deliver_when_pause_already_met():
    """取り込み遅延で候補が pause 通過後に来た (status=deliver) → 生成先行なしで即時配信。"""
    gate = _AfEarlyGenGate()
    agent = _gate_agent()
    assert gate.tick(agent=agent, af=_af_l1(), status="deliver", silence=2.0,
                     new_utterance=False, agent_busy=False) == "deliver"
    assert gate.is_holding is False
    _, kwargs = agent.trigger.call_args
    assert kwargs["hold_playback"] is False


def test_gate_releases_when_status_becomes_deliver():
    """hold 中に Controller が pause 成立で deliver に転じたら一斉再生。"""
    gate = _AfEarlyGenGate()
    agent = _gate_agent()
    agent.last_hold_to_release_ms = 120.0
    gate.tick(agent=agent, af=_af_l1(), status="hold", silence=0.4,
              new_utterance=False, agent_busy=False, now=0.0)
    # まだ間待ち (status=hold) → 保留
    assert gate.tick(agent=agent, af=_af_l1(), status="hold", silence=1.0,
                     new_utterance=False, agent_busy=True, now=0.5) == "holding"
    agent.release_playback.assert_not_called()
    # フロア成立 (status=deliver) → 再生 + hold_to_release 計測
    assert gate.tick(agent=agent, af=_af_l1(), status="deliver", silence=1.6,
                     new_utterance=False, agent_busy=True, now=1.0) == "release"
    agent.release_playback.assert_called_once()
    assert gate.is_holding is False
    assert gate.last_release_ms == 120.0


def test_gate_cancels_on_new_utterance():
    gate = _AfEarlyGenGate()
    agent = _gate_agent()
    gate.tick(agent=agent, af=_af_l1(), status="hold", silence=0.4,
              new_utterance=False, agent_busy=False, now=0.0)
    # フロア成立前に新規確定発話 → 破棄
    assert gate.tick(agent=agent, af=_af_l1(), status="hold", silence=0.8,
                     new_utterance=True, agent_busy=True, now=0.3) == "cancel"
    agent.cancel_held.assert_called_once()
    assert gate.is_holding is False


def test_gate_cancels_when_candidate_disappears():
    """hold 中に候補が消えた (TTL 失効など) → 破棄。"""
    gate = _AfEarlyGenGate()
    agent = _gate_agent()
    gate.tick(agent=agent, af=_af_l1(), status="hold", silence=0.4,
              new_utterance=False, agent_busy=False, now=0.0)
    assert gate.tick(agent=agent, af=None, status="none", silence=0.8,
                     new_utterance=False, agent_busy=True, now=0.3) == "cancel"
    agent.cancel_held.assert_called_once()


def test_gate_holds_while_floor_busy_then_times_out():
    """status=none (フロア占有) の間は保留し続け、上限超過で破棄する (抱え込まない)。"""
    gate = _AfEarlyGenGate()
    agent = _gate_agent()
    gate.tick(agent=agent, af=_af_l1(), status="hold", silence=0.4,
              new_utterance=False, agent_busy=False, now=0.0)
    # フロア占有中 (status=none だが候補は生存) → 保留継続
    assert gate.tick(agent=agent, af=_af_l1(), status="none", silence=1.0,
                     new_utterance=False, agent_busy=True, now=3.0) == "holding"
    # 上限 (8s) 超過 → 破棄
    assert gate.tick(agent=agent, af=_af_l1(), status="none", silence=1.0,
                     new_utterance=False, agent_busy=True,
                     now=_AfEarlyGenGate.MAX_HOLD_SEC + 0.1) == "cancel"
    agent.cancel_held.assert_called_once()


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


# --- _af_gate_status: Controller への WHEN 委譲 -------------------------


def _status_pending(kind="af_l1"):
    now = time.monotonic()
    pending = _PendingInterventions()
    pending.af = {"kind": kind, "brief": "b", "af_text": "[反論] X",
                  "target_speaker": "A", "created_at": now}
    return pending, now


def _status_agent():
    return SimpleNamespace(mode="facilitator", pending_count=0)


def test_af_gate_status_none_when_no_af():
    ctrl = FacilitationController()
    pending = _PendingInterventions()
    status, payload = _af_gate_status(
        ctrl, pending, _status_agent(), now=time.monotonic(), silence_elapsed=3.0,
        recent_interventions=[], cooldown=8.0, last_intervention_at=0.0, epoch=1,
        partner_busy=False, in_echo_window=False)
    assert status == "none"
    assert payload is None


def test_af_gate_status_hold_when_awaiting_pause():
    """沈黙が pause 未満 = awaiting_pause のみで抑制 → hold (採択見込み・間待ち)。"""
    ctrl = FacilitationController()
    pending, now = _status_pending("af_l1")  # pause=1.5
    status, payload = _af_gate_status(
        ctrl, pending, _status_agent(), now=now, silence_elapsed=0.5,
        recent_interventions=[], cooldown=8.0, last_intervention_at=0.0, epoch=1,
        partner_busy=False, in_echo_window=False)
    assert status == "hold"
    assert payload is pending.af


def test_af_gate_status_deliver_when_pause_met():
    ctrl = FacilitationController()
    pending, now = _status_pending("af_l1")
    status, _ = _af_gate_status(
        ctrl, pending, _status_agent(), now=now, silence_elapsed=3.0,
        recent_interventions=[], cooldown=8.0, last_intervention_at=0.0, epoch=1,
        partner_busy=False, in_echo_window=False)
    assert status == "deliver"


def test_af_gate_status_none_on_cooldown():
    """直前に同種 af_l1 を出したばかり (kind cooldown) なら hold ではなく none。"""
    ctrl = FacilitationController()
    pending, now = _status_pending("af_l1")
    recent = [InterventionLogEntry(at=now, kind="af_l1")]
    status, _ = _af_gate_status(
        ctrl, pending, _status_agent(), now=now, silence_elapsed=3.0,
        recent_interventions=recent, cooldown=8.0, last_intervention_at=0.0, epoch=1,
        partner_busy=False, in_echo_window=False)
    assert status == "none"


def test_af_gate_status_none_when_partner_busy():
    """パートナー発話中はフロア未成立 → hold にせず none (待つ)。"""
    ctrl = FacilitationController()
    pending, now = _status_pending("af_l1")
    status, _ = _af_gate_status(
        ctrl, pending, _status_agent(), now=now, silence_elapsed=0.5,
        recent_interventions=[], cooldown=8.0, last_intervention_at=0.0, epoch=1,
        partner_busy=True, in_echo_window=False)
    assert status == "none"

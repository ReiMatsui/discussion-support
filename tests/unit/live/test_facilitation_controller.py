"""FacilitationController（shadow 採否）と shadow ランナーのユニットテスト.

Phase1 の不変条件:
  - Controller は採否だけを行う（抽出・fact検査・文案生成をしない）。
  - shadow ランナーは実際の発話採否を変えず、判断だけをログする。
  - review ログ非対応の state では完全に no-op（既存挙動を壊さない）。
"""
from __future__ import annotations

import time

from das.asr.live._facilitation import (
    FacilitationController,
    FacilitationInput,
    InterventionCandidate,
    InterventionLogEntry,
    confidence_score,
    fact_expires_at,
)
from das.asr.live._workers import (
    _build_candidates,
    _controller_barge_in_decision,
    _controller_normal_decision,
    _PendingInterventions,
    _ShadowControllerRunner,
)


def _inp(candidates, *, silence_elapsed=5.0, recent=(), now=None, epoch=1):
    return FacilitationInput(
        candidates=tuple(candidates),
        recent_interventions=tuple(recent),
        silence_elapsed=silence_elapsed,
        snapshot_epoch=epoch,
        now=now if now is not None else time.monotonic(),
    )


def _fact(now, *, conf=0.9, cid="fact-1"):
    return InterventionCandidate(
        id=cid, kind="fact", brief="指標Xは分子を分母で割ります。",
        confidence=conf, created_at=now, expires_at=fact_expires_at(now))


def _drift(now):
    return InterventionCandidate(id="drift", kind="drift", brief="雑談に脱線",
                                 created_at=now)


# ---------------------------------------------------------------------------
# arbitrate()
# ---------------------------------------------------------------------------

def test_no_candidates_stays_silent():
    c = FacilitationController()
    d = c.arbitrate(_inp([]))
    assert d.candidate_id is None
    assert d.reason == "候補なし"
    assert d.suppressed == ()


def test_fact_preferred_over_drift_and_others_suppressed():
    now = time.monotonic()
    c = FacilitationController()
    d = c.arbitrate(_inp([_drift(now), _fact(now)], now=now))
    assert d.candidate_id == "fact-1"
    assert d.urgency == "wait_for_pause"
    assert d.deadline_ms == 1500
    # 採らなかった drift は理由付きで抑制に出る（透明性）
    assert any(s["candidate_id"] == "drift" for s in d.suppressed)


def test_short_silence_holds_all_and_stays_silent():
    now = time.monotonic()
    c = FacilitationController()
    d = c.arbitrate(_inp([_fact(now), _drift(now)], silence_elapsed=0.1, now=now))
    assert d.candidate_id is None
    reasons = {s["candidate_id"]: s["reason"] for s in d.suppressed}
    assert "発話の切れ目待ち" in reasons["fact-1"]
    assert "発話の切れ目待ち" in reasons["drift"]


def test_same_kind_cooldown_suppresses():
    now = time.monotonic()
    c = FacilitationController()
    recent = [InterventionLogEntry(at=now, kind="fact", brief="x")]
    d = c.arbitrate(_inp([_fact(now)], recent=recent, now=now + 0.5))
    assert d.candidate_id is None
    assert "同種介入済み" in d.suppressed[0]["reason"]


def test_expired_fact_is_suppressed():
    now = time.monotonic()
    c = FacilitationController()
    stale = InterventionCandidate(id="fact-old", kind="fact", brief="古い訂正",
                                  created_at=now - 1000, expires_at=now - 100)
    d = c.arbitrate(_inp([stale], now=now))
    assert d.candidate_id is None
    assert "期限切れ" in d.suppressed[0]["reason"]


def test_decision_uses_candidate_id_not_index():
    now = time.monotonic()
    c = FacilitationController()
    d = c.arbitrate(_inp([_fact(now, cid="fact-xyz")], now=now))
    assert d.candidate_id == "fact-xyz"


def test_higher_confidence_wins_within_same_kind():
    now = time.monotonic()
    c = FacilitationController()
    low = _fact(now, conf=0.2, cid="fact-low")
    high = _fact(now, conf=0.9, cid="fact-high")
    d = c.arbitrate(_inp([low, high], now=now))
    assert d.candidate_id == "fact-high"


def test_confidence_score_mapping():
    assert confidence_score("high") > confidence_score("medium") > confidence_score("low")
    assert confidence_score(None) == 0.0


# ---------------------------------------------------------------------------
# _build_candidates()（読み取り専用・pending を変えない）
# ---------------------------------------------------------------------------

class _FakeAgent:
    def __init__(self, *, mode="facilitator", pending_count=0, trigger_n=10,
                 pending_intervention=None):
        self.mode = mode
        self.pending_count = pending_count
        self.trigger_n = trigger_n
        self._pending_intervention = pending_intervention


def test_build_candidates_does_not_mutate_pending():
    now = time.monotonic()
    pend = _PendingInterventions()
    pend.facts.append({"correction": "訂正です。", "_queued_at": now,
                       "confidence": "high"})
    pend.drift_reason = "脱線"
    pend.invite = "参加者B"
    before_facts = len(pend.facts)

    cands = _build_candidates(pend, _FakeAgent(), now=now)
    kinds = {c.kind for c in cands}

    assert {"fact", "drift", "invite"} <= kinds
    # pending は一切変更されない（pop しない）
    assert len(pend.facts) == before_facts
    assert pend.drift_reason == "脱線"
    assert pend.invite == "参加者B"


def test_build_candidates_count_when_threshold_reached():
    now = time.monotonic()
    pend = _PendingInterventions()
    agent = _FakeAgent(pending_count=10, trigger_n=10)
    cands = _build_candidates(pend, agent, now=now)
    assert any(c.kind == "count" for c in cands)


def test_build_candidates_includes_conversation_silence_and_stall():
    now = time.monotonic()
    pend = _PendingInterventions()

    conversation = _build_candidates(
        pend, _FakeAgent(mode="conversation", pending_count=1), now=now)
    assert any(c.kind == "conversation" for c in conversation)

    silence = _build_candidates(
        pend, _FakeAgent(pending_count=1), now=now, silence_summarize=3.0)
    assert any(c.kind == "silence" and c.payload["pause_required"] == 3.0
               for c in silence)

    agent = _FakeAgent(pending_count=1)
    agent._last_noop_at = now - 10
    stall = _build_candidates(pend, agent, now=now, stall_breaker=True)
    assert any(c.kind == "stall" for c in stall)


def test_build_candidates_retry_from_pending_intervention():
    now = time.monotonic()
    pend = _PendingInterventions()
    agent = _FakeAgent(pending_intervention={"delivered": "中断された指摘",
                                              "created_at": now})
    cands = _build_candidates(pend, agent, now=now)
    retry = [c for c in cands if c.kind == "retry"]
    assert retry and retry[0].brief == "中断された指摘"


# ---------------------------------------------------------------------------
# _ShadowControllerRunner（並走・挙動不変）
# ---------------------------------------------------------------------------

class _ReviewState:
    """add_intervention_review を持つ最小 state."""

    def __init__(self):
        self.reviews: list[dict] = []

    def add_intervention_review(self, entry: dict) -> None:
        self.reviews.append(entry)


class _NoReviewState:
    """review ログ非対応の state（既存 FakeState 相当）."""


def test_shadow_runner_noop_without_review_support():
    """review 非対応の state では完全に no-op（既存挙動を壊さない）."""
    runner = _ShadowControllerRunner()
    pend = _PendingInterventions()
    pend.drift_reason = "脱線"
    # 例外を投げず、何もしないこと（state にメソッドが無くてもOK）
    runner.evaluate(_NoReviewState(), pending=pend, agent=_FakeAgent(),
                    now=time.monotonic(), silence_elapsed=5.0, epoch=0,
                    recent_interventions=[], legacy=None)


def test_shadow_runner_logs_decision_and_dedupes():
    runner = _ShadowControllerRunner()
    state = _ReviewState()
    pend = _PendingInterventions()
    agent = _FakeAgent()
    now = time.monotonic()

    # 候補なし → 「黙る」判断が1件記録される
    runner.evaluate(state, pending=pend, agent=agent, now=now,
                    silence_elapsed=5.0, epoch=0, recent_interventions=[],
                    legacy=None)
    assert len(state.reviews) == 1
    assert state.reviews[0]["controller_decision"]["candidate_id"] is None

    # drift 候補が出た → 採否が変化したので新たに記録
    pend.drift_reason = "雑談"
    pend.last_drift_request_at = now
    runner.evaluate(state, pending=pend, agent=agent, now=now,
                    silence_elapsed=5.0, epoch=1, recent_interventions=[],
                    legacy={"reason": "count", "detail": "10発話"})
    assert len(state.reviews) == 2
    rec = state.reviews[1]
    assert rec["controller_decision"]["candidate_id"] == "drift"
    assert rec["legacy_decision"]["reason"] == "count"
    assert "latency_ms" in rec
    assert rec["candidates"][0]["kind"] == "drift"

    # Controller採否が同じでも legacy が変われば比較上意味があるので記録する
    runner.evaluate(state, pending=pend, agent=agent, now=now,
                    silence_elapsed=5.0, epoch=2, recent_interventions=[],
                    legacy=None)
    assert len(state.reviews) == 3

    # 完全に同じ比較状態なら記録しない（ログ洪水を避ける）
    runner.evaluate(state, pending=pend, agent=agent, now=now,
                    silence_elapsed=5.0, epoch=3, recent_interventions=[],
                    legacy=None)
    assert len(state.reviews) == 3


# ---------------------------------------------------------------------------
# Phase2: 物理コンテキスト（partner/echo/確認/global cooldown）
# ---------------------------------------------------------------------------

def test_partner_busy_suppresses_all():
    now = time.monotonic()
    c = FacilitationController()
    d = c.arbitrate(FacilitationInput(
        candidates=(_fact(now), _drift(now)),
        recent_interventions=(), silence_elapsed=5.0, snapshot_epoch=1, now=now,
        partner_busy=True))
    assert d.candidate_id is None
    assert all("パートナー発話中" in s["reason"] for s in d.suppressed)


def test_echo_window_suppresses_all():
    now = time.monotonic()
    c = FacilitationController()
    d = c.arbitrate(FacilitationInput(
        candidates=(_fact(now),), recent_interventions=(), silence_elapsed=5.0,
        snapshot_epoch=1, now=now, in_echo_window=True))
    assert d.candidate_id is None
    assert "エコーウィンドウ" in d.suppressed[0]["reason"]


def test_drift_confirmation_gate_holds_single_detection():
    now = time.monotonic()
    c = FacilitationController()
    drift = InterventionCandidate(id="drift", kind="drift", brief="脱線",
                                  created_at=now, payload={"drift_count": 1})
    d = c.arbitrate(FacilitationInput(
        candidates=(drift,), recent_interventions=(), silence_elapsed=5.0,
        snapshot_epoch=1, now=now, required_drift_confirmations=2))
    assert d.candidate_id is None
    assert "確認待ち" in d.suppressed[0]["reason"]


def test_drift_fires_after_confirmations_met():
    now = time.monotonic()
    c = FacilitationController()
    drift = InterventionCandidate(id="drift", kind="drift", brief="脱線",
                                  created_at=now, payload={"drift_count": 2})
    d = c.arbitrate(FacilitationInput(
        candidates=(drift,), recent_interventions=(), silence_elapsed=5.0,
        snapshot_epoch=1, now=now, required_drift_confirmations=2))
    assert d.candidate_id == "drift"


def test_drift_global_cooldown_suppresses():
    """drift は『直前のあらゆる介入』から間隔が空くまで採らない（global cooldown）."""
    now = time.monotonic()
    c = FacilitationController()
    drift = InterventionCandidate(id="drift", kind="drift", brief="脱線",
                                  created_at=now, payload={"drift_count": 9})
    # 直前に別種(count)の介入があった → drift は global cooldown で待機
    d = c.arbitrate(FacilitationInput(
        candidates=(drift,), recent_interventions=(), silence_elapsed=5.0,
        snapshot_epoch=1, now=now + 1.0, cooldown=25.0,
        last_intervention_at=now, required_drift_confirmations=1))
    assert d.candidate_id is None
    assert "間隔不足" in d.suppressed[0]["reason"]


def test_fact_cooldown_override_is_respected():
    """fact の同種クールダウンは注入された fast lane 値を使う."""
    now = time.monotonic()
    c = FacilitationController()
    recent = (InterventionLogEntry(at=now, kind="fact"),)
    # override=1.0、経過0.5s → まだ抑制
    d = c.arbitrate(FacilitationInput(
        candidates=(_fact(now),), recent_interventions=recent, silence_elapsed=5.0,
        snapshot_epoch=1, now=now + 0.5, fact_cooldown=1.0))
    assert d.candidate_id is None
    # 経過1.5s → 許可
    d2 = c.arbitrate(FacilitationInput(
        candidates=(_fact(now),), recent_interventions=recent, silence_elapsed=5.0,
        snapshot_epoch=1, now=now + 1.5, fact_cooldown=1.0))
    assert d2.candidate_id == "fact-1"


def test_decision_carries_valid_for_epoch_and_deadline():
    now = time.monotonic()
    c = FacilitationController()
    d = c.arbitrate(FacilitationInput(
        candidates=(_fact(now),), recent_interventions=(), silence_elapsed=5.0,
        snapshot_epoch=42, now=now))
    assert d.valid_for_epoch == 42
    assert d.deadline_ms == 1500  # fact fast lane の予算


# ---------------------------------------------------------------------------
# Phase2: worker adapter（採否を _BargeInDecision/_NormalTriggerDecision に逆変換）
# ---------------------------------------------------------------------------

class _FakeProactivityState:
    def __init__(self, **proactivity):
        self.proactivity = {"drift_confirmations": 1, **proactivity}
        self.agent_cursor = 7


def _barge(agent, pending, state, **kw):
    defaults = dict(
        now=time.monotonic(), last_fact_at=0.0, last_intervention_at=0.0,
        silence_elapsed=10.0, partner_busy=False, in_echo_window=False,
        cooldown=25.0, recent_interventions=[], silence_summarize=18.0,
        stall_breaker=False, last_invited=None, epoch=7)
    defaults.update(kw)
    return _controller_barge_in_decision(FacilitationController(), pending=pending,
                                         agent=agent, state=state, **defaults)


def test_barge_adapter_picks_fact_over_drift():
    now = time.monotonic()
    pend = _PendingInterventions()
    pend.facts.append({"correction": "訂正です。", "_queued_at": now,
                       "confidence": "high"})
    pend.drift_reason = "脱線"
    pend.drift_count = 2
    decision, ctrl, _cands, _latency = _barge(_FakeAgent(), pend, _FakeProactivityState(), now=now)
    assert decision.reason == "fact"
    assert decision.fact["correction"] == "訂正です。"
    assert ctrl.candidate_id.startswith("fact-")


def test_barge_adapter_holds_when_partner_busy():
    now = time.monotonic()
    pend = _PendingInterventions()
    pend.facts.append({"correction": "訂正です。", "_queued_at": now,
                       "confidence": "high"})
    decision, _ctrl, _cands, _latency = _barge(
        _FakeAgent(), pend, _FakeProactivityState(), now=now, partner_busy=True)
    assert decision.reason == "hold"


def test_barge_adapter_none_when_no_candidates():
    decision, ctrl, cands, _latency = _barge(
        _FakeAgent(), _PendingInterventions(), _FakeProactivityState())
    assert decision.reason == "none"
    assert ctrl is None and cands == []


def _normal(agent, pending, **kw):
    defaults = dict(
        now=time.monotonic(), silence_elapsed=100.0, silence_summarize=18.0,
        partner_present=False, stall_breaker=False, last_intervention_at=0.0,
        cooldown=0.0, last_invited=None, recent_interventions=[], epoch=7)
    defaults.update(kw)
    return _controller_normal_decision(FacilitationController(), pending=pending,
                                       agent=agent, **defaults)


def test_normal_adapter_count_before_invite():
    agent = _FakeAgent(pending_count=10, trigger_n=10)
    pend = _PendingInterventions(invite="参加者B")
    decision, _ctrl, _cands, _latency = _normal(agent, pend)
    assert decision.reason == "count"


def test_normal_adapter_invite_fires_with_target():
    agent = _FakeAgent(pending_count=0)
    pend = _PendingInterventions(invite="参加者B")
    decision, _ctrl, _cands, _latency = _normal(agent, pend)
    assert decision.reason == "invite"
    assert decision.invite_target == "参加者B"


def test_normal_adapter_skip_invite_for_same_person():
    agent = _FakeAgent(pending_count=0)
    pend = _PendingInterventions(invite="参加者B")
    decision, _ctrl, _cands, _latency = _normal(agent, pend, last_invited="参加者B")
    assert decision.reason == "skip_invite"
    assert decision.invite_target == "参加者B"


def test_normal_adapter_invite_held_during_global_cooldown():
    """global cooldown 中は声かけを採らず、skip_invite でもなく none（候補は保持）."""
    now = time.monotonic()
    agent = _FakeAgent(pending_count=0)
    pend = _PendingInterventions(invite="参加者B")
    decision, _ctrl, _cands, _latency = _normal(
        agent, pend, now=now, last_intervention_at=now, cooldown=25.0)
    assert decision.reason == "none"


def test_barge_adapter_discards_drift_during_global_cooldown():
    """cooldown中のdriftは旧挙動どおり消費し、後で古い脱線介入を出さない."""
    now = time.monotonic()
    pend = _PendingInterventions(drift_reason="脱線", drift_count=3)
    agent = _FakeAgent(pending_intervention={"delivered": "中断介入", "created_at": now})

    decision, _ctrl, _cands, _latency = _barge(
        agent, pend, _FakeProactivityState(), now=now,
        last_intervention_at=now, cooldown=25.0)

    assert decision.reason == "retry"
    assert pend.drift_reason is None


def test_shadow_record_logs_supplied_controller_decision_without_reevaluation():
    runner = _ShadowControllerRunner()
    state = _ReviewState()
    now = time.monotonic()
    cand = InterventionCandidate(id="fact-1", kind="fact", brief="訂正",
                                 created_at=now, expires_at=fact_expires_at(now))
    decision = FacilitationController().arbitrate(FacilitationInput(
        candidates=(cand,), recent_interventions=(), silence_elapsed=0.0,
        snapshot_epoch=9, now=now, in_echo_window=True))

    runner.record(
        state, candidates=[cand], decision=decision, silence_elapsed=0.0,
        epoch=9, legacy={"reason": "hold", "detail": "echo_window"},
        latency_ms=1.2)

    assert state.reviews[0]["controller_decision"]["candidate_id"] is None
    assert "エコーウィンドウ" in state.reviews[0]["controller_decision"]["suppressed"][0]["reason"]

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

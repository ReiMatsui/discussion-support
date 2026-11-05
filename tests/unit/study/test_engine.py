from __future__ import annotations

import pytest
from pydantic import ValidationError

from das.study.engine import StudyConfig, StudyEngine, Turn, revision_of
from tests.support.study import review


def turns():
    return [
        Turn(id="t1", speaker="C", text="D案なら平日夜間に参加できます。", ms=1000),
        Turn(id="t2", speaker="B", text="D案は費用が重いので除外しよう。", ms=2000),
    ]




def engine(**kw):
    return StudyEngine(
        StudyConfig(
            session_id="synthetic",
            task_id="fixture",
            title="人工課題",
            options=["A案", "D案"],
            **kw,
        )
    )


def evaluate(e=None, r=None, ts=None, **kw):
    e = e or engine()
    ts = ts or turns()
    return e.evaluate(
        ts,
        r or review(),
        revision=revision_of(ts),
        now=100,
        new_turn_ids=kw.pop("new_turn_ids", {"t2"}),
        **kw,
    )


def test_history_and_current_reason_kept():
    c, status = evaluate()
    assert status == "prepared"
    assert "平日夜間に参加できます" in c.text
    assert "費用が重い" in c.text and c.option == "D案"


def test_general_request_remains_without_history():
    r = review()
    r.reasons = []
    c, status = evaluate(r=r)
    assert status == "prepared" and not c.history and "先ほど" not in c.text


def test_no_reason_is_not_invented():
    r = review()
    r.proposal.current_reason = None
    c, _ = evaluate(r=r)
    assert "費用が重い" not in c.text


@pytest.mark.parametrize("field", ["already_handled", "discussion_in_progress"])
def test_already_handled_suppresses(field):
    r = review()
    setattr(r.proposal, field, r.proposal.source)
    assert evaluate(r=r)[1] == "already_or_being_handled"


def test_no_proposal():
    r = review()
    r.proposal = None
    assert evaluate(r=r)[1] == "no_explicit_proposal"


def test_old_proposal_not_reopened():
    assert evaluate(new_turn_ids={"t1"})[1] == "old_proposal"


@pytest.mark.parametrize("target", ["reason", "proposal", "current", "handling"])
def test_absent_quotes_fail_closed(target):
    e = engine()
    r = review()
    if target == "reason":
        r.reasons[0].source.quote = "午前が良い"
    elif target == "proposal":
        r.proposal.source.quote = "採用しよう"
    elif target == "current":
        r.proposal.current_reason.quote = "費用が安い"
    else:
        r.reasons[0].handling = "adopted"
        r.reasons[0].handled_by = r.proposal.source.model_copy(update={"quote": "賛成"})
    with pytest.raises(ValueError):
        evaluate(e=e, r=r)
    assert e.reasons == []


def test_handled_reason_not_represented():
    r = review()
    r.reasons[0].handling = "refuted"
    r.reasons[0].handled_by = r.proposal.source
    c, _ = evaluate(r=r)
    assert not c.history


def test_future_reason_not_represented():
    ts = [*turns(), Turn(id="t3", speaker="A", text="D案なら新しい条件があります。", ms=3000)]
    r = review()
    r.reasons[0].source.turn_id = "t3"
    r.reasons[0].source.quote = "新しい条件があります"
    c, _ = evaluate(r=r, ts=ts)
    assert not c.history


def test_cooldown_and_cap_are_per_option():
    e = engine()
    c, _ = evaluate(e=e)
    rev = revision_of(turns())
    assert e.commit(c, revision=rev, now=101)
    assert not e.commit(c, revision=rev, now=102)
    assert evaluate(e=e)[1] == "already_requested"
    e.spoken_proposals.clear()
    assert evaluate(e=e)[1] == "cooldown"
    e.last_at.clear()
    e.counts["D案"] = 2
    assert evaluate(e=e)[1] == "option_cap"


def test_expired_and_stale_candidates_not_committed():
    e = engine()
    c, _ = evaluate(e=e)
    assert not e.commit(c, revision="new", now=101)
    assert not e.commit(c, revision=c.revision, now=116)
    assert not e.counts


def test_whole_quote_budget_does_not_fallback_to_general_request():
    e = engine(max_spoken_chars=45)
    assert evaluate(e=e)[1] == "history_too_long"


def test_duplicate_reason_not_repeated():
    r = review()
    r.reasons.append(r.reasons[0].model_copy(deep=True))
    c, _ = evaluate(r=r)
    assert len(c.history) == 1


def test_unknown_options_rejected():
    r = review()
    r.proposal.option = "知らない案"
    with pytest.raises(ValueError):
        evaluate(r=r)


def test_config_does_not_accept_private_payoffs():
    with pytest.raises(ValidationError):
        StudyConfig(session_id="s", task_id="t", title="t", options=["A", "B"], scores={"A": 100})


def test_all_ranked_history_that_fits_is_presented():
    ts = [
        Turn(id="t0", speaker="A", text="D案なら設備も使えます。", ms=500),
        *turns(),
    ]
    r = review()
    other = r.reasons[0].model_copy(deep=True)
    other.source.turn_id = "t0"
    other.source.quote = "設備も使えます"
    r.reasons.append(other)
    candidate, status = evaluate(r=r, ts=ts)
    assert status == "prepared"
    assert len(candidate.history) == 2
    assert candidate.history[0].source.turn_id == "t1"

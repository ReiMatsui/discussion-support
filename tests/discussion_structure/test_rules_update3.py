"""A decision consents for its speaker while keeping contrary stances intact."""

import pytest
from test_rules_update import judge, process, tracker, turn

from das.discussion_structure.models import ResolutionEvidence


@pytest.mark.parametrize(
    ("answer_type", "answer", "contrary"),
    [
        ("yes_no", "yes", "concern"),
        ("yes_no", "no", "support"),
        ("choice", "yes", "concern"),
        ("open", "yes", "concern"),
    ],
)
@pytest.mark.parametrize("owner", ["A", "B"])
def test_decision_waives_only_deciders_contrary_stance(answer_type, answer, contrary, owner):
    t = tracker(answer_type)
    target = "i1" if answer_type == "yes_no" else "p2"
    process(t, turn("stance", speaker=owner), judge(target, stance=contrary))
    stance = t.tree.nodes[target].stances[owner]
    process(
        t, turn("decide", ms=2000), judge(target, resolution="decide", decision_answer=answer)
    )
    process(t, turn("agree", ms=3000, speaker="C"), judge(target, agreement="yes"))
    t.advance(11999)
    assert t.tree.nodes["i1"].status == "open"
    t.advance(12000)
    if owner == "B":
        assert t.tree.nodes["i1"].status == "open"
        reason = "explicit_with_concern" if answer == "yes" else "explicit_with_support"
        assert t.resolution.candidates["i1"]["reason"] == reason
        process(t, turn("consent", ms=13000, speaker=owner), judge(target, agreement="yes"))
    issue = t.tree.nodes["i1"]
    assert issue.status == "decided"
    assert issue.decided_answer == (answer if answer_type == "yes_no" else None)
    assert t.tree.nodes[target].stances[owner] == stance
    evidence = issue.resolution_evidence[-1]
    assert evidence.decision_as_agreement is True
    assert evidence.decision_turn.turn_id == "decide"
    assert [source.turn_id for source in evidence.agreements] == (
        ["agree"] if owner == "A" else ["agree", "consent"]
    )
    assert [source.turn_id for source in evidence.explicit_agreements] == (
        ["agree"] if owner == "A" else ["agree", "consent"]
    )
    assert ResolutionEvidence.model_validate_json(evidence.model_dump_json()) == evidence
    event = next(e for e in t.events if e.type == "status_transition")
    assert event.data["resolution_evidence"]["decision_as_agreement"] is True
    assert not t.resolution.candidates


@pytest.mark.parametrize(("answer", "contrary"), [("yes", "concern"), ("no", "support")])
def test_deciders_consent_still_requires_independent_agreement(answer, contrary):
    t = tracker()
    process(t, turn("stance"), judge(stance=contrary))
    process(t, turn("decide", ms=2000), judge(resolution="decide", decision_answer=answer))
    process(t, turn("own_consent", ms=3000), judge(agreement="yes"))
    t.advance(12000)
    assert t.tree.nodes["i1"].status == "open"
    assert not t.tree.nodes["i1"].resolution_evidence
    assert t.resolution.candidates["i1"]["reason"] == "explicit_without_agreement"


@pytest.mark.parametrize(("answer", "contrary"), [("yes", "concern"), ("no", "support")])
@pytest.mark.parametrize("before_timer", [False, True])
def test_deciders_consent_follows_identity_correction(answer, contrary, before_timer):
    t = tracker()
    process(t, turn("stance", speaker_key="a"), judge(stance=contrary))
    process(
        t,
        turn("decide", ms=2000, speaker_key="a"),
        judge(resolution="decide", decision_answer=answer),
    )
    process(t, turn("agree", ms=3000, speaker="C"), judge(agreement="yes"))
    if not before_timer:
        t.advance(12000)
        assert t.tree.nodes["i1"].status == "decided"
    t.identify_speaker("cluster:a", turn("identity", ms=4000, speaker="B"))
    t.advance(12000)
    issue = t.tree.nodes["i1"]
    assert issue.status == "decided"
    assert issue.stances["B"].value == contrary
    assert not issue.needs_confirmation
    assert not t.snapshot()["internal"]["decision_confirmations"]
    evidence = issue.resolution_evidence[-1]
    assert evidence.decision_turn.uid == "B"
    assert evidence.decision_as_agreement is True


def test_hold_does_not_record_decider_consent():
    t = tracker()
    process(t, turn("hold"), judge(resolution="hold"))
    evidence = t.tree.nodes["i1"].resolution_evidence[-1]
    assert evidence.decision_as_agreement is False

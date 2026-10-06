"""Offline regression cases for the second approved rules update."""

import pytest
from test_rules_update import judge, process, tracker, turn

from das.discussion_structure.models import Issue, Judgement, ResolutionEvidence, certain


@pytest.mark.parametrize("answer_type", ["yes_no", "choice", "open"])
@pytest.mark.parametrize("consent", [False, True])
def test_concern_owner_must_explicitly_consent_even_with_another_agreement(answer_type, consent):
    t = tracker(answer_type)
    target = "i1" if answer_type == "yes_no" else "p2"
    process(t, turn("concern", speaker="B"), judge(target, stance="concern"))
    process(t, turn("decide", ms=2000), judge(target, resolution="decide"))
    process(t, turn("other", ms=3000, speaker="C"), judge(target, agreement="yes"))
    t.advance(12000)
    assert t.tree.nodes["i1"].status == "open"
    assert t.resolution.candidates["i1"]["reason"] == "explicit_with_concern"
    if consent:
        process(t, turn("consent", ms=13000, speaker="B"), judge(target, agreement="yes"))
        assert t.tree.nodes["i1"].status == "decided"
        assert not t.resolution.candidates
    assert t.tree.nodes[target].stances["B"].value == "concern"


def test_no_decision_requires_other_supporters_consent():
    owner = "B"
    t = tracker()
    process(t, turn("support", speaker=owner), judge(stance="support"))
    process(t, turn("concern", ms=2000, speaker="D"), judge(stance="concern"))
    process(t, turn("decide", ms=3000), judge(resolution="decide", decision_answer="no"))
    process(t, turn("other", ms=4000, speaker="C"), judge(agreement="yes"))
    t.advance(13000)
    assert t.tree.nodes["i1"].status == "open"
    assert t.resolution.candidates["i1"]["reason"] == "explicit_with_support"
    process(t, turn("consent", ms=14000, speaker=owner), judge(agreement="yes"))
    issue = t.tree.nodes["i1"]
    assert issue.status == "decided" and issue.decided_answer == "no"
    assert issue.stances[owner].value == "support"
    assert issue.stances["D"].value == "concern"


def test_low_probability_or_prior_agreement_does_not_waive_concern():
    t = tracker()
    process(t, turn("concern", speaker="B"), judge(stance="concern", agreement="yes"))
    process(t, turn("decide", ms=2000), judge(resolution="decide"))
    process(t, turn("other", ms=3000, speaker="C"), judge(agreement="yes"))
    j = Judgement(target=certain("i1"), agreement={"probabilities": {"yes": 0.79, "no": 0.21}})
    process(t, turn("weak", ms=4000, speaker="B"), j)
    t.advance(12000)
    assert t.tree.nodes["i1"].status == "open"


@pytest.mark.parametrize("answer_type", ["yes_no", "choice"])
def test_held_concern_is_take_home_and_returns_after_reopening(answer_type):
    t = tracker(answer_type)
    target = "i1" if answer_type == "yes_no" else "p2"
    process(t, turn("concern", speaker="B"), judge(target, stance="concern"))
    process(t, turn("hold", ms=2000), judge(target, resolution="hold"))
    t.advance(100000)
    group = t.snapshot()["internal"]["unanswered_concerns"][0]
    assert group["during_discussion"] is False
    assert group["summary_role"] == "take_home"
    assert group["concerns"][0]["turn_id"] == "concern"
    process(t, turn("reopen", ms=101000), judge(target, resolution="reopen"))
    process(
        t, turn("response", ms=102000, speaker="C"), judge(target, response_to="pending:reopen")
    )
    assert t.tree.nodes["i1"].status == "open"
    group = t.snapshot()["internal"]["unanswered_concerns"][0]
    assert group["during_discussion"] is True
    assert group["summary_role"] == "unanswered_concern"
    assert group["count"] == 1


@pytest.mark.parametrize("answer_type", ["yes_no", "choice"])
def test_identity_correction_marks_decision_without_changing_display(answer_type):
    t = tracker(answer_type)
    target = "i1" if answer_type == "yes_no" else "p2"
    process(t, turn("decide"), judge(target, resolution="decide"))
    process(t, turn("agree", ms=2000, speaker="B"), judge(target, agreement="yes"))
    t.advance(11000)
    t.finish()
    display = t.display.current
    commits = sum(e.type == "display_commit" for e in t.events)
    t.identify_speaker("B", turn("identity", ms=12000, speaker="A"))
    issue = t.tree.nodes["i1"]
    assert issue.status == "decided"
    assert issue.needs_confirmation
    assert issue.confirmation_reasons == ("no_independent_agreement",)
    assert t.snapshot()["internal"]["decision_confirmations"] == [
        {"issue_id": "i1", "reasons": ["no_independent_agreement"]}
    ]
    t.advance(15000)
    t.finish()
    assert t.display.current == display
    assert sum(e.type == "display_commit" for e in t.events) == commits
    assert Issue.model_validate_json(issue.model_dump_json()) == issue
    assert issue.resolution_evidence[-1].agreements[0].uid == "A"


def test_valid_identity_correction_does_not_mark_confirmation():
    t = tracker()
    process(t, turn("concern", speaker="B"), judge(stance="concern"))
    process(t, turn("decide", ms=2000), judge(resolution="decide"))
    process(t, turn("agree", ms=3000, speaker="B"), judge(agreement="yes"))
    process(t, turn("later_support", ms=4000, speaker="C"), judge(stance="support"))
    t.advance(12000)
    # Merge the concern owner into the newer supporter; consent follows the identity.
    t.identify_speaker("B", turn("identity", ms=13000, speaker="C"))
    assert not t.tree.nodes["i1"].needs_confirmation
    assert not t.snapshot()["internal"]["decision_confirmations"]


@pytest.mark.parametrize("interruption", ["none", "backchannel", "off", "other_issue"])
@pytest.mark.parametrize("answer_type", ["yes_no", "choice"])
def test_support_before_display_switch_counts_only_in_unbroken_run(interruption, answer_type):
    t = tracker(answer_type)
    t.tree.add(Issue(id="i3", label="告知するか？", answer_type="yes_no", parent_id="root"))
    t.tree.focus_id = "i3"
    t.focus.belief = {"i3": 1.0}
    target = "i1" if answer_type == "yes_no" else "p2"
    process(t, turn("support", speaker="B"), judge(target, stance="support"))
    assert t.tree.focus_id == "i3"
    if interruption == "other_issue":
        process(t, turn("interrupt", ms=2000, speaker="C"), judge("i3"))
    elif interruption == "backchannel":
        t.process(turn("interrupt", ms=2000, text="はい"))
    elif interruption == "off":
        process(t, turn("interrupt", ms=2000), judge(relevance="off"))
    process(t, turn("continue", ms=3000), judge(target))
    process(t, turn("decide", ms=14000), judge(target, resolution="decide"))
    assert t.tree.focus_id == "i1"
    t.advance(24000)
    assert (t.tree.nodes["i1"].status == "decided") == (interruption != "other_issue")


def test_new_issue_creation_starts_run_without_reusing_source_support():
    t = tracker()
    # Proposer support on creation is not manufactured for an explicit question.
    t.labels.labels["new"] = {
        "issue": {"label": "回収するか？", "answer_type": "yes_no", "original_position": "回収"}
    }
    process(t, turn("old", speaker="B"), judge(stance="support"))
    process(t, turn("new", ms=2000), judge("new_issue", trigger="explicit", parent="root"))
    process(t, turn("continue", ms=12000), judge("i2"))
    process(t, turn("decide", ms=13000), judge("i2", resolution="decide"))
    t.advance(23000)
    assert t.tree.nodes["i2"].status == "open"
    assert not t.resolution.pending["i2"].agreements


def test_confirmation_evidence_round_trip_keeps_explicit_consent():
    evidence = ResolutionEvidence(
        status="decided",
        answer="yes",
        target_id="i1",
        decision_turn=turn("decide"),
        explicit_agreements=(turn("consent", ms=2000, speaker="B"),),
    )
    assert ResolutionEvidence.model_validate_json(evidence.model_dump_json()) == evidence


def test_scripted_scenario_internal_results():
    from test_rules_update import replay_to

    t = replay_to("rules_update2", 18)
    assert t.tree.nodes["i3"].status == "held"
    assert t.nodes.unanswered_concerns()[0]["summary_role"] == "take_home"
    t = replay_to("rules_update2", 20)
    assert t.nodes.unanswered_concerns()[0]["during_discussion"]
    t = replay_to("rules_update2", 34)
    assert t.snapshot()["internal"]["decision_confirmations"] == [
        {"issue_id": "i5", "reasons": ["no_independent_agreement"]}
    ]
    assert t.tree.nodes["i5"].status == "decided"


def test_same_cluster_correction_before_timer_blocks_automatic_decision():
    t = tracker()
    process(t, turn("decide", speaker_key="a"), judge(resolution="decide"))
    process(t, turn("agree", ms=2000, speaker="B", speaker_key="b"), judge(agreement="yes"))
    t.identify_speaker("cluster:b", turn("correct", ms=3000, speaker="B", speaker_key="a"))
    t.advance(11000)
    assert t.tree.nodes["i1"].status == "open"
    assert t.resolution.candidates["i1"]["reason"] == "explicit_without_agreement"

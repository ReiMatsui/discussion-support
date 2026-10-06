"""Regression cases for the approved 2026-10-06 rules, with no network backends."""

import json
from pathlib import Path

import pytest

from das.discussion_structure.config import Config
from das.discussion_structure.judgement import ScriptedBackend
from das.discussion_structure.labels import ScriptedLabels
from das.discussion_structure.models import Issue, Judgement, Position, Turn, certain
from das.discussion_structure.tracker import Tracker

FIXTURES = Path(__file__).parents[1] / "fixtures/discussion_structure"


def turn(tid, ms=1000, speaker="A", **kw):
    return Turn(
        turn_id=tid,
        ms=ms - 1000,
        end_ms=ms,
        speaker=speaker,
        text=kw.pop("text", "内容を検討しましょう"),
        **kw,
    )


def judge(target="i1", **kw):
    return Judgement(target=certain(target), **{k: certain(v) for k, v in kw.items()})


def tracker(answer_type="yes_no"):
    t = Tracker("議題", ScriptedBackend({}), ScriptedLabels({}))
    t.tree.add(Issue(id="i1", label="導入するか？", answer_type=answer_type, parent_id="root"))
    t.tree.focus_id = "i1"
    t.focus.belief = {"i1": 1.0}
    if answer_type != "yes_no":
        t.tree.add(Position(id="p2", label="紙", parent_id="i1"))
    t.nodes.counter = 2 if answer_type != "yes_no" else 1
    return t


def process(t, raw, judgement):
    t.backend.judgements[raw.turn_id] = judgement.model_dump()
    t.process(raw)


def replay_to(name, count):
    data = json.loads((FIXTURES / f"{name}.json").read_text())
    t = Tracker(data["agenda"], ScriptedBackend(data["judgements"]), ScriptedLabels(data["labels"]))
    for row in data["turns"][:count]:
        t.process(Turn.model_validate(row))
    return t


@pytest.mark.parametrize("answer_type", ["yes_no", "choice", "open"])
def test_prior_support_is_agreement_for_adopted_answer_only(answer_type):
    t = tracker(answer_type)
    target = "i1" if answer_type == "yes_no" else "p2"
    process(t, turn("support", speaker="B"), judge(target, stance="support"))
    process(t, turn("neutral", ms=2000, speaker="B"), judge(target))
    process(t, turn("decision", ms=3000), judge(target, resolution="decide"))
    t.advance(12999)
    assert t.tree.nodes["i1"].status == "open"
    t.advance(13000)
    issue = t.tree.nodes["i1"]
    assert issue.status == "decided"
    evidence = issue.resolution_evidence[0]
    assert evidence.target_id == target
    assert evidence.decision_turn.turn_id == "decision"
    assert [source.turn_id for source in evidence.agreements] == ["support"]
    assert t.tree.nodes[target].stances["B"].source_turns == ("support",)
    assert evidence.summary_confirmation == ()
    if answer_type != "yes_no":
        assert t.tree.nodes[target].status == "adopted"
        assert evidence.answer is None


def test_other_answer_and_own_prior_support_do_not_count():
    t = tracker("open")
    t.tree.add(Position(id="p3", label="バイオプラ", parent_id="i1"))
    process(t, turn("other", speaker="B"), judge("p3", stance="support"))
    process(t, turn("own", ms=2000), judge("p2", stance="support"))
    process(t, turn("decision", ms=3000), judge("p2", resolution="decide"))
    t.advance(13000)
    assert t.tree.nodes["i1"].status == "open"
    assert t.resolution.candidates["i1"]["reason"] == "explicit_without_agreement"


def test_prior_support_is_discarded_after_focus_exit_and_return():
    t = replay_to("focus_agreement_rules", 14)
    assert t.tree.nodes["i2"].status == "open"
    assert not t.resolution.pending["i2"].agreements
    assert t.resolution.candidates["i2"]["reason"] == "explicit_without_agreement"
    assert [
        (e.data["previous"], e.data["current"]) for e in t.events if e.type == "focus_transition"
    ] == [("root", "i1"), ("i1", "i2"), ("i2", "i3"), ("i3", "i2")]


def test_no_decision_retains_concern_and_uses_later_agreement_evidence():
    t = replay_to("concern_identity_rules", 9)
    assert not t.resolution.pending["i1"].agreements
    assert not t.tree.nodes["i1"].resolution_evidence
    assert t.snapshot()["internal"]["unanswered_concerns"][0]["count"] == 1
    data = json.loads((FIXTURES / "concern_identity_rules.json").read_text())
    t.process(Turn.model_validate(data["turns"][9]))
    issue = t.tree.nodes["i1"]
    assert issue.decided_answer == "no"
    assert issue.stances["cluster:c"].value == "concern"
    assert issue.stances["cluster:a"].value == "support"
    assert not t.snapshot()["internal"]["unanswered_concerns"]
    evidence = issue.resolution_evidence[0]
    assert evidence.answer == "no"
    assert evidence.decision_turn.turn_id == "t8"
    assert [(s.uid, s.turn_id) for s in evidence.agreements] == [("cluster:d", "t10")]
    assert "cluster:d" not in issue.stances
    assert t.tree.snapshot()["nodes"][1]["resolution_evidence"][0]["answer"] == "no"


def test_grouped_concerns_end_on_response_and_owner_change_only():
    t = replay_to("concern_identity_rules", 4)
    groups = t.snapshot()["internal"]["unanswered_concerns"]
    assert len(groups) == 1 and groups[0]["count"] == 2
    data = json.loads((FIXTURES / "concern_identity_rules.json").read_text())
    t.process(Turn.model_validate(data["turns"][4]))
    groups = t.snapshot()["internal"]["unanswered_concerns"]
    assert groups[0]["count"] == 1
    assert groups[0]["concerns"][0]["turn_id"] == "t4"
    assert t.tree.nodes["i1"].stances["cluster:b"].value == "concern"
    t.process(Turn.model_validate(data["turns"][5]))
    assert not t.snapshot()["internal"]["unanswered_concerns"]


@pytest.mark.parametrize("end", ["withdraw", "agenda", "self", "time", "hold"])
def test_concern_lifetime_is_structural(end):
    t = tracker()
    process(t, turn("concern", speaker="B"), judge(stance="concern"))
    t.advance(120000)
    assert len(t.snapshot()["internal"]["unanswered_concerns"]) == 1
    if end == "agenda":
        t.switch_agenda("次の議題")
        assert len(t.archived_agendas) == 1
        assert t.archived_agendas[0]["nodes"][1]["stances"]["B"]["value"] == "concern"
        assert len(t.tree.nodes) == 1
    elif end == "self":
        process(t, turn("correct", ms=121000, speaker="B"), judge(correction="self"))
    elif end != "time":
        process(t, turn("end", ms=121000), judge(resolution=end))
        assert t.tree.nodes["i1"].resolution_evidence[0].decision_turn.turn_id == "end"
    assert bool(t.snapshot()["internal"]["unanswered_concerns"]) == (end in {"time", "hold"})


def test_reopen_expires_at_60_seconds_and_late_response_does_not_reopen():
    t = tracker()
    t.tree.replace("i1", status="decided", decided_answer="yes")
    process(t, turn("reopen"), judge(resolution="reopen"))
    t.advance(60999)
    assert "pending:reopen" in t.nodes.pending
    process(t, turn("late", ms=61000, speaker="B"), judge(response_to="pending:reopen"))
    assert not t.nodes.pending
    assert t.tree.nodes["i1"].status == "decided"
    assert [e.ms for e in t.events if e.type == "candidate_expired"] == [61000]


@pytest.mark.parametrize("kind", ["implicit", "reopen", "agreement"])
@pytest.mark.parametrize(
    "keys,unsure,uid,accepted",
    [
        ((None, None), True, None, False),
        ((None, None), True, "u2", False),
        (("same", "same"), True, None, False),
        (("a", "b"), True, None, True),
        (("a", None), True, None, False),
        ((None, None), False, None, True),
    ],
)
def test_independent_speaker_guard_shared_by_responses_and_agreement(
    kind, keys, unsure, uid, accepted
):
    t = tracker()
    first = turn("first", speaker_key=keys[0])
    if kind == "agreement":
        process(t, first, judge(resolution="decide"))
        judgement = judge(agreement="yes")
    else:
        if kind == "reopen":
            t.tree.replace("i1", status="decided")
            process(t, first, judge(resolution="reopen"))
        else:
            process(t, first, judge("new_issue", trigger="proposal", parent="root"))
            t.labels.labels["second"] = {
                "issue": {
                    "label": "回収制にするか？",
                    "answer_type": "yes_no",
                    "original_position": "回収制",
                }
            }
        judgement = judge(response_to="pending:first")
    second = turn(
        "second", ms=2000, speaker="B", unsure=unsure, speaker_key=keys[1], speaker_uid=uid
    )
    process(t, second, judgement)
    if kind == "agreement":
        assert bool(t.resolution.pending["i1"].agreements) == accepted
    elif kind == "reopen":
        assert (t.tree.nodes["i1"].status == "open") == accepted
    else:
        assert (len(t.tree.nodes) == 3) == accepted


@pytest.mark.parametrize(
    "confidence,unsure,name",
    [
        (0.49, False, "?"),
        (0.5, False, "A"),
        (0.9, True, "?"),
        (None, False, "A"),
    ],
)
def test_name_threshold_and_legacy_missing_confidence(confidence, unsure, name):
    t = tracker()
    process(
        t,
        turn("stance", speaker_uid="u", speaker_confidence=confidence, unsure=unsure),
        judge(stance="support"),
    )
    assert t.tree.nodes["i1"].stances["u"].name == name
    assert t.config.name_threshold == 0.5
    assert Config.load(Path("configs/discussion_structure.toml")).name_threshold == 0.5


def test_later_identification_updates_other_node_without_changing_stance():
    t = tracker()
    process(t, turn("stance", speaker_key="a", unsure=True), judge(stance="concern"))
    process(t, turn("later", ms=2000, speaker_key="a", speaker_confidence=0.5), judge("root"))
    stance = t.tree.nodes["i1"].stances["cluster:a"]
    assert stance.name == "A" and stance.value == "concern"
    assert stance.source_turns == ("stance",)
    assert t.snapshot()["internal"]["unanswered_concerns"][0]["concerns"][0]["name"] == "A"


@pytest.mark.parametrize("corrected_ms", [1000, 3000])
def test_retrospective_correction_merges_marks_and_keeps_newest(corrected_ms):
    t = tracker()
    old = turn("uncertain", ms=corrected_ms, speaker="?", unsure=True)
    known = turn("known", ms=2000, speaker="B")
    for raw, stance in sorted([(old, "concern"), (known, "support")], key=lambda x: x[0].end_ms):
        process(t, raw, judge(stance=stance))
    t.identify_speaker(old.uid, turn("identity", ms=4000, speaker="B"))
    t.nodes.clean_concerns()
    assert list(t.tree.nodes["i1"].stances) == ["B"]
    stance = t.tree.nodes["i1"].stances["B"]
    assert stance.value == ("concern" if corrected_ms == 3000 else "support")
    assert stance.name == "B"


def test_focus_interval_support_and_waiting_concern_transfer_on_upgrade():
    t = tracker()
    process(t, turn("support", speaker="B"), judge(stance="support"))
    process(t, turn("concern", ms=2000, speaker="D"), judge(stance="concern"))
    t.labels.labels["upgrade"] = {
        "upgrade": {"label": "何を導入するか？", "answer_type": "choice"},
        "position": {"label": "バイオプラ"},
    }
    # The original yes/no issue in this unit fixture needs the proposal label.
    t.tree.replace("i1", original_position="紙")
    process(
        t,
        turn("upgrade", ms=3000, speaker="C"),
        judge("new_position", parent="i1", alternative="alternative"),
    )
    assert "B" in t.resolution.supports["p2"]
    assert "i1" not in t.resolution.supports
    assert t.nodes.pending["pending:concern"].target_id == "p2"
    assert t.snapshot()["internal"]["unanswered_concerns"][0]["node_id"] == "p2"
    process(t, turn("cleared", ms=4000, speaker="D"), judge("p2", stance="support"))
    process(t, turn("decision", ms=5000), judge("p2", resolution="decide"))
    t.advance(15000)
    assert t.tree.nodes["i1"].status == "decided"
    assert {s.turn_id for s in t.tree.nodes["i1"].resolution_evidence[0].agreements} == {
        "support",
        "cleared",
    }


def test_later_stable_uid_is_linked_to_previous_cluster_mark():
    t = tracker()
    process(t, turn("uncertain", speaker_key="a", unsure=True), judge(stance="support"))
    process(t, turn("identified", ms=2000, speaker_key="a", speaker_uid="person:a"), judge("root"))
    assert list(t.tree.nodes["i1"].stances) == ["person:a"]
    assert t.tree.nodes["i1"].stances["person:a"].name == "A"


def test_retrospective_correction_deduplicates_waiting_concerns():
    t = tracker()
    process(t, turn("first", speaker="?", unsure=True), judge(stance="concern"))
    process(t, turn("second", ms=2000, speaker="B"), judge(stance="concern"))
    t.identify_speaker("unknown:first", turn("confirmed", ms=3000, speaker="B"))
    group = t.snapshot()["internal"]["unanswered_concerns"][0]
    assert group["count"] == 1
    assert group["concerns"][0]["turn_id"] == "second"


def test_resolution_evidence_schema_preserves_summary_confirmation():
    from das.discussion_structure.models import ResolutionEvidence

    question = turn("question", role="ai", text="紙で決まりでよいですか？")
    response = turn("response", ms=2000, speaker="B", text="はい")
    evidence = ResolutionEvidence(
        status="decided",
        answer="yes",
        target_id="i1",
        decision_turn=response,
        summary_confirmation=(question, response),
    )
    issue = Issue(
        id="i1",
        label="導入するか？",
        answer_type="yes_no",
        parent_id="root",
        resolution_evidence=(evidence,),
    )
    restored = Issue.model_validate_json(issue.model_dump_json())
    assert restored.resolution_evidence[0].summary_confirmation == (question, response)
    assert restored.stances == {}


def test_low_confidence_without_clusters_does_not_establish_independence():
    t = tracker()
    process(t, turn("decision"), judge(resolution="decide"))
    process(
        t, turn("uncertain", ms=2000, speaker="B", speaker_confidence=0.49), judge(agreement="yes")
    )
    t.advance(11000)
    assert t.tree.nodes["i1"].status == "open"

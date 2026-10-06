import json
from itertools import pairwise
from pathlib import Path

import pytest

from das.discussion_structure.config import Config
from das.discussion_structure.display import Display, timeline
from das.discussion_structure.focus import Focus
from das.discussion_structure.judgement import ScriptedBackend
from das.discussion_structure.labels import ScriptedLabels
from das.discussion_structure.metrics import comparison, compute, projection
from das.discussion_structure.models import (
    Distribution,
    Event,
    Issue,
    Judgement,
    Position,
    Tree,
    Turn,
)
from das.discussion_structure.replay import run
from das.discussion_structure.tracker import Tracker
from das.discussion_structure.views import build_view

FIXTURES = Path(__file__).parents[1] / "fixtures/discussion_structure"
SCENARIOS = sorted(FIXTURES.glob("*.json"))


def turn(tid="test", ms=2000, speaker="A", text="実質的な発話をして議論を続けましょう", **extra):
    return Turn(turn_id=tid, speaker=speaker, text=text, ms=ms - 1000, end_ms=ms, **extra)


def scenario_tracker(name):
    data = json.loads((FIXTURES / f"{name}.json").read_text())
    return data, Tracker(
        data["agenda"], ScriptedBackend(data["judgements"]), ScriptedLabels(data["labels"])
    )


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_all_scenarios_exact_tree_and_checkpoints(path, tmp_path):
    tracker = run(
        path.with_suffix(".turns.jsonl"), "学食の環境対策", scenario_path=path, out=tmp_path
    )
    report = json.loads((tmp_path / "scenario_check.json").read_text())
    assert report["mismatches"] == 0, report
    assert {p.name for p in tmp_path.iterdir()} == {
        "events.jsonl",
        "snapshot.json",
        "timeline.txt",
        "metrics.json",
        "scenario_check.json",
    }
    commits = [e for e in tracker.events if e.type == "display_commit"]
    assert all(b.ms - a.ms >= 3000 for a, b in pairwise(commits))
    assert "reason" not in (tmp_path / "timeline.txt").read_text()
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["label_overflow_rate"] == 0
    assert metrics["api_cost_usd"] == 0


def test_upgrade_transfers_full_stances_keeps_children_and_is_atomic():
    data, tracker = scenario_tracker("upgrade_concerns")
    for raw in data["turns"][:5]:
        tracker.process(Turn.model_validate(raw))
    before = tracker.tree.nodes["i1"]
    parents = {i: tracker.tree.nodes[i].parent_id for i in ("i2", "i3")}
    tracker.process(Turn.model_validate(data["turns"][5]))
    assert tracker.tree.nodes["p4"].stances == before.stances
    assert tracker.tree.nodes["i1"].stances == {}
    assert {i: tracker.tree.nodes[i].parent_id for i in parents} == parents
    assert tracker.proposals[0].node_id == "i3"
    assert tracker.proposals[0].parent_id == "p4"
    commits = [e for e in tracker.events if e.type == "display_commit" and e.ms >= 25000]
    assert len(commits) == 1
    rows = {n["id"]: n for n in commits[0].data["state"]["nodes"]}
    assert rows["i1"]["answer_type"] == "choice"
    assert set(rows) >= {"p4", "p5"}
    assert not any(s.speaker_uid == "D" for s in tracker.tree.nodes["p4"].stances.values())


def test_decision_candidates_with_concern_remain_internal_and_objection_cancels():
    data, tracker = scenario_tracker("decisions_reopen")
    for raw in data["turns"][:10]:
        tracker.process(Turn.model_validate(raw))
    assert tracker.resolution.candidates["i1"]["reason"] == "explicit_with_concern"
    assert tracker.tree.nodes["i1"].status == "open"
    assert "decision_candidates" not in tracker.display.current
    tracker.process(Turn.model_validate(data["turns"][10]))
    assert not tracker.resolution.candidates


def test_implicit_convergence_is_candidate_only():
    data, tracker = scenario_tracker("open_stances_focus")
    for raw in data["turns"][:10]:
        tracker.process(Turn.model_validate(raw))
    assert tracker.tree.focus_id == "i4"
    assert tracker.tree.nodes["i1"].status == "open"
    assert tracker.resolution.candidates["i1"]["reason"] == "convergence_on_focus_exit"


def test_failed_upgrade_labels_leave_original_unchanged():
    data, tracker = scenario_tracker("upgrade_concerns")
    for raw in data["turns"][:5]:
        tracker.process(Turn.model_validate(raw))
    before = tracker.tree.snapshot()
    tracker.labels.labels["t6"]["upgrade"] = {"label": "疑問形でない", "answer_type": "choice"}
    tracker.process(Turn.model_validate(data["turns"][5]))
    assert tracker.tree.snapshot() == before
    assert tracker.proposals[-1].operation == "label_review"


def test_pending_expiry_at_exact_60_seconds_and_same_speaker_no_response():
    data, tracker = scenario_tracker("pending_additions")
    tracker.process(Turn.model_validate(data["turns"][0]))
    tracker.process(Turn.model_validate(data["turns"][1]))
    assert "pending:t1" in tracker.nodes.pending
    tracker.advance(61999)
    assert "pending:t1" in tracker.nodes.pending
    tracker.advance(62000)
    assert not tracker.nodes.pending
    assert len(tracker.tree.nodes) == 1


def test_eof_does_not_manufacture_decision_wait():
    data, tracker = scenario_tracker("decisions_reopen")
    for raw in data["turns"][:4]:
        tracker.process(Turn.model_validate(raw))
    tracker.finish()
    assert tracker.tree.nodes["i1"].status == "open"


def test_objection_at_exact_decision_deadline_blocks():
    data, tracker = scenario_tracker("decisions_reopen")
    for raw in data["turns"][:4]:
        tracker.process(Turn.model_validate(raw))
    tracker.backend.judgements["obj"] = Judgement(
        target=Distribution(probabilities={"i1": 1}),
        resolution=Distribution(probabilities={"object": 1}),
    ).model_dump()
    tracker.process(turn("obj", 25000, "C", "その決定に異議があります"))
    assert tracker.tree.nodes["i1"].status == "open"


def test_new_decision_without_other_agreement_never_resolves_after_silence():
    data, tracker = scenario_tracker("decision_guards")
    for raw in data["turns"][:4]:
        tracker.process(Turn.model_validate(raw))
    assert tracker.tree.nodes["i1"].status == "open"


def test_open_addition_choice_switch_and_concern_owner_only():
    data, tracker = scenario_tracker("open_stances_focus")
    for raw in data["turns"][:4]:
        tracker.process(Turn.model_validate(raw))
    assert "A" in tracker.tree.nodes["p2"].stances
    assert "A" in tracker.tree.nodes["p3"].stances
    for raw in data["turns"][4:7]:
        tracker.process(Turn.model_validate(raw))
    assert "A" not in tracker.tree.nodes["p2"].stances
    assert tracker.tree.nodes["p2"].stances["B"].value == "concern"
    tracker.process(Turn.model_validate(data["turns"][7]))
    assert tracker.tree.nodes["p2"].stances["B"].value == "support"


@pytest.mark.parametrize("text", ["はい", "うん", "なるほど", "そうですね。", "", "   "])
def test_backchannels_and_silence_are_skipped(text):
    tracker = Tracker("議題", ScriptedBackend({}), ScriptedLabels({}))
    tracker.process(turn(text=text))
    assert len(tracker.tree.nodes) == 1
    assert not any(e.type == "judgement" for e in tracker.events)


def test_ai_speech_is_excluded_and_unknown_marks_hide_name():
    tracker = Tracker("議題", ScriptedBackend({}), ScriptedLabels({}))
    tracker.process(turn(role="ai_facilitator"))
    assert tracker.events[-1].type == "turn_skipped"
    data, tracker = scenario_tracker("upgrade_concerns")
    for raw in data["turns"]:
        tracker.process(Turn.model_validate(raw))
    assert tracker.tree.nodes["p4"].stances["u1"].name == "?"
    assert "表示してはいけない名前" not in timeline(tracker.events)


def test_focus_hysteresis_short_visit_and_tree_moves():
    data, tracker = scenario_tracker("open_stances_focus")
    for raw in data["turns"][:12]:
        tracker.process(Turn.model_validate(raw))
    assert tracker.tree.focus_id == "i4"
    for raw in data["turns"][12:14]:
        tracker.process(Turn.model_validate(raw))
    assert tracker.tree.focus_id == "i1"
    for raw in data["turns"][14:16]:
        tracker.process(Turn.model_validate(raw))
    assert tracker.tree.focus_id == "i5"
    assert tracker.tree.nodes["i4"].status == "open"


def test_focus_two_utterances_and_ten_seconds_are_both_required():
    tree = Tree("議題")
    tree.add(Issue(id="i1", label="費用は？", answer_type="open", parent_id="root"))
    focus = Focus(Config())
    obs = Distribution(probabilities={"i1": 1})
    assert focus.update(tree, obs, turn(ms=2000), False) is None
    assert focus.update(tree, obs, turn(ms=11999), False) is None
    assert tree.focus_id == "root"
    assert focus.update(tree, obs, turn(ms=12000), False) == "root"
    assert tree.focus_id == "i1"


def test_focus_transition_prior_favors_continuation_and_shift_raises_movement():
    tree = Tree("議題")
    tree.add(Issue(id="i1", label="費用は？", answer_type="open", parent_id="root"))
    obs = Distribution(probabilities={"root": 0.5, "i1": 0.5})
    a, b = Focus(Config()), Focus(Config())
    a.update(tree, obs, turn(), False)
    b.update(tree, obs, turn(), True)
    assert a.belief["root"] > a.belief["i1"]
    assert b.belief["i1"] > a.belief["i1"]


def test_view_excludes_unrelated_resolved_tree_and_bounds_recent_context():
    tree = Tree("議題")
    for index in range(12):
        tree.add(
            Issue(
                id=f"i{index}",
                label=f"問い{index}は？",
                answer_type="open",
                parent_id="root",
                status="decided" if index == 11 else "open",
                last_mentioned_ms=index,
            )
        )
    tree.focus_id = "i0"
    view = build_view(tree, [turn(tid=str(i)) for i in range(20)], turn(), Config())
    assert len(view["recent_turns"]) == 6
    assert len(view["other_issues"]) == 4
    assert "i11" not in {
        n["id"] for group in ("path", "subtree", "other_issues") for n in view[group]
    }
    assert "nodes" not in view


def test_display_depth_counts_issues_not_positions_and_breadcrumb_moves_only_with_focus():
    tree = Tree("議題")
    for index in range(1, 6):
        parent = "root" if index == 1 else f"p{index - 1}"
        tree.add(
            Issue(id=f"i{index}", label=f"問い{index}は？", parent_id=parent, answer_type="open")
        )
        tree.add(Position(id=f"p{index}", label=f"案{index}", parent_id=f"i{index}"))
    tree.focus_id = "i3"
    display = Display(Config())
    state = display.build(tree)
    assert state["breadcrumb"] == ["議題", "問い1は？", "案1"]
    assert {n["id"] for n in state["nodes"]} == {"i2", "p2", "i3", "p3", "i4", "p4"}
    assert display.build(tree)["breadcrumb"] == state["breadcrumb"]
    tree.focus_id = "i4"
    assert display.build(tree)["breadcrumb"] != state["breadcrumb"]


def test_display_cap_preserves_focus_and_relative_sibling_order():
    tree = Tree("議題")
    for index in range(20):
        tree.add(
            Issue(
                id=f"i{index}",
                label=f"問い{index}は？",
                parent_id="root",
                answer_type="open",
                last_mentioned_ms=index,
                status="decided" if index < 5 else "open",
            )
        )
    tree.focus_id = "i19"
    state = Display(Config()).build(tree)
    assert len(state["nodes"]) == 15
    ids = [n["id"] for n in state["nodes"]]
    assert "i19" in ids
    assert ids == [i for i in tree.nodes if i in ids]
    assert state["nodes"][0]["collapsed_count"] == 6


def test_tree_rejects_positions_under_yes_no_and_invalid_turns():
    tree = Tree("議題")
    tree.add(Issue(id="i1", label="導入するか？", answer_type="yes_no", parent_id="root"))
    with pytest.raises(ValueError, match="choice/open"):
        tree.add(Position(id="p2", label="紙", parent_id="i1"))
    with pytest.raises(ValueError, match="end_ms"):
        Turn(turn_id="x", text="t", speaker="A", ms=20, end_ms=10)
    with pytest.raises(ValueError, match="nondecreasing"):
        Tracker("議題", ScriptedBackend({}), ScriptedLabels({})).advance(-1)


def test_label_retry_and_double_failure_produce_proposal():
    data, tracker = scenario_tracker("decision_guards")
    for raw in data["turns"]:
        tracker.process(Turn.model_validate(raw))
    assert "費用をどうするか？" in [n.label for n in tracker.tree.nodes.values()]
    assert len(tracker.proposals) == 1
    assert len(tracker.tree.nodes) == 3


def test_metrics_and_comparison_report_missing_stages_and_probability_bins():
    data, tracker = scenario_tracker("upgrade_concerns")
    for raw in data["turns"]:
        tracker.process(Turn.model_validate(raw))
    metrics = compute(tracker.events)
    assert metrics["display_commits"] > 0
    assert metrics["unapplied_structure_proposals"] == 1
    assert metrics["latency_stages"]["end_to_asr_final"] is None
    report = comparison(tracker.events, data["judgements"])
    assert all(row["accuracy"] == 1 for row in report.values())
    assert report["target"]["calibration"][-1]["accuracy"] == 1
    assert projection(tracker.snapshot()) == data["expected"]
    empty = compute([Event(type="noop", ms=0)])
    assert empty["judgement_latency_ms"]["p50"] is None


def test_legacy_ai_null_time_and_uncertain_speaker_input(tmp_path):
    from das.discussion_structure.replay import read_turns

    path = tmp_path / "input.jsonl"
    rows = [
        {"turn_id": 1, "speaker": "A", "text": "議論しよう", "ms": 10, "end_ms": 50},
        {
            "turn_id": 2,
            "speaker": "ファシリテーター",
            "text": "AIの発話",
            "ms": None,
            "end_ms": None,
        },
        {"turn_id": 3, "speaker": "未確定", "text": "はい", "ms": 60, "end_ms": 90, "bc": True},
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    turns, notices = read_turns(path)
    assert len(turns) == 3 and len(notices) == 1
    assert turns[1].end_ms == 50 and not turns[1].substantive
    assert turns[2].unsure and turns[2].backchannel
    rows[0]["ms"] = None
    path.write_text(json.dumps(rows[0]))
    with pytest.raises(ValueError, match="human turn requires"):
        read_turns(path)


def test_scripted_suite_produces_aggregate_report(tmp_path):
    from das.discussion_structure.compare import run_suite

    summary = run_suite(FIXTURES, tmp_path, backend="scripted")
    assert len(summary["completed"]) == len(SCENARIOS)
    assert summary["failure"] is None
    assert summary["api_cost_usd"] == 0
    assert all(row["accuracy"] == 1 for row in summary["agreement_by_item"].values())


def test_silent_turn_exactly_at_deadline_advances_timer_without_agreement():
    data, tracker = scenario_tracker("decisions_reopen")
    for raw in data["turns"][:4]:
        tracker.process(Turn.model_validate(raw))
    tracker.process(turn("silence", 25000, "C", ""))
    assert tracker.tree.nodes["i1"].status == "decided"
    assert len(tracker.tree.nodes["i1"].stances) == 2


def test_choice_both_preserves_support_and_switch_keeps_concern():
    from das.discussion_structure import stances
    from das.discussion_structure.models import certain

    tree = Tree("議題")
    tree.add(Issue(id="i1", label="何を選ぶか？", answer_type="choice", parent_id="root"))
    for index in range(2, 5):
        tree.add(Position(id=f"p{index}", label=f"案{index}", parent_id="i1"))
    events = []

    def emit(kind, **data):
        events.append((kind, data))

    support = Judgement(stance=certain("support"))
    stances.apply(tree, "p2", turn("a"), support, Config(), emit)
    stances.apply(
        tree,
        "p3",
        turn("b"),
        Judgement(stance=certain("support"), switch=certain("both")),
        Config(),
        emit,
    )
    assert "A" in tree.nodes["p2"].stances and "A" in tree.nodes["p3"].stances
    stances.apply(tree, "p2", turn("c"), Judgement(stance=certain("concern")), Config(), emit)
    stances.apply(tree, "p4", turn("d"), support, Config(), emit)
    assert tree.nodes["p2"].stances["A"].value == "concern"
    assert "A" not in tree.nodes["p3"].stances
    assert "A" in tree.nodes["p4"].stances


def test_config_file_matches_packaged_defaults():
    assert Config.load(Path("configs/discussion_structure.toml")) == Config()

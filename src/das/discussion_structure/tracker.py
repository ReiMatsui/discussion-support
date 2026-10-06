"""Replay orchestration using transcript time; monotonic clocks only measure API latency."""

from dataclasses import replace
from time import perf_counter

from .config import Config
from .display import Display
from .focus import Focus
from .models import Event, Issue, StructureProposal, Tree, Turn
from .nodes import Nodes
from .resolution import Resolution
from .views import build_view


class Tracker:
    def __init__(self, agenda, backend, labels, config: Config | None = None):
        self.config = config or Config()
        self.tree = Tree(agenda)
        self.backend, self.labels = backend, labels
        self.events: list[Event] = []
        self.proposals: list[StructureProposal] = []
        self.recent: list[Turn] = []
        self.archived_agendas: list[dict] = []
        self.now = 0
        self.turn_id = None
        self.focus = Focus(self.config)
        self.display = Display(self.config)
        self.resolution = Resolution(self.tree, self.config, self.emit)
        self.nodes = Nodes(self.tree, self.config, labels, self.emit, self.propose)
        self.display.commit(self.tree, 0, self.emit)

    def emit(self, event_type, **data):
        self.events.append(Event(type=event_type, ms=self.now, turn_id=self.turn_id, data=data))

    def propose(self, proposal):
        self.proposals.append(proposal)
        self.emit("structure_proposal", proposal=proposal.model_dump())

    def advance(self, ms, *, inclusive=True):
        if ms < self.now:
            raise ValueError("transcript end_ms must be nondecreasing")
        boundary = ms if inclusive else ms - 1
        # Explicit timer events avoid delaying a display/decision until the next utterance.
        while True:
            deadlines = []
            if self.display.build(self.tree) != self.display.current:
                deadlines.append(
                    max(self.now, (self.display.last_ms or 0) + self.config.display_ms)
                )
            for d in self.resolution.pending.values():
                due = d.since + self.config.decision_ms
                if due > self.now:
                    deadlines.append(due)
            for p in self.nodes.pending.values():
                if p.kind in {"implicit", "reopen"}:
                    deadlines.append(p.turn.end_ms + self.config.pending_ms)
            next_ms = min(deadlines, default=boundary + 1)
            if next_ms > boundary:
                break
            self.now = next_ms
            self.turn_id = None
            self.nodes.expire(self.now)
            self.resolution.advance(self.now)
            self.nodes.clean_concerns()
            self.display.commit(self.tree, self.now, self.emit)
        self.now = ms

    def process(self, turn: Turn):
        self.advance(turn.end_ms, inclusive=False)
        self.turn_id = turn.turn_id
        self.nodes.expire(self.now)
        if turn.speaker_key is not None and turn.uid != f"cluster:{turn.speaker_key}":
            self.identify_speaker(f"cluster:{turn.speaker_key}", turn)
        self.identify_speaker(turn.uid, turn)
        if not turn.substantive:
            self.emit("turn_skipped", reason="AI/backchannel/silence")
        else:
            view = build_view(self.tree, self.recent, turn, self.config, self.nodes.pending_view())
            start = perf_counter()
            judgement = self.backend.judge(turn, view)
            elapsed = (perf_counter() - start) * 1000
            self.emit(
                "judgement",
                judgement=judgement.model_dump(),
                latency_ms=elapsed,
                origin_end_ms=turn.end_ms,
            )
            start = perf_counter()
            event_start = len(self.events)
            target = self.nodes.process(turn, judgement, self.recent)
            for event in self.events[event_start:]:
                if event.type == "issue_upgraded":
                    self.resolution.upgraded(event.data["node_id"], event.data["original_id"])
            self.emit(
                "rules_completed",
                latency_ms=(perf_counter() - start) * 1000,
                target_id=target,
                origin_end_ms=turn.end_ms,
            )
            if judgement.correction.choice == "merge":
                self.propose(
                    StructureProposal(
                        operation="merge", node_id=target, reason="participant requested merge"
                    )
                )
            if judgement.relevance.choice != "off":
                self.resolution.observe(target)
                observation = self.nodes.observation(judgement, target)
                old = self.focus.update(
                    self.tree, observation, turn, judgement.shift.choice == "yes"
                )
                self.emit("focus_belief", probabilities=self.focus.belief)
                if old:
                    self.emit(
                        "focus_transition",
                        previous=old,
                        current=self.tree.focus_id,
                        origin_end_ms=self.focus.since,
                    )
                    self.resolution.left_focus(old)
                    self.resolution.entered_focus()
                self.resolution.process(target, turn, judgement)
        self.resolution.advance(self.now)
        self.nodes.clean_concerns()
        self.recent.append(turn)
        self.display.commit(self.tree, self.now, self.emit)

    def identify_speaker(self, previous_uid: str, confirmed: Turn):
        """Apply a later identification (or an explicit retrospective correction).

        When marks collide, keep the most recent source turn. This changes identity,
        not the stance expressed by another participant.
        """
        if not confirmed.identified(self.config.name_threshold):
            return
        changed_issues = set()
        sources = {t.turn_id: t.end_ms for t in self.recent}
        for node in list(self.tree.nodes.values()):
            stances = dict(node.stances)
            old = stances.pop(previous_uid, None)
            if old:
                updated = old.model_copy(
                    update={
                        "speaker_uid": confirmed.uid,
                        "name": confirmed.speaker,
                        "speaker_confidence": confirmed.speaker_confidence,
                    }
                )
                collision = stances.get(confirmed.uid)
                if collision and max(sources.get(t, -1) for t in collision.source_turns) > max(
                    sources.get(t, -1) for t in old.source_turns
                ):
                    updated = collision
                stances[confirmed.uid] = updated
                if stances != node.stances:
                    changed_issues.add(self.tree.issue_id(node.id))
                    self.tree.replace(node.id, stances=stances)
                    self.emit(
                        "stance_identity",
                        node_id=node.id,
                        previous_uid=previous_uid,
                        speaker_uid=confirmed.uid,
                        name=confirmed.speaker,
                    )

        def corrected(source):
            if source.uid != previous_uid:
                return source
            return source.model_copy(
                update={
                    "speaker": confirmed.speaker,
                    "speaker_uid": confirmed.uid,
                    "speaker_key": confirmed.speaker_key
                    if confirmed.speaker_key is not None
                    else source.speaker_key,
                    "speaker_confidence": confirmed.speaker_confidence,
                    "unsure": False,
                }
            )

        self.recent = [corrected(t) for t in self.recent]
        for key, candidate in list(self.nodes.pending.items()):
            self.nodes.pending[key] = replace(candidate, turn=corrected(candidate.turn))
        for decision in self.resolution.pending.values():
            decision.turn = corrected(decision.turn)
            consent = [corrected(source) for source in decision.explicit_agreements.values()]
            decision.explicit_agreements = {t.uid: t for t in consent}
            decision.agreements = {
                t.uid: t
                for source in decision.agreements.values()
                if (t := corrected(source)).uid != decision.turn.uid
            }
        for target, supports in self.resolution.supports.items():
            corrected_supports = [corrected(source) for source in supports.values()]
            self.resolution.supports[target] = {t.uid: t for t in corrected_supports}
        for node in list(self.tree.nodes.values()):
            if isinstance(node, Issue) and node.resolution_evidence:
                evidence = tuple(
                    e.model_copy(
                        update={
                            "decision_turn": corrected(e.decision_turn),
                            "agreements": tuple(corrected(t) for t in e.agreements),
                            "explicit_agreements": tuple(
                                corrected(t) for t in e.explicit_agreements
                            ),
                            "summary_confirmation": tuple(
                                corrected(t) for t in e.summary_confirmation
                            ),
                        }
                    )
                    for e in node.resolution_evidence
                )
                if evidence != node.resolution_evidence:
                    changed_issues.add(node.id)
                    self.tree.replace(node.id, resolution_evidence=evidence)

        if changed_issues:
            self.resolution.revalidate(changed_issues)
        self.nodes.clean_concerns()

    def switch_agenda(self, agenda: str):
        """Archive the old agenda and start a new tree; no replay/UI wiring."""
        for key, candidate in list(self.nodes.pending.items()):
            if candidate.kind == "concern":
                del self.nodes.pending[key]
                self.emit("concern_resolved", candidate_id=key, reason="agenda_changed")
        self.archived_agendas.append(self.tree.snapshot())
        self.tree = Tree(agenda)
        self.focus = Focus(self.config)
        self.display = Display(self.config)
        self.resolution = Resolution(self.tree, self.config, self.emit)
        self.nodes = Nodes(self.tree, self.config, self.labels, self.emit, self.propose)
        self.recent.clear()
        self.proposals.clear()
        self.emit("agenda_transition", agenda=agenda)
        self.display.commit(self.tree, self.now, self.emit)

    def finish(self):
        # Flush display only. Never manufacture ten seconds of observed discussion
        # after EOF to decide a still-pending explicit decision.
        if self.display.build(self.tree) != self.display.current:
            self.now = max(self.now, self.display.last_ms + self.config.display_ms)
            self.turn_id = None
            self.display.commit(self.tree, self.now, self.emit)

    def snapshot(self):
        return {
            "internal": self.tree.snapshot()
            | {
                "pending": self.nodes.pending_view(),
                "unanswered_concerns": self.nodes.unanswered_concerns(),
                "focus_belief": self.focus.belief,
                "decision_candidates": list(self.resolution.candidates.values()),
                "decision_confirmations": [
                    {"issue_id": n.id, "reasons": list(n.confirmation_reasons)}
                    for n in self.tree.nodes.values()
                    if isinstance(n, Issue) and n.status == "decided" and n.needs_confirmation
                ],
                "decision_pending": [
                    {
                        "issue_id": d.issue_id,
                        "target_id": d.target_id,
                        "since": d.since,
                        "answer": d.answer,
                        "decision_turn": d.turn.model_dump(mode="json"),
                        "agreement_turns": [
                            t.model_dump(mode="json") for t in d.agreements.values()
                        ],
                        "agreements": sorted(d.agreements),
                        "objected": d.objected,
                    }
                    for d in self.resolution.pending.values()
                ],
                "structure_proposals": [p.model_dump() for p in self.proposals],
            },
            "display": self.display.current,
        }

"""Replay orchestration using transcript time; monotonic clocks only measure API latency."""

from time import perf_counter

from .config import Config
from .display import Display
from .focus import Focus
from .models import Event, StructureProposal, Tree, Turn
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
                if p.kind == "implicit":
                    deadlines.append(p.turn.end_ms + self.config.pending_ms)
            next_ms = min(deadlines, default=boundary + 1)
            if next_ms > boundary:
                break
            self.now = next_ms
            self.turn_id = None
            self.nodes.expire(self.now)
            self.resolution.advance(self.now)
            self.display.commit(self.tree, self.now, self.emit)
        self.now = ms

    def process(self, turn: Turn):
        self.advance(turn.end_ms, inclusive=False)
        self.turn_id = turn.turn_id
        self.nodes.expire(self.now)
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
            target = self.nodes.process(turn, judgement, self.recent)
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
                self.resolution.process(target, turn, judgement)
        self.resolution.advance(self.now)
        self.recent.append(turn)
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
                "focus_belief": self.focus.belief,
                "decision_candidates": list(self.resolution.candidates.values()),
                "decision_pending": [
                    {
                        "issue_id": d.issue_id,
                        "target_id": d.target_id,
                        "since": d.since,
                        "agreements": sorted(d.agreements),
                        "objected": d.objected,
                    }
                    for d in self.resolution.pending.values()
                ],
                "structure_proposals": [p.model_dump() for p in self.proposals],
            },
            "display": self.display.current,
        }

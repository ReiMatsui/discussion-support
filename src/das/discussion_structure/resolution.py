"""Explicit decisions need independent agreement and an objection-free window."""

from dataclasses import dataclass, field

from .config import Config
from .models import Issue, Judgement, Position, Tree, Turn


@dataclass
class Decision:
    issue_id: str
    target_id: str
    speaker_uid: str
    since: int
    answer: str
    agreements: set[str] = field(default_factory=set)
    objected: bool = False


class Resolution:
    def __init__(self, tree: Tree, config: Config, emit):
        self.tree, self.config, self.emit = tree, config, emit
        self.pending: dict[str, Decision] = {}
        self.candidates: dict[str, dict] = {}

    def process(self, target: str | None, turn: Turn, judgement: Judgement):
        if target not in self.tree.nodes:
            return
        issue_id = self.tree.issue_id(target)
        issue = self.tree.nodes[issue_id]
        action = judgement.resolution.choice
        clear = judgement.resolution.probability >= self.config.resolution_threshold
        if clear and action in {"hold", "withdraw"}:
            status = "held" if action == "hold" else "withdrawn"
            self.tree.replace(issue_id, status=status, decided_answer=None)
            self.pending.pop(issue_id, None)
            self.candidates.pop(issue_id, None)
            self.emit("status_transition", node_id=issue_id, previous=issue.status, current=status)
            return
        if clear and action == "decide":
            if issue.answer_type != "yes_no" and not isinstance(self.tree.nodes[target], Position):
                return  # No adopted answer can be inferred from the Issue alone.
            previous = self.pending.get(issue_id)
            if previous and previous.target_id == target and previous.speaker_uid != turn.uid:
                previous.agreements.add(turn.uid)
            else:
                self.pending[issue_id] = Decision(
                    issue_id, target, turn.uid, turn.end_ms, judgement.decision_answer.choice
                )
                self.emit("decision_waiting", node_id=issue_id, target_id=target)
        decision = self.pending.get(issue_id)
        if decision:
            if clear and action == "object":
                decision.objected = True
                self.candidates.pop(issue_id, None)
                self.emit("decision_objected", node_id=issue_id)
            if (
                target == decision.target_id
                and judgement.stance.choice == "support"
                and judgement.stance.probability >= self.config.stance_threshold
                and turn.uid != decision.speaker_uid
            ):
                decision.agreements.add(turn.uid)

    def advance(self, ms: int):
        for issue_id, decision in list(self.pending.items()):
            issue = self.tree.nodes[issue_id]
            if issue.status != "open":
                del self.pending[issue_id]
                continue
            target = self.tree.nodes[decision.target_id]
            concerns = any(s.value == "concern" for s in target.stances.values())
            if concerns and not decision.objected:
                self.candidate(issue_id, decision.target_id, "explicit_with_concern")
            if ms < decision.since + self.config.decision_ms:
                continue
            if decision.objected or not decision.agreements or concerns:
                continue
            if isinstance(target, Issue) and target.answer_type != "yes_no":
                del self.pending[issue_id]
                continue
            self.tree.replace(
                issue_id,
                status="decided",
                decided_answer=decision.answer if issue.answer_type == "yes_no" else None,
            )
            if isinstance(target, Position):
                for node in self.tree.children(issue_id):
                    if isinstance(node, Position):
                        self.tree.replace(
                            node.id, status="adopted" if node.id == target.id else "rejected"
                        )
            self.emit(
                "status_transition",
                node_id=issue_id,
                previous="open",
                current="decided",
                target_id=target.id,
                decided_answer=decision.answer,
                origin_end_ms=decision.since,
            )
            self.candidates.pop(issue_id, None)
            del self.pending[issue_id]

    def candidate(self, issue_id, target_id, reason):
        candidate = {"issue_id": issue_id, "target_id": target_id, "reason": reason}
        if self.candidates.get(issue_id) != candidate:
            self.candidates[issue_id] = candidate
            self.emit("decision_candidate", **candidate)

    def left_focus(self, issue_id):
        issue = self.tree.nodes[issue_id]
        if issue.status != "open":
            return
        targets = [issue] if issue.answer_type == "yes_no" else self.tree.children(issue_id)
        for node in targets:
            if node.stances and all(s.value == "support" for s in node.stances.values()):
                self.candidate(issue_id, node.id, "convergence_on_focus_exit")

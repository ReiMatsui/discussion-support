"""Explicit decisions keep agreement evidence separate from proposal stances."""

from dataclasses import dataclass, field

from .config import Config
from .models import Judgement, Position, ResolutionEvidence, Tree, Turn


@dataclass
class Decision:
    issue_id: str
    target_id: str
    turn: Turn
    answer: str
    flow: int
    agreements: dict[str, Turn] = field(default_factory=dict)
    explicit_agreements: dict[str, Turn] = field(default_factory=dict)
    objected: bool = False

    @property
    def since(self):
        return self.turn.end_ms


class Resolution:
    def __init__(self, tree: Tree, config: Config, emit):
        self.tree, self.config, self.emit = tree, config, emit
        self.pending: dict[str, Decision] = {}
        self.candidates: dict[str, dict] = {}
        self.flow = 0
        self.run_issue = None
        self.supports: dict[str, dict[str, Turn]] = {}

    def entered_focus(self):
        # Pending decisions distinguish display focus visits; observe scopes acceptances.
        self.flow += 1

    def observe(self, target):
        # Display hysteresis does not define the start of a conversation run.
        issue_id = self.tree.issue_id(target) if target in self.tree.nodes else None
        if issue_id != self.run_issue:
            self.supports.clear()
            self.run_issue = issue_id

    def blockers(self, target_id, answer, since, explicit_agreements):
        contrary = "concern" if answer == "yes" else "support"
        consenting = {t.uid for t in explicit_agreements if t.end_ms > since}
        return [
            uid
            for uid, stance in self.tree.nodes[target_id].stances.items()
            if stance.value == contrary and uid not in consenting
        ]

    def revalidate(self, issue_ids):
        for issue in list(self.tree.nodes.values()):
            if (
                issue.id not in issue_ids
                or issue.status != "decided"
                or not issue.resolution_evidence
            ):
                continue
            evidence = issue.resolution_evidence[-1]
            if evidence.status != "decided" or evidence.summary_confirmation:
                continue
            reasons = []
            if not any(
                t.different_person(evidence.decision_turn, self.config.name_threshold)
                for t in evidence.agreements
            ):
                reasons.append("no_independent_agreement")
            if self.blockers(
                evidence.target_id,
                evidence.answer or "yes",
                evidence.decision_turn.end_ms,
                evidence.explicit_agreements,
            ):
                reasons.append("contrary_stance_without_agreement")
            if reasons and not issue.needs_confirmation:
                self.emit("decision_needs_confirmation", issue_id=issue.id, reasons=reasons)
            # A flag is sticky until a summary confirmation, even if later evidence recovers.
            if reasons:
                self.tree.replace(
                    issue.id,
                    needs_confirmation=True,
                    confirmation_reasons=tuple(
                        dict.fromkeys((*issue.confirmation_reasons, *reasons))
                    ),
                )

    def upgraded(self, issue_id, original_id):
        if issue_id in self.supports:
            self.supports[original_id] = self.supports.pop(issue_id)

    def process(self, target: str | None, turn: Turn, judgement: Judgement):
        if target not in self.tree.nodes:
            return
        issue_id = self.tree.issue_id(target)
        issue = self.tree.nodes[issue_id]
        action = judgement.resolution.choice
        clear = judgement.resolution.probability >= self.config.resolution_threshold
        if clear and action in {"hold", "withdraw"}:
            status = "held" if action == "hold" else "withdrawn"
            evidence = ResolutionEvidence(status=status, target_id=target, decision_turn=turn)
            self.tree.replace(
                issue_id,
                status=status,
                decided_answer=None,
                resolution_evidence=(*issue.resolution_evidence, evidence),
            )
            self.pending.pop(issue_id, None)
            self.candidates.pop(issue_id, None)
            self.emit(
                "status_transition",
                node_id=issue_id,
                previous=issue.status,
                current=status,
                resolution_evidence=evidence.model_dump(mode="json"),
            )
            return
        if issue.status != "open":
            return
        if clear and action == "decide":
            if issue.answer_type != "yes_no" and not isinstance(self.tree.nodes[target], Position):
                return
            answer = judgement.decision_answer.choice if issue.answer_type == "yes_no" else "yes"
            previous = self.pending.get(issue_id)
            if (
                previous
                and previous.flow == self.flow
                and not previous.objected
                and previous.target_id == target
                and previous.answer == answer
                and turn.different_person(previous.turn, self.config.name_threshold)
            ):
                previous.agreements[turn.uid] = turn
            else:
                decision = Decision(issue_id, target, turn, answer, self.flow)
                if answer == "yes" and issue_id == self.tree.focus_id:
                    decision.agreements = {
                        uid: source
                        for uid, source in self.supports.get(target, {}).items()
                        if source.different_person(turn, self.config.name_threshold)
                        and (stance := self.tree.nodes[target].stances.get(uid))
                        and stance.value == "support"
                    }
                self.pending[issue_id] = decision
                self.emit("decision_waiting", node_id=issue_id, target_id=target)
        decision = self.pending.get(issue_id)
        if decision:
            if clear and action == "object":
                decision.objected = True
                self.candidates.pop(issue_id, None)
                self.emit("decision_objected", node_id=issue_id)
            explicit_agreement = (
                judgement.agreement.choice == "yes"
                and judgement.agreement.probability >= self.config.stance_threshold
            )
            if target == decision.target_id and turn.end_ms > decision.since and explicit_agreement:
                decision.explicit_agreements[turn.uid] = turn
            if (
                target == decision.target_id
                and turn.end_ms > decision.since
                and explicit_agreement
                and turn.different_person(decision.turn, self.config.name_threshold)
            ):
                decision.agreements[turn.uid] = turn
        if issue_id == self.run_issue:
            # Includes automatic proposer support, but never guesses another person's stance.
            for node in [issue, *self.tree.children(issue_id)]:
                stance = self.tree.nodes[node.id].stances.get(turn.uid)
                if stance and stance.value == "support" and turn.turn_id in stance.source_turns:
                    self.supports.setdefault(node.id, {})[turn.uid] = turn
                elif not stance or stance.value != "support":
                    self.supports.get(node.id, {}).pop(turn.uid, None)

    def advance(self, ms: int):
        for issue_id, decision in list(self.pending.items()):
            issue = self.tree.nodes[issue_id]
            if issue.status != "open":
                del self.pending[issue_id]
                continue
            target = self.tree.nodes[decision.target_id]
            concerns = self.blockers(
                target.id, decision.answer, decision.since, decision.explicit_agreements.values()
            )
            reason = (
                "explicit_with_concern" if decision.answer == "yes" else "explicit_with_support"
            )
            if concerns and not decision.objected:
                self.candidate(issue_id, decision.target_id, reason)
            if ms < decision.since + self.config.decision_ms:
                continue
            if decision.objected:
                continue
            independent = any(
                t.different_person(decision.turn, self.config.name_threshold)
                for t in decision.agreements.values()
            )
            if not independent or concerns:
                self.candidate(
                    issue_id,
                    decision.target_id,
                    reason if concerns else "explicit_without_agreement",
                )
                continue
            evidence = ResolutionEvidence(
                status="decided",
                answer=decision.answer if issue.answer_type == "yes_no" else None,
                target_id=target.id,
                decision_turn=decision.turn,
                agreements=tuple(decision.agreements.values()),
                explicit_agreements=tuple(decision.explicit_agreements.values()),
            )
            self.tree.replace(
                issue_id,
                status="decided",
                decided_answer=decision.answer if issue.answer_type == "yes_no" else None,
                resolution_evidence=(*issue.resolution_evidence, evidence),
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
                resolution_evidence=evidence.model_dump(mode="json"),
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

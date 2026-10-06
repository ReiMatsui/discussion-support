"""Creation, identity, pending responses and atomic yes/no upgrades."""

from dataclasses import dataclass
from time import perf_counter

from . import stances
from .config import Config
from .labels import Label, checked, valid
from .models import Distribution, Issue, Judgement, Position, StructureProposal, Tree, Turn
from .views import build_view


@dataclass(frozen=True)
class Pending:
    id: str
    kind: str
    turn: Turn
    judgement: Judgement
    parent_id: str
    target_id: str | None = None


class Nodes:
    def __init__(self, tree: Tree, config: Config, generator, emit, propose):
        self.tree, self.config, self.generator = tree, config, generator
        self.emit, self.propose = emit, propose
        self.pending: dict[str, Pending] = {}
        self.concern_children: dict[str, str] = {}
        self.counter = 0

    def pending_view(self):
        return {
            key: {
                "kind": p.kind,
                "text": p.turn.text,
                "speaker_uid": p.turn.uid,
                "parent_id": p.parent_id,
                "target_id": p.target_id,
            }
            for key, p in self.pending.items()
        }

    def expire(self, ms):
        for key, p in list(self.pending.items()):
            if p.kind == "implicit" and ms - p.turn.end_ms >= self.config.pending_ms:
                del self.pending[key]
                self.emit("candidate_expired", candidate_id=key)

    def add_pending(self, kind, turn, judgement, parent_id, target_id=None):
        key = f"pending:{turn.turn_id}"
        self.pending[key] = Pending(key, kind, turn, judgement, parent_id, target_id)
        self.emit("candidate_waiting", candidate_id=key, kind=kind, parent_id=parent_id)

    def label(self, purpose, source, parent_id, recent):
        context = {
            "source_turns": [t.model_dump(mode="json") for t in source],
            "view": build_view(self.tree, recent, source[-1], self.config),
            "parent_path": [{"id": n.id, "label": n.label} for n in self.tree.path(parent_id)],
            "siblings": [n.label for n in self.tree.children(parent_id)],
        }
        start = perf_counter()
        result = checked(
            self.generator,
            purpose,
            context,
            self.config,
            list({t.speaker for t in recent + source if not t.unsure}),
        )
        self.emit(
            "label_generation",
            purpose=purpose,
            latency_ms=(perf_counter() - start) * 1000,
            success=result is not None,
            origin_end_ms=source[-1].end_ms,
        )
        if result is None:
            self.propose(StructureProposal(operation="label_review", reason="label failed twice"))
        return result

    def new_id(self, prefix):
        self.counter += 1
        return f"{prefix}{self.counter}"

    def create(self, kind, label, parent_id, source):
        # Semantic identity is judged by fast layer; exact labels are a final guard.
        for sibling in self.tree.children(parent_id):
            if sibling.kind == kind and sibling.label == label.label:
                self.tree.replace(
                    sibling.id,
                    source_turns=tuple(
                        dict.fromkeys((*sibling.source_turns, *(t.turn_id for t in source)))
                    ),
                )
                self.emit("identity_attached", node_id=sibling.id)
                return sibling.id
        node_id = self.new_id("i" if kind == "issue" else "p")
        common = {
            "id": node_id,
            "label": label.label,
            "parent_id": parent_id,
            "source_turns": tuple(t.turn_id for t in source),
            "last_mentioned_ms": source[-1].end_ms,
        }
        node = (
            Issue(
                **common, answer_type=label.answer_type, original_position=label.original_position
            )
            if kind == "issue"
            else Position(**common)
        )
        self.tree.add(node)
        self.emit(
            "node_created", node=node.model_dump(mode="json"), origin_end_ms=source[-1].end_ms
        )
        return node_id

    def parent(self, judgement):
        scores = {}
        focus_path = {n.id for n in self.tree.path(self.tree.focus_id)}
        for key, p in judgement.parent.probabilities.items():
            key = self.tree.focus_id if key == "focus" else key
            if key in self.tree.nodes:
                prior = (
                    1
                    if key == self.tree.focus_id
                    else (
                        0.8
                        if self.tree.nodes[key].parent_id == self.tree.focus_id
                        else 0.6
                        if key in focus_path
                        else 0.3
                    )
                )
                scores[key] = scores.get(key, 0) + p * prior
        return max(scores, key=scores.get) if scores else self.tree.focus_id

    def target(self, judgement):
        probs = judgement.target.probabilities
        best = judgement.target.choice
        existing = {k: v for k, v in probs.items() if k in self.tree.nodes}
        if existing:
            attached = max(existing, key=existing.get)
            if probs[best] - existing[attached] <= self.config.identity_margin:
                return attached
        return best

    def upgrade(self, issue_id, turn, judgement, recent):
        issue = self.tree.nodes[issue_id]
        if not isinstance(issue, Issue) or issue.answer_type != "yes_no":
            return None
        upgraded = self.label("upgrade", [turn], issue_id, recent)
        new = self.label("position", [turn], issue_id, recent)
        original = Label(label=issue.original_position)
        if not upgraded or not new:
            return None
        # Validate before ANY mutation, so a failed label leaves the old tree intact.
        original_id, new_id = self.new_id("p"), self.new_id("p")
        previous_stances = issue.stances
        self.tree.replace(
            issue_id, answer_type="choice", label=upgraded.label, stances={}, decided_answer=None
        )
        self.tree.add(
            Position(
                id=original_id,
                label=original.label,
                parent_id=issue_id,
                stances=previous_stances,
                source_turns=issue.source_turns,
                reasons=tuple(
                    {"text": s.reason, "source_turns": s.source_turns}
                    for s in previous_stances.values()
                    if s.reason
                ),
            )
        )
        self.tree.add(
            Position(
                id=new_id,
                label=new.label,
                parent_id=issue_id,
                source_turns=(turn.turn_id,),
                last_mentioned_ms=turn.end_ms,
            )
        )
        for node_id in (original_id, new_id):
            self.emit(
                "node_created",
                node=self.tree.nodes[node_id].model_dump(mode="json"),
                origin_end_ms=turn.end_ms,
            )
        for uid, stance in previous_stances.items():
            self.emit(
                "stance_transfer",
                from_id=issue_id,
                to_id=original_id,
                speaker_uid=uid,
                stance=stance.model_dump(),
            )
        stances.apply(self.tree, new_id, turn, judgement, self.config, self.emit, proposer=True)
        # Comparison alone never creates a concern on the original proposal.
        if judgement.stance.choice == "concern":
            stances.apply(self.tree, original_id, turn, judgement, self.config, self.emit)
        for child in self.tree.children(issue_id):
            if self.concern_children.get(child.id) == issue_id:
                self.propose(
                    StructureProposal(
                        operation="reparent",
                        node_id=child.id,
                        parent_id=original_id,
                        reason="original proposal concern",
                    )
                )
        self.emit(
            "issue_upgraded",
            node_id=issue_id,
            old_label=issue.label,
            new_label=upgraded.label,
            original_id=original_id,
            new_id=new_id,
        )
        return new_id

    def process(self, turn: Turn, judgement: Judgement, recent) -> str | None:
        if judgement.relevance.choice == "off":
            return None
        parent_id = self.parent(judgement)
        target = self.target(judgement)
        response = self.pending.get(judgement.response_to.choice)
        if response and response.turn.uid != turn.uid:
            del self.pending[response.id]
            if response.kind == "reopen":
                issue_id = self.tree.issue_id(response.target_id)
                self.tree.replace(issue_id, status="open", decided_answer=None)
                for child in self.tree.children(issue_id):
                    if isinstance(child, Position):
                        self.tree.replace(child.id, status="proposed")
                self.emit("status_transition", node_id=issue_id, previous="decided", current="open")
                target = response.target_id
                if response.judgement.target.choice == "new_position":
                    target = (
                        self.upgrade(issue_id, response.turn, response.judgement, recent) or target
                    )
            else:
                source = [response.turn, turn]
                label = self.label("issue", source, response.parent_id, recent)
                if not label:
                    return None
                required = "open" if response.judgement.trigger.choice == "problem" else "yes_no"
                if response.kind == "implicit" and response.judgement.trigger.choice != "fact":
                    label = label.model_copy(update={"answer_type": required})
                    if not valid(
                        label,
                        "issue",
                        self.config,
                        list({t.speaker for t in recent + source if not t.unsure}),
                    ):
                        self.propose(
                            StructureProposal(
                                operation="label_review", reason="missing original proposal"
                            )
                        )
                        return None
                target = self.create("issue", label, response.parent_id, source)
                if response.kind == "concern":
                    self.concern_children[target] = response.target_id
                if response.judgement.trigger.choice == "proposal":
                    stances.apply(
                        self.tree,
                        target,
                        response.turn,
                        response.judgement,
                        self.config,
                        self.emit,
                        proposer=True,
                    )
                stances.apply(self.tree, target, turn, judgement, self.config, self.emit)
                if (
                    judgement.stance.choice == "concern"
                    and judgement.stance.probability >= self.config.stance_threshold
                ):
                    self.add_pending("concern", turn, judgement, target, target)
                return target
        if target in self.tree.nodes:
            node = self.tree.nodes[target]
            self.tree.replace(
                target,
                source_turns=tuple(dict.fromkeys((*node.source_turns, turn.turn_id))),
                last_mentioned_ms=turn.end_ms,
            )
            if (
                judgement.resolution.choice == "reopen"
                and judgement.resolution.probability >= self.config.resolution_threshold
                and self.tree.nodes[self.tree.issue_id(target)].status == "decided"
            ):
                self.add_pending("reopen", turn, judgement, parent_id, target)
            stances.apply(self.tree, target, turn, judgement, self.config, self.emit)
            current_stance = self.tree.nodes[target].stances.get(turn.uid)
            if current_stance and current_stance.value == "support":
                for key, candidate in list(self.pending.items()):
                    if (
                        candidate.kind == "concern"
                        and candidate.target_id == target
                        and candidate.turn.uid == turn.uid
                    ):
                        del self.pending[key]
                        self.emit("concern_resolved", candidate_id=key, node_id=target)
            if (
                judgement.stance.choice == "concern"
                and judgement.stance.probability >= self.config.stance_threshold
            ):
                self.add_pending("concern", turn, judgement, target, target)
            return target
        if target not in {"new_issue", "new_position"}:
            return None
        trigger = judgement.trigger.choice
        parent = self.tree.nodes[parent_id]
        if target == "new_issue" and trigger == "explicit":
            label = self.label("issue", [turn], parent_id, recent)
            return self.create("issue", label, parent_id, [turn]) if label else None
        if target == "new_position" and isinstance(parent, Issue) and parent.id != "root":
            if parent.answer_type == "yes_no":
                if (
                    judgement.alternative.choice == "alternative"
                    and parent.id == self.tree.focus_id
                ):
                    if parent.status == "decided":
                        self.add_pending("reopen", turn, judgement, parent_id, parent_id)
                        return None
                    return self.upgrade(parent_id, turn, judgement, recent)
            else:
                label = self.label("position", [turn], parent_id, recent)
                if label:
                    node_id = self.create("position", label, parent_id, [turn])
                    stances.apply(
                        self.tree, node_id, turn, judgement, self.config, self.emit, proposer=True
                    )
                    return node_id
        if trigger in {"proposal", "problem", "fact"}:
            self.add_pending("implicit", turn, judgement, parent_id)
        return None

    def observation(self, judgement, target):
        if target is None:
            return judgement.target
        values = dict(judgement.target.probabilities)
        if (
            judgement.target.choice in {"new_issue", "new_position"}
            or judgement.response_to.choice != "none"
        ):
            mass = values.pop(judgement.target.choice)
            values[target] = values.get(target, 0) + mass
        return Distribution(probabilities=values)

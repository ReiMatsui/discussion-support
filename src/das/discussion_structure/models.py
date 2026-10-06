"""Validated immutable records; the tracker replaces records on every change."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Turn(Record):
    model_config = ConfigDict(frozen=True, extra="ignore", coerce_numbers_to_str=True)
    turn_id: str
    speaker: str
    text: str
    ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    speaker_confidence: float | None = Field(default=None, ge=0, le=1)
    unsure: bool = False
    speaker_uid: str | None = None
    speaker_key: str | None = None
    role: str = "human"
    backchannel: bool = False

    @model_validator(mode="after")
    def interval(self):
        if self.end_ms < self.ms:
            raise ValueError("end_ms must be >= ms")
        return self

    @property
    def uid(self) -> str:
        return self.speaker_uid or (
            f"cluster:{self.speaker_key}"
            if self.speaker_key is not None
            else f"unknown:{self.turn_id}"
            if self.unsure
            else self.speaker
        )

    def identified(self, threshold: float) -> bool:
        return not self.unsure and (
            self.speaker_confidence is None or self.speaker_confidence >= threshold
        )

    def different_person(self, other: Turn, threshold: float) -> bool:
        if self.uid == other.uid:
            return False
        if self.speaker_key is not None and other.speaker_key is not None:
            return self.speaker_key != other.speaker_key
        return self.identified(threshold) and other.identified(threshold)

    @property
    def substantive(self) -> bool:
        return (
            self.role == "human"
            and not self.backchannel
            and self.text.strip(" 。、!?？！\n")
            not in {"", "うん", "はい", "なるほど", "そうですね"}
        )


class Distribution(Record):
    probabilities: dict[str, float]

    @model_validator(mode="after")
    def normalized(self):
        values = list(self.probabilities.values())
        if not values or any(not math.isfinite(p) or p < 0 or p > 1 for p in values):
            raise ValueError("invalid probability distribution")
        if not math.isclose(sum(values), 1, abs_tol=1e-5):
            raise ValueError("probabilities must sum to 1")
        return self

    @property
    def choice(self) -> str:
        return max(self.probabilities, key=self.probabilities.get)

    @property
    def probability(self) -> float:
        return self.probabilities[self.choice]


def certain(value: str) -> Distribution:
    return Distribution(probabilities={value: 1.0})


class Judgement(Record):
    target: Distribution = Field(default_factory=lambda: certain("unrelated"))
    stance: Distribution = Field(default_factory=lambda: certain("none"))
    alternative: Distribution = Field(default_factory=lambda: certain("additional"))
    shift: Distribution = Field(default_factory=lambda: certain("no"))
    relevance: Distribution = Field(default_factory=lambda: certain("on"))
    resolution: Distribution = Field(default_factory=lambda: certain("none"))
    correction: Distribution = Field(default_factory=lambda: certain("none"))
    # Atomic routing questions needed to implement creation rules in spec 05.
    trigger: Distribution = Field(default_factory=lambda: certain("none"))
    parent: Distribution = Field(default_factory=lambda: certain("focus"))
    response_to: Distribution = Field(default_factory=lambda: certain("none"))
    switch: Distribution = Field(default_factory=lambda: certain("default"))
    agreement: Distribution = Field(default_factory=lambda: certain("no"))
    decision_answer: Distribution = Field(default_factory=lambda: certain("yes"))


class Stance(Record):
    speaker_uid: str
    name: str
    speaker_confidence: float | None
    probability: float
    value: Literal["support", "concern"]
    source_turns: tuple[str, ...]
    reason: str = ""


class ResolutionEvidence(Record):
    status: Literal["decided", "held", "withdrawn"]
    answer: Literal["yes", "no"] | None = None
    target_id: str
    decision_turn: Turn
    decision_as_agreement: bool = False
    agreements: tuple[Turn, ...] = ()
    explicit_agreements: tuple[Turn, ...] = ()
    summary_confirmation: tuple[Turn, ...] = ()


class Issue(Record):
    kind: Literal["issue"] = "issue"
    id: str
    label: str
    answer_type: Literal["yes_no", "choice", "open"]
    parent_id: str | None
    status: Literal["open", "decided", "held", "withdrawn"] = "open"
    decided_answer: Literal["yes", "no"] | None = None
    resolution_evidence: tuple[ResolutionEvidence, ...] = ()
    needs_confirmation: bool = False
    confirmation_reasons: tuple[str, ...] = ()
    stances: dict[str, Stance] = Field(default_factory=dict)
    source_turns: tuple[str, ...] = ()
    original_position: str = ""
    last_mentioned_ms: int = 0


class Position(Record):
    kind: Literal["position"] = "position"
    id: str
    label: str
    parent_id: str
    status: Literal["proposed", "adopted", "rejected"] = "proposed"
    stances: dict[str, Stance] = Field(default_factory=dict)
    reasons: tuple[dict[str, Any], ...] = ()
    source_turns: tuple[str, ...] = ()
    last_mentioned_ms: int = 0


Node = Issue | Position


class Event(Record):
    type: str
    ms: int
    turn_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class StructureProposal(Record):
    operation: str
    node_id: str | None = None
    parent_id: str | None = None
    reason: str


class Tree:
    """Single-owner mutable collection of immutable node records."""

    def __init__(self, agenda: str):
        self.nodes: dict[str, Node] = {
            "root": Issue(id="root", label=agenda, answer_type="open", parent_id=None)
        }
        self.focus_id = "root"

    def children(self, node_id: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.parent_id == node_id]

    def path(self, node_id: str) -> list[Node]:
        result = []
        while node_id:
            node = self.nodes[node_id]
            result.append(node)
            node_id = node.parent_id
        return result[::-1]

    def issue_id(self, node_id: str) -> str:
        node = self.nodes[node_id]
        return node.id if isinstance(node, Issue) else node.parent_id

    def add(self, node: Node):
        if node.id in self.nodes or node.parent_id not in self.nodes:
            raise ValueError("duplicate ID or missing parent")
        parent = self.nodes[node.parent_id]
        if isinstance(node, Position) and (
            not isinstance(parent, Issue) or parent.answer_type == "yes_no"
        ):
            raise ValueError("Position requires choice/open Issue parent")
        self.nodes[node.id] = node

    def replace(self, node_id: str, **updates):
        node = self.nodes[node_id]
        self.nodes[node_id] = type(node).model_validate(node.model_dump() | updates)

    def snapshot(self) -> dict:
        return {
            "focus_id": self.focus_id,
            "nodes": [n.model_dump(mode="json") for n in self.nodes.values()],
        }

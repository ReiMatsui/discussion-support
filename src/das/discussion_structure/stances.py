"""Only the speaker's own clear utterance changes their stance."""

from .config import Config
from .models import Issue, Judgement, Position, Stance, Tree, Turn


def apply(
    tree: Tree,
    node_id: str,
    turn: Turn,
    judgement: Judgement,
    config: Config,
    emit,
    *,
    proposer: bool = False,
):
    node = tree.nodes[node_id]
    if isinstance(node, Issue) and node.answer_type != "yes_no":
        return
    value = "support" if proposer else judgement.stance.choice
    probability = 1.0 if proposer else judgement.stance.probability
    if not turn.substantive or probability < config.stance_threshold:
        return
    if value == "none":
        if (
            judgement.correction.choice == "self"
            and judgement.correction.probability >= config.stance_threshold
        ):
            stances = dict(node.stances)
            previous = stances.pop(turn.uid, None)
            if previous:
                tree.replace(node_id, stances=stances)
                emit(
                    "stance_transition",
                    node_id=node_id,
                    speaker_uid=turn.uid,
                    previous=previous.model_dump(),
                    current=None,
                )
        return
    if value not in {"support", "concern"}:
        return
    if isinstance(node, Position) and value == "support":
        parent = tree.nodes[node.parent_id]
        switch = judgement.switch.choice
        remove = switch == "replace" or (parent.answer_type == "choice" and switch != "both")
        if remove:
            for sibling in tree.children(parent.id):
                old = sibling.stances.get(turn.uid)
                if sibling.id != node_id and old and old.value == "support":
                    stances = dict(sibling.stances)
                    del stances[turn.uid]
                    tree.replace(sibling.id, stances=stances)
                    emit(
                        "stance_transition",
                        node_id=sibling.id,
                        speaker_uid=turn.uid,
                        previous=old.model_dump(),
                        current=None,
                        reason="switch",
                    )
    old = node.stances.get(turn.uid)
    stance = Stance(
        speaker_uid=turn.uid,
        name=turn.speaker if turn.identified(config.name_threshold) else "?",
        speaker_confidence=turn.speaker_confidence,
        probability=probability,
        value=value,
        source_turns=(*(old.source_turns if old else ()), turn.turn_id),
        reason=turn.text if value == "concern" else "",
    )
    tree.replace(node_id, stances=node.stances | {turn.uid: stance})
    emit(
        "stance_transition",
        node_id=node_id,
        speaker_uid=turn.uid,
        previous=old.model_dump() if old else None,
        current=stance.model_dump(),
    )
    if isinstance(node, Position) and value == "concern":
        tree.replace(node_id, reasons=(*node.reasons, {"text": turn.text, "turn_id": turn.turn_id}))

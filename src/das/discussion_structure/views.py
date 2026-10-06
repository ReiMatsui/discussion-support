"""Bounded conversation context: never send the entire unrelated tree."""

from .config import Config
from .models import Issue, Tree, Turn


def build_view(
    tree: Tree, recent: list[Turn], turn: Turn, config: Config, pending: dict | None = None
) -> dict:
    path = tree.path(tree.focus_id)
    subtree = []

    def visit(node_id):
        subtree.append(tree.nodes[node_id])
        for child in tree.children(node_id):
            visit(child.id)

    visit(tree.focus_id)
    local = {n.id for n in path + subtree}
    other = sorted(
        (
            n
            for n in tree.nodes.values()
            if isinstance(n, Issue) and n.status == "open" and n.id not in local
        ),
        key=lambda n: n.last_mentioned_ms,
        reverse=True,
    )[: config.other_issues]

    def brief(n):
        return {
            "id": n.id,
            "label": n.label,
            "kind": n.kind,
            "parent_id": n.parent_id,
            "status": n.status,
            **({"answer_type": n.answer_type} if isinstance(n, Issue) else {}),
        }

    return {
        "agenda_question": f"{tree.nodes['root'].label}として何を話し合うか？",
        "focus_id": tree.focus_id,
        "path": [brief(n) for n in path],
        "subtree": [brief(n) for n in subtree],
        "other_issues": [brief(n) for n in other],
        "recent_turns": [t.model_dump(mode="json") for t in recent[-config.recent_turns :]],
        "turn": turn.model_dump(mode="json"),
        "pending": pending or {},
    }

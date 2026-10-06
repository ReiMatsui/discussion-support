"""Stable insertion order, Issue-depth windows, bounded deterministic folding."""

from .config import Config
from .models import Issue, Tree


class Display:
    def __init__(self, config: Config):
        self.config = config
        self.current: dict | None = None
        self.last_ms: int | None = None
        self.top_id = "root"
        self.focus_id = None

    def build(self, tree: Tree) -> dict:
        if tree.focus_id != self.focus_id:
            path = tree.path(tree.focus_id)
            issues = [n for n in path if isinstance(n, Issue)]
            # One Issue above the focus, focus, one below (three Issue levels).
            top = issues[max(0, len(issues) - 2)]
            self.top_id, self.focus_id = top.id, tree.focus_id
        ordered = []
        depths = {}
        collapsed = {}

        def visit(node_id, depth, issue_depth):
            if issue_depth > self.config.display_levels:
                return
            ordered.append(node_id)
            depths[node_id] = depth
            for child in tree.children(node_id):
                next_depth = issue_depth + int(isinstance(child, Issue))
                if next_depth > self.config.display_levels:
                    collapsed[node_id] = collapsed.get(node_id, 0) + 1
                else:
                    visit(child.id, depth + 1, next_depth)

        visit(self.top_id, 0, 1)
        visible = set(ordered)
        protected = {n.id for n in tree.path(tree.focus_id)}
        focus_path = {n.id for n in tree.path(tree.focus_id)}

        def distance(node_id):
            path = tree.path(node_id)
            return len(path) + len(focus_path) - 2 * len({n.id for n in path} & focus_path)

        candidates = sorted(
            (i for i in ordered if i not in protected),
            key=lambda i: (
                -distance(i),
                tree.nodes[i].status not in {"decided", "held", "withdrawn", "rejected"},
                tree.nodes[i].last_mentioned_ms,
                ordered.index(i),
            ),
        )
        for node_id in candidates:
            if len(visible) <= self.config.display_nodes:
                break
            descendants = {i for i in visible if node_id in {n.id for n in tree.path(i)}}
            if not descendants or descendants & protected:
                continue
            visible -= descendants
            parent_id = tree.nodes[node_id].parent_id
            collapsed[parent_id] = collapsed.get(parent_id, 0) + len(descendants)
        rows = []
        for node_id in ordered:
            if node_id not in visible:
                continue
            node = tree.nodes[node_id]
            rows.append(
                {
                    "id": node.id,
                    "kind": node.kind,
                    "label": node.label,
                    "parent_id": node.parent_id,
                    "depth": depths[node_id],
                    "status": node.status,
                    "focus": node.id == tree.focus_id,
                    "answer_type": node.answer_type if isinstance(node, Issue) else None,
                    "decided_answer": node.decided_answer if isinstance(node, Issue) else None,
                    "marks": [
                        {"name": s.name, "value": s.value}
                        for s in node.stances.values()
                        if s.probability >= self.config.stance_threshold
                    ],
                    "collapsed_count": collapsed.get(node_id, 0),
                }
            )
        return {
            "focus_id": tree.focus_id,
            "breadcrumb": [n.label for n in tree.path(self.top_id)[:-1]],
            "nodes": rows,
        }

    def commit(self, tree, ms, emit):
        if self.last_ms is not None and ms - self.last_ms < self.config.display_ms:
            return
        state = self.build(tree)
        if state != self.current:
            previous = self.current
            self.current, self.last_ms = state, ms
            changes = []
            if previous is None:
                changes = ["initial"]
            else:
                old = {n["id"]: n for n in previous["nodes"]}
                new = {n["id"]: n for n in state["nodes"]}
                if old.keys() != new.keys():
                    changes.append("structure")
                if previous["focus_id"] != state["focus_id"]:
                    changes.append("focus")
                for field, kind in (
                    ("label", "label"),
                    ("marks", "stance"),
                    ("status", "status"),
                    ("collapsed_count", "fold"),
                ):
                    if any(old[i][field] != new[i][field] for i in old.keys() & new.keys()):
                        changes.append(kind)
            emit("display_commit", state=state, changes=changes)


def timeline(events) -> str:
    lines = []
    for event in events:
        if event.type != "display_commit":
            continue
        state = event.data["state"]
        lines.append(
            f"[{event.ms / 1000:.3f}s] display_commit ({', '.join(event.data['changes'])})"
        )
        if state["breadcrumb"]:
            lines.append(" › ".join(state["breadcrumb"]))
        for row in state["nodes"]:
            marks = " ".join(
                ("○" if s["value"] == "support" else "□") + s["name"] for s in row["marks"]
            )
            answer = {"yes": "する", "no": "しない"}.get(row["decided_answer"], "")
            lines.append(
                "  " * row["depth"]
                + ("▶ " if row["focus"] else "- ")
                + f"{row['label']} [{row['status']}{': ' + answer if answer else ''}] {marks}"
                + (f" (+{row['collapsed_count']} 折りたたみ)" if row["collapsed_count"] else "")
            )
        lines.append("")
    return "\n".join(lines)

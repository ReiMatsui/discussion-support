"""pyvis ベースの議論グラフ可視化。

研究プロトタイプ向けのデバッグ用ビューアとして位置づけ、
公開仕様の出力ではない (公開向けは将来 ``presentation/`` 配下で別途設計)。

色とシェイプの規約:
  - source: utterance=楕円/青, document=四角/緑, web=ダイヤ/オレンジ
  - relation: support=緑, attack=赤、太さは confidence に比例
  - hover (title) には全文・author・rationale を表示
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from das.graph.schema import Edge, Node
from das.graph.store import GraphStore

if TYPE_CHECKING:
    from pyvis.network import Network as PyvisNetwork

_NODE_COLOR = {
    "utterance": "#5b9bd5",
    "document": "#70ad47",
    "web": "#ed7d31",
}

_NODE_SHAPE = {
    "utterance": "ellipse",
    "document": "box",
    "web": "diamond",
}

_EDGE_COLOR = {
    "support": "#2e7d32",
    "attack": "#c62828",
}

_DEFAULT_HEIGHT = "800px"
_LABEL_MAX_CHARS = 36


def _truncate(text: str, limit: int = _LABEL_MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _node_title(node: Node) -> str:
    return (
        f"id: {node.id}\n"
        f"type: {node.node_type}\n"
        f"source: {node.source}\n"
        f"author: {node.author}\n"
        f"text: {node.text}"
    )


def _edge_title(edge: Edge) -> str:
    rationale = edge.rationale or "(no rationale)"
    return (
        f"{edge.relation} (confidence={edge.confidence:.2f})\n"
        f"created_by: {edge.created_by}\n"
        f"rationale: {rationale}"
    )


def _build_network(store: GraphStore, *, height: str = _DEFAULT_HEIGHT) -> PyvisNetwork:
    from pyvis.network import Network  # 遅延 import: viz extras 未インストール環境を考慮

    net = Network(
        height=height,
        width="100%",
        directed=True,
        notebook=False,
        cdn_resources="in_line",
    )
    net.toggle_physics(True)

    for node in store.nodes():
        net.add_node(
            str(node.id),
            label=_truncate(node.text),
            title=_node_title(node),
            color=_NODE_COLOR.get(node.source, "#999999"),
            shape=_NODE_SHAPE.get(node.source, "ellipse"),
        )

    for edge in store.edges():
        net.add_edge(
            str(edge.src_id),
            str(edge.dst_id),
            title=_edge_title(edge),
            color=_EDGE_COLOR.get(edge.relation, "#999999"),
            width=1 + 3 * edge.confidence,
            arrows="to",
        )

    return net


def render_html(
    store: GraphStore,
    output: Path,
    *,
    height: str = _DEFAULT_HEIGHT,
) -> Path:
    """``store`` の内容を pyvis HTML として ``output`` に書き出す。"""

    net = _build_network(store, height=height)
    output.parent.mkdir(parents=True, exist_ok=True)
    html = net.generate_html(notebook=False)
    output.write_text(html, encoding="utf-8")
    return output


# --- JSON 入出力 ----------------------------------------------------------


def dump_snapshot(store: GraphStore, path: Path) -> Path:
    """ストアのスナップショットを JSON として書き出す。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(store.snapshot(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def load_snapshot(path: Path, store: GraphStore | None = None) -> GraphStore:
    """JSON スナップショットを読み込み、(必要なら新規) ストアに展開する。"""

    from das.graph.store import NetworkXGraphStore

    payload = json.loads(path.read_text(encoding="utf-8"))
    target = store if store is not None else NetworkXGraphStore()
    target.load_snapshot(payload)
    return target


__all__ = ["dump_snapshot", "load_snapshot", "render_html"]

"""NetworkX (in-memory) + SQLite (append-only ログ) 実装。

設計:
  - メモリ上は ``networkx.MultiDiGraph``。
  - SQLite には ``nodes`` ``edges`` 二表に append-only で書き込む。
  - 復元時はテーブルをリプレイして MultiDiGraph を再構築。
  - ``snapshot()`` で JSON 形式の dict を返し、外部ファイルにそのまま dump できる。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from uuid import UUID

import networkx as nx

from das.graph.schema import Edge, Node, NodeSource

_NODES_TABLE = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    written_at TEXT NOT NULL
)
"""

_EDGES_TABLE = """
CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    src_id TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    written_at TEXT NOT NULL
)
"""


class NetworkXGraphStore:
    """NetworkX をメモリ表現とした GraphStore 実装。"""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """``db_path`` が None なら in-memory SQLite を使う (テスト向け)。"""

        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._nodes: dict[UUID, Node] = {}
        self._edges: dict[UUID, Edge] = {}
        if db_path is None:
            self._conn = sqlite3.connect(":memory:")
            self._db_path: Path | None = None
        else:
            self._db_path = Path(db_path)
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path)
        self._conn.execute(_NODES_TABLE)
        self._conn.execute(_EDGES_TABLE)
        self._conn.commit()
        self._replay_from_db()

    # --- 書き込み -----------------------------------------------------

    def add_node(self, node: Node) -> None:
        if node.id in self._nodes:
            return
        self._nodes[node.id] = node
        self._graph.add_node(node.id, **{"node": node})
        self._conn.execute(
            "INSERT OR IGNORE INTO nodes(id, payload, written_at) VALUES (?, ?, ?)",
            (
                str(node.id),
                node.model_dump_json(),
                datetime.utcnow().isoformat(),
            ),
        )
        self._conn.commit()

    def add_edge(self, edge: Edge) -> None:
        if edge.id in self._edges:
            return
        if edge.src_id not in self._nodes or edge.dst_id not in self._nodes:
            raise ValueError(
                f"edge {edge.id} references unknown node(s): src={edge.src_id} dst={edge.dst_id}"
            )
        self._edges[edge.id] = edge
        self._graph.add_edge(edge.src_id, edge.dst_id, key=edge.id, **{"edge": edge})
        self._conn.execute(
            (
                "INSERT OR IGNORE INTO edges(id, src_id, dst_id, payload, written_at) "
                "VALUES (?, ?, ?, ?, ?)"
            ),
            (
                str(edge.id),
                str(edge.src_id),
                str(edge.dst_id),
                edge.model_dump_json(),
                datetime.utcnow().isoformat(),
            ),
        )
        self._conn.commit()

    # --- 読み出し -----------------------------------------------------

    def get_node(self, node_id: UUID) -> Node | None:
        return self._nodes.get(node_id)

    def get_edge(self, edge_id: UUID) -> Edge | None:
        return self._edges.get(edge_id)

    def nodes(self, source: NodeSource | None = None) -> Iterable[Node]:
        if source is None:
            return list(self._nodes.values())
        return [n for n in self._nodes.values() if n.source == source]

    def edges(self) -> Iterable[Edge]:
        return list(self._edges.values())

    def neighbors(
        self,
        node_id: UUID,
        *,
        direction: str = "both",
    ) -> Iterable[Edge]:
        if direction not in {"in", "out", "both"}:
            raise ValueError("direction must be 'in', 'out', or 'both'")
        result: list[Edge] = []
        if direction in {"out", "both"} and node_id in self._graph:
            for _, _, key in self._graph.out_edges(node_id, keys=True):
                edge = self._edges.get(key)
                if edge is not None:
                    result.append(edge)
        if direction in {"in", "both"} and node_id in self._graph:
            for _, _, key in self._graph.in_edges(node_id, keys=True):
                edge = self._edges.get(key)
                if edge is not None:
                    result.append(edge)
        return result

    # --- スナップショット --------------------------------------------

    def snapshot(self) -> dict:
        return {
            "nodes": [json.loads(n.model_dump_json()) for n in self._nodes.values()],
            "edges": [json.loads(e.model_dump_json()) for e in self._edges.values()],
        }

    def load_snapshot(self, payload: dict) -> None:
        """既存内容に追記する形でスナップショットを取り込む。"""

        for raw in payload.get("nodes", []):
            node = Node.model_validate(raw)
            self.add_node(node)
        for raw in payload.get("edges", []):
            edge = Edge.model_validate(raw)
            self.add_edge(edge)

    def close(self) -> None:
        self._conn.close()

    # --- 内部 ---------------------------------------------------------

    def _replay_from_db(self) -> None:
        cursor = self._conn.execute("SELECT payload FROM nodes ORDER BY written_at")
        for (payload,) in cursor.fetchall():
            node = Node.model_validate_json(payload)
            self._nodes[node.id] = node
            self._graph.add_node(node.id, **{"node": node})

        cursor = self._conn.execute("SELECT payload FROM edges ORDER BY written_at")
        for (payload,) in cursor.fetchall():
            edge = Edge.model_validate_json(payload)
            if edge.src_id in self._nodes and edge.dst_id in self._nodes:
                self._edges[edge.id] = edge
                self._graph.add_edge(edge.src_id, edge.dst_id, key=edge.id, **{"edge": edge})


__all__ = ["NetworkXGraphStore"]

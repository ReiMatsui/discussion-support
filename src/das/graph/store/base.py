"""GraphStore の Protocol 定義。

実装は ``networkx_store.py`` を既定とし、将来 Neo4j などに差し替え可能。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable
from uuid import UUID

from das.graph.schema import Edge, Node, NodeSource


@runtime_checkable
class GraphStore(Protocol):
    """議論グラフの読み書きインタフェース。"""

    # --- 書き込み -----------------------------------------------------

    def add_node(self, node: Node) -> None: ...

    def add_edge(self, edge: Edge) -> None: ...

    # --- 読み出し -----------------------------------------------------

    def get_node(self, node_id: UUID) -> Node | None: ...

    def get_edge(self, edge_id: UUID) -> Edge | None: ...

    def nodes(self, source: NodeSource | None = None) -> Iterable[Node]: ...

    def edges(self) -> Iterable[Edge]: ...

    def neighbors(
        self,
        node_id: UUID,
        *,
        direction: str = "both",
    ) -> Iterable[Edge]:
        """``direction`` は 'in' | 'out' | 'both'."""
        ...

    # --- スナップショット ----------------------------------------------

    def snapshot(self) -> dict: ...

    def load_snapshot(self, payload: dict) -> None: ...

    def close(self) -> None: ...


__all__ = ["GraphStore"]

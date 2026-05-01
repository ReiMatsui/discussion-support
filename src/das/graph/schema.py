"""議論グラフ (Argumentation Framework) のノード・エッジ定義。

ノード:
  発話・文献・Web 由来の論証単位 (claim / premise) を統一的に保持する。

エッジ:
  ノード間の支持 (support) / 攻撃 (attack) 関係を、推定信頼度と理由付きで保持する。

設計上の決め事:
  - 両者とも frozen にして、変更したい場合は新しいオブジェクトを作る (履歴性を担保)。
  - id は UUID4 を自動採番、メタ情報は metadata: dict[str, Any] にまとめる。
  - 議論側 (utterance) と知識側 (document, web) は source 列で見分ける。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

NodeType = Literal["claim", "premise"]
NodeSource = Literal["utterance", "document", "web"]
Relation = Literal["support", "attack"]
EdgeCreator = Literal["extraction", "linking", "manual"]


class Node(BaseModel):
    """論証グラフの 1 ノード。"""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    text: str
    node_type: NodeType
    source: NodeSource
    author: str | None = None
    """発話なら話者ID、文献なら doc_id、Web なら domain。"""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class Edge(BaseModel):
    """ノード間の支持・攻撃エッジ。"""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    src_id: UUID
    dst_id: UUID
    relation: Relation
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    rationale: str = ""
    created_by: EdgeCreator = "linking"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


__all__ = [
    "Edge",
    "EdgeCreator",
    "Node",
    "NodeSource",
    "NodeType",
    "Relation",
]

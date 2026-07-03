"""議論グラフ (Argumentation Framework) のノード・エッジ定義。

ノード:
  発話・文献・Web 由来のノードを統一的に保持する。
  - 議論側 (source="utterance") は立場を持つ論証単位なので claim / premise。
  - 知識側 (source="document" / "web") は中立な事実なので evidence。
    事実が「支持」か「攻撃」かは対象主張ごとに相対的なので、スタンスは
    ノード自体ではなくエッジ (対象主張ごと) に持たせる。

エッジ:
  ノード間の支持 (support) / 攻撃 (attack) 関係を、推定信頼度と理由付きで保持する。
  中立 (どちらでもない) は「エッジを張らない」で表現する (新しい relation は足さない)。

設計上の決め事:
  - 両者とも frozen にして、変更したい場合は新しいオブジェクトを作る (履歴性を担保)。
  - id は UUID4 を自動採番、メタ情報は metadata: dict[str, Any] にまとめる。
  - 議論側 (utterance) と知識側 (document, web) は source 列で見分ける。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

NodeType = Literal["claim", "premise", "evidence"]
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

    turn_index: int = 0
    """会議内の確定発話連番 (アクティブ窓判定に使う, logic_review A5)。
    発話ノードは元発話の連番、evidence (文献/Web) は投入時点の連番。
    後方互換のため default 0 (未設定の古いスナップショットは 0 扱い)。"""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


__all__ = [
    "Edge",
    "EdgeCreator",
    "Node",
    "NodeSource",
    "NodeType",
    "Relation",
]

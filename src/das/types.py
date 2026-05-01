"""エージェント間で共有する基本型。

Event はオーケストレータが配信する入力、Mutation はエージェントが返す差分、
Intervention はファシリテーション結果として参加者に届く提示物。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

# --- Event 型 -------------------------------------------------------------


class Utterance(BaseModel):
    """1 つの発話チャンク。論証抽出エージェントの入力単位。"""

    model_config = ConfigDict(frozen=True)

    turn_id: int
    speaker: str
    text: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NodeAdded(BaseModel):
    """グラフにノードが追加されたイベント。下流エージェントを起動する。"""

    model_config = ConfigDict(frozen=True)

    node_id: UUID
    source: Literal["utterance", "document", "web"]


class Tick(BaseModel):
    """時間トリガ。ファシリテーションエージェントが拾う。"""

    model_config = ConfigDict(frozen=True)

    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


Event = Utterance | NodeAdded | Tick


# --- Mutation 型 ----------------------------------------------------------


class AddNode(BaseModel):
    """ノード追加の差分。ノード本体は graph.schema.Node を後で生成して扱う。"""

    model_config = ConfigDict(frozen=True)

    node_id: UUID = Field(default_factory=uuid4)
    payload: dict


class AddEdge(BaseModel):
    model_config = ConfigDict(frozen=True)

    edge_id: UUID = Field(default_factory=uuid4)
    src_id: UUID
    dst_id: UUID
    payload: dict


Mutation = AddNode | AddEdge


# --- Intervention --------------------------------------------------------


class Intervention(BaseModel):
    """参加者または全員に対する介入。"""

    model_config = ConfigDict(frozen=True)

    audience: Literal["participant", "all", "facilitator"]
    target_speaker: str | None = None
    text: str
    related_node_ids: list[UUID] = Field(default_factory=list)
    rationale: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


__all__ = [
    "AddEdge",
    "AddNode",
    "Event",
    "Intervention",
    "Mutation",
    "NodeAdded",
    "Tick",
    "Utterance",
]

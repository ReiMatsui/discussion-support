"""論証抽出エージェント。

発話 (``Utterance``) を入力に取り、claim / premise の論証単位に分解して
``Node`` のリストを返す。話者 ID は ``Node.author``、turn_id 等の発話メタは
``Node.metadata`` に保持する。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from das.agents.base import BaseAgent
from das.graph.schema import Edge, Node, NodeType
from das.llm import OpenAIClient
from das.types import Utterance

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# 指示語・省略の解決に使う参照文脈の発話数 (G2, レビュー H-2)
_CONTEXT_TURNS = 3


class _ExtractedUnit(BaseModel):
    """LLM から返ってくる 1 つの論証単位。"""

    text: str = Field(description="抽出された論証文 (指示語は文脈から解決した自己完結文)")
    node_type: Literal["claim", "premise"] = Field(description="claim=主張, premise=前提・根拠")


class _IntraEdge(BaseModel):
    """発話内の論証単位どうしの関係 (premise→claim の支持など)。

    出力を軽くするため rationale 等は持たせない (生成時間を抑える, G2)。
    """

    src: int = Field(description="関係の起点となる unit の index (0 始まり)")
    dst: int = Field(description="関係の終点となる unit の index (0 始まり)")
    relation: Literal["support", "attack"] = Field(description="src が dst を support/attack")


class _ExtractionResult(BaseModel):
    """LLM からの構造化出力全体。"""

    units: list[_ExtractedUnit] = Field(default_factory=list)
    intra_edges: list[_IntraEdge] = Field(default_factory=list)


@dataclass(frozen=True)
class ExtractionOutput:
    """抽出結果: ノード群と、発話内で確定した関係エッジ群。"""

    nodes: list[Node]
    edges: list[Edge]


def _load_system_prompt() -> str:
    return (_PROMPTS_DIR / "extraction.md").read_text(encoding="utf-8")


class ExtractionAgent(BaseAgent):
    """発話を claim / premise ノードに分解する。"""

    name = "extraction"

    def __init__(self, llm: OpenAIClient | None = None) -> None:
        super().__init__(llm=llm)
        self._system_prompt = _load_system_prompt()

    async def extract(
        self,
        utterance: Utterance,
        context: list[Utterance] | None = None,
    ) -> ExtractionOutput:
        """発話 1 つを論証ノード + 発話内エッジに分解する (G2)。

        ``context`` に直近発話 (話者名付き) を渡すと、指示語・省略を解決した
        自己完結文としてノード化させる。**ノード化するのは ``utterance`` のみ** で、
        context は参照専用 (fact 判定 / triage と同じ規約)。発話内の premise→claim
        などの関係は ``intra_edges`` としてまとめて返し、``created_by="extraction"``
        でエッジ化する (後段 linking での再判定を省く)。
        """

        user_content = self._build_user_content(utterance, context)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]
        result = await self.llm.chat_structured(
            messages,  # type: ignore[arg-type]
            response_format=_ExtractionResult,
        )

        # unit index → 生成ノード (空文字でスキップした unit は None) の対応表
        nodes: list[Node] = []
        index_to_node: dict[int, Node] = {}
        for idx, unit in enumerate(result.units):
            text = unit.text.strip()
            if not text:
                continue
            node_type: NodeType = unit.node_type
            node = Node(
                text=text,
                node_type=node_type,
                source="utterance",
                author=utterance.speaker,
                turn_index=utterance.turn_id,
                timestamp=utterance.timestamp,
                metadata={"turn_id": utterance.turn_id},
            )
            nodes.append(node)
            index_to_node[idx] = node

        # 発話内エッジ (premise→claim 等)。両端が有効な unit を指すもののみ採用。
        edges: list[Edge] = []
        for ie in result.intra_edges:
            src = index_to_node.get(ie.src)
            dst = index_to_node.get(ie.dst)
            if src is None or dst is None or src.id == dst.id:
                continue
            edges.append(
                Edge(
                    src_id=src.id,
                    dst_id=dst.id,
                    relation=ie.relation,
                    confidence=1.0,
                    created_by="extraction",
                )
            )

        self.log.info(
            "extraction.done",
            turn_id=utterance.turn_id,
            speaker=utterance.speaker,
            n_units=len(nodes),
            n_intra_edges=len(edges),
            has_context=bool(context),
        )
        return ExtractionOutput(nodes=nodes, edges=edges)

    @staticmethod
    def _build_user_content(
        utterance: Utterance, context: list[Utterance] | None
    ) -> str:
        parts: list[str] = []
        if context:
            recent = context[-_CONTEXT_TURNS:]
            ctx_lines = "\n".join(f"  {u.speaker}: {u.text}" for u in recent)
            parts.append(
                "## 参照文脈 (指示語・省略の解決にのみ使う。ノード化しない)\n" + ctx_lines
            )
        parts.append(
            "## 判定対象の発話 (この発話だけをノード化する)\n"
            f"発話番号: {utterance.turn_id}\n話者: {utterance.speaker}\n発話: {utterance.text}"
        )
        return "\n\n".join(parts)


__all__ = ["ExtractionAgent", "ExtractionOutput"]

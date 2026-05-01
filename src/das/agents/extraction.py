"""論証抽出エージェント。

発話 (``Utterance``) を入力に取り、claim / premise の論証単位に分解して
``Node`` のリストを返す。話者 ID は ``Node.author``、turn_id 等の発話メタは
``Node.metadata`` に保持する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from das.agents.base import BaseAgent
from das.graph.schema import Node, NodeType
from das.llm import OpenAIClient
from das.types import Utterance

_PROMPTS_DIR = Path(__file__).parent / "prompts"


class _ExtractedUnit(BaseModel):
    """LLM から返ってくる 1 つの論証単位。"""

    text: str = Field(description="抽出された論証文 (原文の意味を保つ)")
    node_type: Literal["claim", "premise"] = Field(description="claim=主張, premise=前提・根拠")


class _ExtractionResult(BaseModel):
    """LLM からの構造化出力全体。"""

    units: list[_ExtractedUnit] = Field(default_factory=list)


def _load_system_prompt() -> str:
    return (_PROMPTS_DIR / "extraction.md").read_text(encoding="utf-8")


class ExtractionAgent(BaseAgent):
    """発話を claim / premise ノードに分解する。"""

    name = "extraction"

    def __init__(self, llm: OpenAIClient | None = None) -> None:
        super().__init__(llm=llm)
        self._system_prompt = _load_system_prompt()

    async def extract(self, utterance: Utterance) -> list[Node]:
        """発話 1 つを論証ノードに分解する。"""

        user_content = (
            f"発話番号: {utterance.turn_id}\n話者: {utterance.speaker}\n発話: {utterance.text}"
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]
        result = await self.llm.chat_structured(
            messages,  # type: ignore[arg-type]
            response_format=_ExtractionResult,
        )

        nodes: list[Node] = []
        for unit in result.units:
            text = unit.text.strip()
            if not text:
                continue
            node_type: NodeType = unit.node_type
            nodes.append(
                Node(
                    text=text,
                    node_type=node_type,
                    source="utterance",
                    author=utterance.speaker,
                    timestamp=utterance.timestamp,
                    metadata={"turn_id": utterance.turn_id},
                )
            )

        self.log.info(
            "extraction.done",
            turn_id=utterance.turn_id,
            speaker=utterance.speaker,
            n_units=len(nodes),
        )
        return nodes


__all__ = ["ExtractionAgent"]

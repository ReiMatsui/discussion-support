"""エージェント共通の基底。

各エージェントは自分の入力/出力に合わせた特化メソッドを持つ
(``ExtractionAgent.extract``, ``LinkingAgent.link`` のように)。
基底クラスは LLM クライアントとロガーの依存だけを面倒見る。
"""

from __future__ import annotations

from das.llm import OpenAIClient
from das.logging import get_logger


class BaseAgent:
    """全エージェントが継承する最小の基底クラス。"""

    name: str = "base"

    def __init__(self, llm: OpenAIClient | None = None) -> None:
        self.llm = llm or OpenAIClient()
        self.log = get_logger(f"das.agents.{self.name}")


__all__ = ["BaseAgent"]

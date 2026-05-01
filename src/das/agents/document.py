"""ドキュメント知識エージェント。

`data/docs/` 配下の事前文書を論証単位に分解し、``source="document"`` の
``Node`` として ``GraphStore`` に書き込む。議論進行中は対象ノードに対して
関連する文書ノードを返す。

M1 段階では retrieve は単純に「ストア内の全文書ノード」を返す。
embedding 類似度による絞り込みは将来拡張。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from das.agents.base import BaseAgent
from das.graph.schema import Node
from das.graph.store import GraphStore
from das.llm import OpenAIClient
from das.types import Utterance

_PROMPTS_DIR = Path(__file__).parent / "prompts"


class _DocumentUnit(BaseModel):
    text: str = Field(description="抽出された論証文")
    node_type: Literal["claim", "premise"]


class _DocumentExtraction(BaseModel):
    units: list[_DocumentUnit] = Field(default_factory=list)


def _load_system_prompt() -> str:
    return (_PROMPTS_DIR / "document.md").read_text(encoding="utf-8")


class DocumentAgent(BaseAgent):
    """事前文書を AF 化し、対象ノード向けに候補ノードを返すエージェント。"""

    name = "document"

    def __init__(self, llm: OpenAIClient | None = None) -> None:
        super().__init__(llm=llm)
        self._system_prompt = _load_system_prompt()

    # --- 取り込み ----------------------------------------------------

    async def ingest_text(
        self,
        text: str,
        *,
        doc_id: str,
        store: GraphStore,
        source_path: str | None = None,
    ) -> list[Node]:
        """1 文書のテキストを AF 化して ``store`` に追記する。"""

        user_content = f"文書ID: {doc_id}\n\n本文:\n{text}"
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]
        result = await self.llm.chat_structured(
            messages,  # type: ignore[arg-type]
            response_format=_DocumentExtraction,
        )

        nodes: list[Node] = []
        for unit in result.units:
            cleaned = unit.text.strip()
            if not cleaned:
                continue
            metadata: dict[str, object] = {"doc_id": doc_id}
            if source_path is not None:
                metadata["source_path"] = source_path
            node = Node(
                text=cleaned,
                node_type=unit.node_type,
                source="document",
                author=doc_id,
                metadata=metadata,
            )
            store.add_node(node)
            nodes.append(node)

        self.log.info(
            "document.ingest",
            doc_id=doc_id,
            n_units=len(nodes),
            source_path=source_path,
        )
        return nodes

    async def ingest_directory(
        self,
        directory: Path,
        store: GraphStore,
        *,
        extensions: tuple[str, ...] = (".md", ".txt"),
    ) -> list[Node]:
        """ディレクトリ内のテキストファイルをすべて AF 化する。"""

        all_nodes: list[Node] = []
        for path in sorted(directory.iterdir()):
            if path.is_dir():
                continue
            if path.suffix.lower() not in extensions:
                continue
            text = path.read_text(encoding="utf-8")
            doc_id = path.stem
            nodes = await self.ingest_text(
                text,
                doc_id=doc_id,
                store=store,
                source_path=str(path),
            )
            all_nodes.extend(nodes)
        return all_nodes

    # --- 取得 --------------------------------------------------------

    def retrieve(
        self,
        target: Node | Utterance,
        store: GraphStore,
        *,
        limit: int | None = None,
    ) -> list[Node]:
        """対象ノード/発話に対する候補となる文書ノードを返す。

        M1: 単純にストア内の全文書ノードを (制限つきで) 返す。
        """

        document_nodes = [n for n in store.nodes() if n.source == "document"]
        if limit is None:
            return document_nodes
        return document_nodes[:limit]


__all__ = ["DocumentAgent"]

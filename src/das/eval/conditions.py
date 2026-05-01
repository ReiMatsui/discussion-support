"""3 つの比較条件 (None / FlatRAG / FullProposal)。

各 ``Condition`` は ``SessionRunner.info_provider`` として注入できる。

- :class:`ConditionNone`: 常に ``None`` を返す (情報提供なし)
- :class:`ConditionFlatRAG`: 文書を段落単位で embed しておき、直近発話に類似する
  チャンクを top-k 件、生テキストで連結して返す
- :class:`ConditionFullProposal`: 内部で :class:`Orchestrator` を 1 つ持ち、
  履歴の発話を逐次バスに流して統合議論グラフを構築。直近発話に対して張られた
  支持・攻撃エッジを参加者向けの短い通知として整形して返す

研究計画書 §5.2 の段階 B (シミュレーション評価) で 3 条件比較を行うときの
基本コンポーネント。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from das.agents.linking import cosine_similarity
from das.eval.persona import PersonaSpec
from das.graph.schema import NodeSource
from das.graph.store import NetworkXGraphStore
from das.llm import OpenAIClient
from das.logging import get_logger
from das.runtime import Orchestrator
from das.types import Utterance


@dataclass(frozen=True)
class InfoItem:
    """1 件の参考情報 (UI/分析向けの構造化表現)。

    LLM 向けの prompt 文字列とは別に保持して、Streamlit UI で色分け・出典付きの
    リッチ表示ができるようにする。
    """

    relation: Literal["support", "attack"]
    target_text: str
    target_speaker: str | None
    source_text: str
    source_kind: NodeSource
    source_author: str | None
    confidence: float
    rationale: str = ""


@dataclass(frozen=True)
class FlatRAGItem:
    """FlatRAG が返す 1 チャンク (UI/分析向け)。"""

    doc_id: str
    text: str
    score: float = 0.0


class Condition(Protocol):
    """情報提供条件の共通プロトコル。"""

    name: str

    async def setup(self, *, docs_dir: Path | None = None) -> None: ...

    async def info_provider(self, history: list[Utterance], persona: PersonaSpec) -> str | None: ...


# --- None ---------------------------------------------------------------


class ConditionNone:
    """情報提供を一切行わないベースライン。"""

    name = "none"

    async def setup(self, *, docs_dir: Path | None = None) -> None:
        return None

    async def info_provider(self, history: list[Utterance], persona: PersonaSpec) -> str | None:
        return None


# --- FlatRAG ------------------------------------------------------------


def _chunk_document(text: str) -> list[str]:
    """空行区切りで段落に分割する単純チャンカ。"""

    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


class ConditionFlatRAG:
    """文書を段落単位で embed し、直近発話との類似度 top-k を渡す。"""

    name = "flat_rag"

    def __init__(
        self,
        llm: OpenAIClient | None = None,
        *,
        top_k: int = 3,
    ) -> None:
        self._llm = llm or OpenAIClient()
        self._top_k = top_k
        self._chunks: list[tuple[str, str]] = []
        self._embeddings: list[list[float]] = []
        self._last_items: list[FlatRAGItem] = []
        self._log = get_logger("das.eval.condition.flat_rag")

    @property
    def last_items(self) -> list[FlatRAGItem]:
        return list(self._last_items)

    async def setup(self, *, docs_dir: Path | None = None) -> None:
        if docs_dir is None or not docs_dir.exists():
            return
        chunks: list[tuple[str, str]] = []
        for path in sorted(docs_dir.iterdir()):
            if path.is_dir():
                continue
            if path.suffix.lower() not in {".md", ".txt"}:
                continue
            text = path.read_text(encoding="utf-8")
            for paragraph in _chunk_document(text):
                chunks.append((path.stem, paragraph))
        if not chunks:
            return
        texts = [t for _, t in chunks]
        vectors = await self._llm.embed(texts)
        self._chunks = chunks
        self._embeddings = vectors
        self._log.info("flat_rag.setup", n_chunks=len(chunks))

    async def info_provider(self, history: list[Utterance], persona: PersonaSpec) -> str | None:
        if not history or not self._chunks:
            self._last_items = []
            return None
        query = history[-1].text
        query_vec = await self._llm.embed_one(query)
        scored = [
            (chunk, cosine_similarity(query_vec, vec))
            for chunk, vec in zip(self._chunks, self._embeddings, strict=True)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        top = scored[: self._top_k]
        if not top:
            self._last_items = []
            return None
        self._last_items = [
            FlatRAGItem(doc_id=chunk[0], text=chunk[1], score=score) for chunk, score in top
        ]
        return "\n\n".join(f"[{doc_id}] {text}" for (doc_id, text), _ in top)


# --- FullProposal -------------------------------------------------------


class ConditionFullProposal:
    """提案手法。Orchestrator が AF を育てつつ、直近発話への支持/攻撃を返す。"""

    name = "full_proposal"

    def __init__(
        self,
        llm: OpenAIClient | None = None,
        *,
        threshold: float | None = None,
        top_k: int = 5,
        max_info_items: int = 3,
    ) -> None:
        self._llm = llm or OpenAIClient()
        self._threshold = threshold
        self._top_k = top_k
        self._max_info_items = max_info_items
        self._orchestrator: Orchestrator | None = None
        self._processed_turn_ids: set[int] = set()
        self._last_items: list[InfoItem] = []
        self._log = get_logger("das.eval.condition.full_proposal")

    @property
    def orchestrator(self) -> Orchestrator | None:
        """セットアップ後にライブビューなどから参照するためのフック。"""

        return self._orchestrator

    @property
    def last_items(self) -> list[InfoItem]:
        """直近の info_provider 呼び出しで生成された InfoItem 群。"""

        return list(self._last_items)

    async def setup(self, *, docs_dir: Path | None = None) -> None:
        store = NetworkXGraphStore()
        self._orchestrator = Orchestrator.assemble(
            llm=self._llm,
            store=store,
            threshold=self._threshold,
            top_k=self._top_k,
        )
        if docs_dir is not None and docs_dir.exists():
            await self._orchestrator.ingest_documents(docs_dir)
        self._log.info("full_proposal.setup")

    async def info_provider(self, history: list[Utterance], persona: PersonaSpec) -> str | None:
        self._last_items = []
        if not history or self._orchestrator is None:
            return None

        # 未処理の発話だけ追加で流す
        for utterance in history:
            if utterance.turn_id not in self._processed_turn_ids:
                await self._orchestrator.bus.publish(utterance)
                self._processed_turn_ids.add(utterance.turn_id)
        await self._orchestrator.bus.drain()

        last_turn_id = history[-1].turn_id
        store = self._orchestrator.store
        related_nodes = [
            n
            for n in store.nodes()
            if n.source == "utterance" and n.metadata.get("turn_id") == last_turn_id
        ]
        if not related_nodes:
            return None

        items: list[InfoItem] = []
        seen: set[str] = set()
        for node in related_nodes:
            for edge in store.neighbors(node.id, direction="in"):
                src = store.get_node(edge.src_id)
                if src is None or src.id == node.id:
                    continue
                key = f"{edge.relation}|{src.id}|{node.id}"
                if key in seen:
                    continue
                seen.add(key)
                items.append(
                    InfoItem(
                        relation=edge.relation,
                        target_text=node.text,
                        target_speaker=node.author,
                        source_text=src.text,
                        source_kind=src.source,
                        source_author=src.author,
                        confidence=edge.confidence,
                        rationale=edge.rationale,
                    )
                )

        items = items[: self._max_info_items]
        self._last_items = items
        if not items:
            return None
        lines = [
            f"[{'支持' if it.relation == 'support' else '反論'}] {it.source_text}" for it in items
        ]
        return "\n".join(lines)


__all__ = [
    "Condition",
    "ConditionFlatRAG",
    "ConditionFullProposal",
    "ConditionNone",
    "FlatRAGItem",
    "InfoItem",
]

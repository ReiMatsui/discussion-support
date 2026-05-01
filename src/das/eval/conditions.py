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

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from das.agents.facilitation import FacilitationAgent, InfoItem, InterventionDecision
from das.agents.linking import cosine_similarity
from das.eval.persona import PersonaSpec
from das.graph.store import NetworkXGraphStore
from das.llm import OpenAIClient
from das.logging import get_logger
from das.runtime import Orchestrator
from das.types import Utterance


@dataclass(frozen=True)
class InterventionLogEntry:
    """1 介入分の記録 (§4.3 介入の透明性)。

    L1/L2/skip いずれの判断も全て記録する。配信チャネル非依存にするため、
    ``turn_id`` は補助情報として残し、本質的な識別は ``timestamp`` と
    ``triggered_by_speaker`` で行う。
    """

    turn_id: int
    persona_name: str
    """旧フィールド: シミュレーションでの「次話者」または triggered_by_speaker。
    対面では triggered_by_speaker と同義。"""
    timestamp: str
    items: list[dict] = field(default_factory=list)
    kind: str = "l1"
    """"l1" / "l2" / "skip"。"""

    addressed_to: str | None = None
    """L1 の通知対象話者。L2 / skip では None (= 全員 / 該当なし)。"""

    brief: str = ""
    """L2 のときの俯瞰サマリ本文。L1 / skip では空。"""

    decision_reason: str = ""
    """この判断に至ったロジカルな理由。"""


def write_intervention_log(entries: list[InterventionLogEntry], path: Path) -> Path:
    """介入ログを JSONL として書き出す。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry), ensure_ascii=False, default=str))
            f.write("\n")
    return path


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
        max_info_items: int = 2,
        facilitator: FacilitationAgent | None = None,
    ) -> None:
        self._llm = llm or OpenAIClient()
        self._threshold = threshold
        self._top_k = top_k
        self._max_info_items = max_info_items
        self._orchestrator: Orchestrator | None = None
        self._processed_turn_ids: set[int] = set()
        self._last_items: list[InfoItem] = []
        self._last_decision_kind: str = "skip"
        self._intervention_log: list[InterventionLogEntry] = []
        # ファシリテーション (中央調停) は FacilitationAgent に委譲する
        self._facilitator = facilitator or FacilitationAgent(
            llm=self._llm,
            max_items=max_info_items,
        )
        self._log = get_logger("das.eval.condition.full_proposal")

    @property
    def intervention_log(self) -> list[InterventionLogEntry]:
        """各ターンで誰に何を提示したかの履歴 (§4.3 介入の透明性)。"""

        return list(self._intervention_log)

    @property
    def orchestrator(self) -> Orchestrator | None:
        """セットアップ後にライブビューなどから参照するためのフック。"""

        return self._orchestrator

    @property
    def facilitator(self) -> FacilitationAgent:
        """ライブ UI から bias/stage を読むためのフック。"""

        return self._facilitator

    @property
    def last_items(self) -> list[InfoItem]:
        """直近の info_provider 呼び出しで生成された InfoItem 群。"""

        return list(self._last_items)

    @property
    def last_decision_kind(self) -> str:
        """直近の info_provider 呼び出しが返した介入種別 (skip/l1/l2)。"""

        return self._last_decision_kind

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
        """シミュレーション側のアダプタ。

        FacilitationAgent.decide_intervention で「介入の要否と内容」を
        モダリティ非依存に決定し、その結果を「次話者のプロンプト用テキスト」
        に翻訳する。対面 UI / 音声に展開するときはここを置き換える。
        """

        self._last_items = []
        self._last_decision_kind = "skip"
        if not history or self._orchestrator is None:
            return None

        # 未処理の発話だけ追加で流す (extraction → linking が走る)
        for utterance in history:
            if utterance.turn_id not in self._processed_turn_ids:
                await self._orchestrator.bus.publish(utterance)
                self._processed_turn_ids.add(utterance.turn_id)
        await self._orchestrator.bus.drain()

        store = self._orchestrator.store

        # 同期 API で「いつ・誰に・何を」を決める。L2 が選ばれた場合は
        # この段階での brief は deterministic なので、必要なら LLM 整文を後置する。
        decision = self._facilitator.decide_intervention(history, store)
        self._last_decision_kind = decision.kind

        # L2 のときは LLM で自然文に整える (失敗時は decision.brief をそのまま使う)
        if decision.kind == "l2":
            try:
                better_brief = await self._facilitator.compose_l2_brief(
                    history, store
                )
                if better_brief:
                    decision = InterventionDecision(
                        kind=decision.kind,
                        items=decision.items,
                        brief=better_brief,
                        addressed_to=decision.addressed_to,
                        reason=decision.reason,
                    )
            except Exception as exc:  # pragma: no cover - 防御的
                self._log.warning("l2_brief.format_failed", error=str(exc))

        # 介入ログ
        self._last_items = list(decision.items)
        self._intervention_log.append(
            InterventionLogEntry(
                turn_id=history[-1].turn_id,
                persona_name=persona.name,
                timestamp=datetime.now(timezone.utc).isoformat(),
                items=[asdict(it) for it in decision.items],
                kind=decision.kind,
                addressed_to=decision.addressed_to,
                brief=decision.brief,
                decision_reason=decision.reason,
            )
        )

        # シミュレーション固有の翻訳: 次話者のプロンプトに混ぜるテキストへ
        if decision.kind == "skip":
            return None
        if decision.kind == "l2":
            return f"[議論の整理 (全体共有)]\n{decision.brief}"
        # L1: addressed_to が次話者 (=persona.name) と一致するなら自分宛、
        # 違えば三人称で「Aさんの先ほどの発言には〜」と整形
        if decision.addressed_to == persona.name:
            return self._format_l1_self(decision)
        return self._format_l1_third_person(decision)

    @staticmethod
    def _format_l1_self(decision: InterventionDecision) -> str:
        lines = ["[あなたの先ほどの発言に対する関連情報]"]
        for it in decision.items:
            tag = "支持" if it.relation == "support" else "反論"
            lines.append(f"- [{tag}] {it.source_text}")
        return "\n".join(lines)

    @staticmethod
    def _format_l1_third_person(decision: InterventionDecision) -> str:
        speaker = decision.addressed_to or "直前の発言者"
        lines = [f"[{speaker}さんの先ほどの発言に対する関連情報]"]
        for it in decision.items:
            tag = "支持" if it.relation == "support" else "反論"
            lines.append(f"- [{tag}] {it.source_text}")
        return "\n".join(lines)


__all__ = [
    "Condition",
    "ConditionFlatRAG",
    "ConditionFullProposal",
    "ConditionNone",
    "FlatRAGItem",
    "InfoItem",
    "InterventionLogEntry",
    "write_intervention_log",
]

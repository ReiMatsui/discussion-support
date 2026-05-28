"""連結エージェント (本研究の中核)。

新しいノード ``target`` がストアに加わったとき:

  1. embedding 類似度で候補ノードを top-k 件に絞る (検索コスト削減)
  2. 各候補と ``target`` の論証関係を LLM に判定させる
       - 5 値: a_supports_b / a_attacks_b / b_supports_a / b_attacks_a / none
  3. confidence が閾値以上のものだけエッジとしてストアに書き込む

埋め込みはエージェント内のメモリキャッシュで管理し、同じノードを 2 度
embed しないようにする。M1 段階では十分。

候補選定モード:
  - 既定 (``top_k_per_source=None``): 全 source 混合で cosine top-k
  - source 別 (``top_k_per_source=N``): utterance / document / web の各 source ごとに
    cosine top-N を取り、結合して候補とする (貢献①「異種ソースの統合」を構造的に保証)
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from das.agents.base import BaseAgent
from das.graph.schema import Edge, Node, NodeSource, Relation
from das.graph.store import GraphStore
from das.llm import OpenAIClient
from das.settings import get_settings

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_RelationLabel = Literal[
    "a_supports_b",
    "a_attacks_b",
    "b_supports_a",
    "b_attacks_a",
    "none",
]


class _LinkJudgment(BaseModel):
    """LLM が返す 1 ペアの判定結果。"""

    relation: _RelationLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="")


def _load_system_prompt() -> str:
    return (_PROMPTS_DIR / "linking.md").read_text(encoding="utf-8")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """2 ベクトル間の cosine 類似度。次元不一致や零ベクトルは 0.0 を返す。"""

    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class LinkingAgent(BaseAgent):
    """対象ノードの近傍を埋め込みで絞り込み、LLM で支持/攻撃を判定する。"""

    name = "linking"

    def __init__(
        self,
        llm: OpenAIClient | None = None,
        *,
        threshold: float | None = None,
        top_k: int = 5,
        top_k_per_source: int | None = None,
        embedding_model: str | None = None,
        model_override: str | None = None,
        judge_timeout: float | None = 30.0,
    ) -> None:
        """
        Args:
            top_k: 混合 top-k モード時の候補上限 (``top_k_per_source`` 未設定時のみ有効)。
            top_k_per_source: source 別 top-k モード。``utterance`` / ``document`` /
                ``web`` の各バケットから cosine top-N を取り、結合して候補とする。
                指定すると ``top_k`` は無視される。発話どうしの類似度が高い議論で
                文書/Web ノードが top-k から押し出される問題を防ぐ。
            model_override: ``_judge_pair`` で使う LLM モデルを上書きする。
                例: ``"gpt-5-nano"`` を渡すと judgment 呼び出しのコストを大幅削減できる。
                None なら ``self.llm.fast_model`` (=既定モデル)。
            judge_timeout: 1 ペアの judge 呼び出しに対する秒数タイムアウト。
                ``asyncio.gather`` で並列実行しているため、1 件が hang するとバッチ全体が
                blocking してしまう。``asyncio.wait_for`` でラップし、超過したら
                relation=none / confidence=0 として次に進む。
                既定 30.0 秒 (gpt-5-nano は通常 5 秒以内に応答)。``None`` で無効化。
        """

        super().__init__(llm=llm)
        self._system_prompt = _load_system_prompt()
        settings = get_settings()
        self._threshold = threshold if threshold is not None else settings.linking_threshold
        self._top_k = top_k
        self._top_k_per_source = top_k_per_source
        self._embedding_model = embedding_model
        self._model_override = model_override
        self._judge_timeout = judge_timeout
        self._embeddings: dict[UUID, list[float]] = {}

    # --- 公開 ---------------------------------------------------------

    async def link_node(self, target: Node, store: GraphStore) -> list[Edge]:
        """``target`` と既存ノード群との関係を推定し、閾値超のエッジを書き込む。

        判定は ``asyncio.gather`` で並列に走らせる (候補ごとに独立)。リアルタイム
        運用では候補数を増やしてもレイテンシが一定になるため、source 別 top-k と
        相性が良い。

        個別の judge 呼び出しは ``_judge_pair_or_timeout`` で ``asyncio.wait_for``
        ラップし、1 件が hang してもバッチ全体を待たせないようにする。
        """

        candidates = await self._select_candidates(target, store)
        if not candidates:
            self.log.info(
                "linking.done",
                target_id=str(target.id),
                n_candidates=0,
                n_edges=0,
            )
            return []

        judgments = await asyncio.gather(
            *(self._judge_pair_or_timeout(target, cand) for cand in candidates)
        )

        edges: list[Edge] = []
        n_timeouts = 0
        for cand, judgment in zip(candidates, judgments, strict=True):
            if judgment.rationale.startswith("judge_timeout"):
                n_timeouts += 1
            edge = self._maybe_make_edge(target, cand, judgment)
            if edge is not None:
                store.add_edge(edge)
                edges.append(edge)

        self.log.info(
            "linking.done",
            target_id=str(target.id),
            n_candidates=len(candidates),
            n_edges=len(edges),
            n_timeouts=n_timeouts,
        )
        return edges

    async def _judge_pair_or_timeout(self, target: Node, cand: Node) -> _LinkJudgment:
        """``_judge_pair`` を ``asyncio.wait_for`` でラップして hang を防ぐ。

        ``self._judge_timeout`` が ``None`` のときは保護なしの素呼び出し。
        タイムアウト時は relation="none" / confidence=0 を返し、エッジは作られない。
        """

        if self._judge_timeout is None:
            return await self._judge_pair(target, cand)
        try:
            return await asyncio.wait_for(
                self._judge_pair(target, cand),
                timeout=self._judge_timeout,
            )
        except (TimeoutError, asyncio.TimeoutError):
            self.log.warning(
                "linking.judge_timeout",
                target_id=str(target.id),
                cand_id=str(cand.id),
                timeout_seconds=self._judge_timeout,
            )
            return _LinkJudgment(
                relation="none",
                confidence=0.0,
                rationale=f"judge_timeout_after_{self._judge_timeout}s",
            )

    # --- 候補選定 -----------------------------------------------------

    async def _select_candidates(
        self,
        target: Node,
        store: GraphStore,
    ) -> list[Node]:
        target_vec = await self._ensure_embedding(target)
        others = [n for n in store.nodes() if n.id != target.id]
        if not others:
            return []

        # 未キャッシュのものをまとめて 1 リクエストで埋め込み
        uncached = [n for n in others if n.id not in self._embeddings]
        if uncached:
            vectors = await self.llm.embed([n.text for n in uncached], model=self._embedding_model)
            for n, v in zip(uncached, vectors, strict=True):
                self._embeddings[n.id] = v

        if self._top_k_per_source is not None:
            return self._select_per_source(target_vec, others)

        # 既定: 混合 top-k
        scored = [(n, cosine_similarity(target_vec, self._embeddings[n.id])) for n in others]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [n for n, _ in scored[: self._top_k]]

    def _select_per_source(
        self,
        target_vec: list[float],
        others: list[Node],
    ) -> list[Node]:
        """source 別に cosine top-N を取り結合する。

        各バケット (utterance / document / web) 内で独立に類似度を並べるため、
        発話どうしの類似が高い議論でも文書/Web 枠が押し出されない。
        """

        assert self._top_k_per_source is not None
        buckets: dict[NodeSource, list[Node]] = {
            "utterance": [],
            "document": [],
            "web": [],
        }
        for n in others:
            buckets[n.source].append(n)

        candidates: list[Node] = []
        for nodes in buckets.values():
            if not nodes:
                continue
            scored = [(n, cosine_similarity(target_vec, self._embeddings[n.id])) for n in nodes]
            scored.sort(key=lambda pair: pair[1], reverse=True)
            candidates.extend(n for n, _ in scored[: self._top_k_per_source])
        return candidates

    async def _ensure_embedding(self, node: Node) -> list[float]:
        if node.id in self._embeddings:
            return self._embeddings[node.id]
        vec = await self.llm.embed_one(node.text, model=self._embedding_model)
        self._embeddings[node.id] = vec
        return vec

    # --- 判定 ---------------------------------------------------------

    async def _judge_pair(self, a: Node, b: Node) -> _LinkJudgment:
        user_content = (
            f"A:\n  text: {a.text}\n  source: {a.source}\n  author: {a.author}\n\n"
            f"B:\n  text: {b.text}\n  source: {b.source}\n  author: {b.author}"
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]
        return await self.llm.chat_structured(
            messages,  # type: ignore[arg-type]
            response_format=_LinkJudgment,
            model=self._model_override,
        )

    def _maybe_make_edge(self, target: Node, cand: Node, judgment: _LinkJudgment) -> Edge | None:
        if judgment.relation == "none":
            return None
        if judgment.confidence < self._threshold:
            return None

        relation: Relation
        # 連結エージェントの呼び出し慣習: a=target, b=candidate
        if judgment.relation == "a_supports_b":
            src_id, dst_id, relation = target.id, cand.id, "support"
        elif judgment.relation == "a_attacks_b":
            src_id, dst_id, relation = target.id, cand.id, "attack"
        elif judgment.relation == "b_supports_a":
            src_id, dst_id, relation = cand.id, target.id, "support"
        elif judgment.relation == "b_attacks_a":
            src_id, dst_id, relation = cand.id, target.id, "attack"
        else:  # pragma: no cover - 防御的
            return None

        return Edge(
            src_id=src_id,
            dst_id=dst_id,
            relation=relation,
            confidence=judgment.confidence,
            rationale=judgment.rationale,
            created_by="linking",
        )


__all__ = ["LinkingAgent", "cosine_similarity"]

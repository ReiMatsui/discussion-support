"""連結エージェント (本研究の中核)。

新しいノード ``target`` がストアに加わったとき:

  1. hybrid retrieval (embedding + BM25) で候補ノードを top-k 件に絞る (検索コスト削減)
  2. top-k 候補と ``target`` の論証関係を LLM に判定させる
       - 既定では候補をまとめて 1 回の呼び出しで判定し、ノードあたりの判定
         呼び出しを O(top_k)→O(1) に抑える (``_judge_batch``)
       - 5 値: a_supports_b / a_attacks_b / b_supports_a / b_attacks_a / none
  3. confidence が閾値以上のものだけエッジとしてストアに書き込む

埋め込みはエージェント内のメモリキャッシュで管理し、同じノードを 2 度
embed しないようにする。M1 段階では十分。

候補選定モード:
  - 既定 (``top_k_per_source=None``): 全 source 混合で cosine top-k
  - source 別 (``top_k_per_source=N``): utterance / document / web の各 source ごとに
    cosine top-N を取り、結合して候補とする (貢献①「異種ソースの統合」を構造的に保証)

Retrieval 品質ログ:
  - 各 ``link_node`` 呼び出しで、top-k 候補の判定結果 (support/attack/none の内訳)
    を ``RetrievalQualityLog`` として記録する。事後分析で retrieval が論証的関連を
    拾えているかを検証できる。
"""

from __future__ import annotations

import asyncio
import math
import re
from collections import Counter
from dataclasses import dataclass, field
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


class _IndexedJudgment(BaseModel):
    """バッチ判定で 1 候補ぶんの判定結果。``index`` で候補と対応付ける。"""

    index: int = Field(ge=0)
    relation: _RelationLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(default="")


class _BatchJudgment(BaseModel):
    """target と複数候補をまとめて判定した結果のリスト。"""

    judgments: list[_IndexedJudgment]


def _load_system_prompt() -> str:
    return (_PROMPTS_DIR / "linking.md").read_text(encoding="utf-8")


def _same_utterance(a: Node, b: Node) -> bool:
    """2 ノードが同一発話由来か (G2: 発話内ペアを linking 候補から除外する)。"""

    if a.source != "utterance" or b.source != "utterance":
        return False
    a_turn = a.metadata.get("turn_id")
    b_turn = b.metadata.get("turn_id")
    if a_turn is not None and b_turn is not None:
        return bool(a_turn == b_turn and a.author == b.author)
    return False


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


# --- BM25 (軽量インメモリ実装) -----------------------------------------

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Unicode 単語トークナイザ (日本語は文字 unigram にフォールバック)。"""
    tokens = _TOKEN_RE.findall(text.lower())
    # 日本語テキスト: 1文字の漢字/ひらがな/カタカナが多い場合は文字分割を追加
    if not tokens or (len(tokens) == 1 and len(text) > 4):
        tokens = list(text.lower().replace(" ", ""))
    return tokens


def bm25_score(
    query_tokens: list[str],
    doc_tokens: list[str],
    avg_dl: float,
    df: dict[str, int],
    n_docs: int,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """単一文書に対する BM25 スコア。"""
    if not query_tokens or not doc_tokens or n_docs == 0:
        return 0.0
    dl = len(doc_tokens)
    doc_tf: Counter[str] = Counter(doc_tokens)
    score = 0.0
    for term in query_tokens:
        if term not in doc_tf:
            continue
        tf = doc_tf[term]
        d = df.get(term, 0)
        idf = math.log((n_docs - d + 0.5) / (d + 0.5) + 1.0)
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * dl / max(avg_dl, 1.0))
        score += idf * numerator / denominator
    return score


# --- Retrieval 品質ログ -------------------------------------------------


@dataclass
class RetrievalQualityEntry:
    """1 回の link_node 呼び出しで得られた retrieval 品質情報。"""

    target_id: str
    target_text: str
    target_source: str
    n_candidates: int
    n_support: int = 0
    n_attack: int = 0
    n_none: int = 0
    retrieval_mode: str = "hybrid"
    """'embedding' | 'bm25' | 'hybrid' | 'per_source'"""
    candidate_sources: dict[str, int] = field(default_factory=dict)
    """source 別の候補数内訳 (utterance/document/web)。"""


@dataclass
class RetrievalQualityLog:
    """セッション全体の retrieval 品質記録。事後分析用。"""

    entries: list[RetrievalQualityEntry] = field(default_factory=list)

    @property
    def total_candidates(self) -> int:
        return sum(e.n_candidates for e in self.entries)

    @property
    def total_support(self) -> int:
        return sum(e.n_support for e in self.entries)

    @property
    def total_attack(self) -> int:
        return sum(e.n_attack for e in self.entries)

    @property
    def total_none(self) -> int:
        return sum(e.n_none for e in self.entries)

    @property
    def hit_rate(self) -> float:
        """top-k のうち support or attack と判定された割合。"""
        total = self.total_candidates
        if total == 0:
            return 0.0
        return (self.total_support + self.total_attack) / total

    @property
    def attack_ratio(self) -> float:
        """有効候補 (support+attack) のうち attack の割合。反論の拾い具合の指標。"""
        hits = self.total_support + self.total_attack
        if hits == 0:
            return 0.0
        return self.total_attack / hits

    def summary(self) -> dict:
        """集計サマリ辞書。"""
        return {
            "n_link_calls": len(self.entries),
            "total_candidates": self.total_candidates,
            "total_support": self.total_support,
            "total_attack": self.total_attack,
            "total_none": self.total_none,
            "hit_rate": round(self.hit_rate, 4),
            "attack_ratio": round(self.attack_ratio, 4),
        }


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
        batch: bool = True,
        hybrid_alpha: float = 0.7,
        cluster_threshold: float = 0.9,
        active_window: int = 12,
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
                バッチモードでは「バッチ呼び出し 1 回」全体に対するタイムアウト。
            batch: True (既定) なら top-k 候補を 1 回の LLM 呼び出しでまとめて判定する
                (``_judge_batch``)。ノードあたりの関係判定呼び出しを O(top_k)→O(1) に
                抑え、リアルタイム運用のコスト/レート制限を削減する。False なら従来の
                候補 1 件 1 呼び出し (``_judge_pair`` を ``asyncio.gather`` で並列) に戻す。
            hybrid_alpha: embedding スコアと BM25 スコアの混合比率 (0-1)。
                ``alpha * embedding + (1-alpha) * bm25_normalized`` で最終スコアを算出する。
                alpha=1.0 で従来の embedding-only、alpha=0.0 で BM25-only。
                既定 0.7 (embedding 重視だが BM25 でキーワード一致も考慮)。
                BM25 を混ぜることで、embedding では遠いが論証的に関連する候補
                (反論など意味が対立するもの) を拾いやすくする。
            cluster_threshold: soft-merge の cosine 類似度しきい値 (既定 0.9)。既存 claim と
                これ以上似た新 claim を同クラスタに合流させる (logic_review A2, 非破壊)。
            active_window: claim↔claim の照合をアクティブ窓内 (turn_index 基準) に限定する
                窓幅 (既定 12, facilitation と揃える, G6)。発話側ノードが増えても照合対象を
                直近ターンに絞ってレイテンシを抑える。evidence↔claim は窓で切らない
                (古い文書知識と新しい主張のリンクは価値があるため)。
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
        self._batch = batch
        self._hybrid_alpha = hybrid_alpha
        self._cluster_threshold = cluster_threshold
        self._active_window = active_window
        self._embeddings: dict[UUID, list[float]] = {}
        self._quality_log = RetrievalQualityLog()
        # G1: evidence↔claim の方向制約で正規化/破棄した件数 (プロンプト遵守率の観測)
        self._n_edges_direction_normalized = 0
        self._n_edges_dropped_evidence_pair = 0

    # --- 公開 ---------------------------------------------------------


    @property
    def embeddings(self) -> dict[UUID, list[float]]:
        """計算済みノード embedding のキャッシュ (node_id -> ベクトル)。読み取り用。"""
        return self._embeddings

    async def link_node(self, target: Node, store: GraphStore) -> list[Edge]:
        """``target`` と既存ノード群との関係を推定し、閾値超のエッジを書き込む。

        既定 (``batch=True``) では top-k 候補を **1 回の LLM 呼び出し**でまとめて
        判定する (``_judge_batch``)。ノードあたりの関係判定呼び出しを O(top_k)→O(1)
        に抑える。``asyncio.wait_for`` はバッチ呼び出し 1 回全体に対してラップし、
        超過時は全候補 none 扱いとする。

        ``batch=False`` のときは従来挙動 (候補ごとに ``_judge_pair_or_timeout`` を
        ``asyncio.gather`` で並列実行) に戻る。
        """

        candidates = await self._select_candidates(target, store)
        # soft-merge: 候補選定で全ノードの embedding が cache 済みなので、それを
        # 再利用して同義 claim を同クラスタに割り当てる (logic_review A2, 非破壊)。
        self._maybe_assign_cluster(target, store)
        if not candidates:
            self.log.info(
                "linking.done",
                target_id=str(target.id),
                n_candidates=0,
                n_edges=0,
            )
            return []

        if self._batch:
            judgments: list[_LinkJudgment] = await self._judge_batch_or_timeout(
                target, candidates
            )
        else:
            judgments = list(
                await asyncio.gather(
                    *(self._judge_pair_or_timeout(target, cand) for cand in candidates)
                )
            )

        edges: list[Edge] = []
        n_timeouts = 0
        n_support = 0
        n_attack = 0
        n_none = 0
        for cand, judgment in zip(candidates, judgments, strict=True):
            if judgment.rationale.startswith("judge_timeout"):
                n_timeouts += 1
            if judgment.relation == "none":
                n_none += 1
            elif "support" in judgment.relation:
                n_support += 1
            elif "attack" in judgment.relation:
                n_attack += 1
            edge = self._maybe_make_edge(target, cand, judgment)
            if edge is not None:
                store.add_edge(edge)
                edges.append(edge)

        # Retrieval 品質ログ
        source_counts: dict[str, int] = {}
        for c in candidates:
            source_counts[c.source] = source_counts.get(c.source, 0) + 1
        retrieval_mode = "per_source" if self._top_k_per_source else "hybrid"
        self._quality_log.entries.append(
            RetrievalQualityEntry(
                target_id=str(target.id),
                target_text=target.text[:100],
                target_source=target.source,
                n_candidates=len(candidates),
                n_support=n_support,
                n_attack=n_attack,
                n_none=n_none,
                retrieval_mode=retrieval_mode,
                candidate_sources=source_counts,
            )
        )

        self.log.info(
            "linking.done",
            target_id=str(target.id),
            n_candidates=len(candidates),
            n_edges=len(edges),
            n_timeouts=n_timeouts,
            n_support=n_support,
            n_attack=n_attack,
            n_none=n_none,
        )
        return edges

    async def _judge_batch_or_timeout(
        self, target: Node, candidates: list[Node]
    ) -> list[_LinkJudgment]:
        """``_judge_batch`` を ``asyncio.wait_for`` でラップして hang を防ぐ。

        タイムアウトは「バッチ呼び出し 1 回」全体に対して適用する。超過時は全候補を
        relation="none" / confidence=0 で返し (既存の per-pair timeout 挙動と整合)、
        エッジは作られない。``self._judge_timeout`` が ``None`` のときは素呼び出し。
        """

        if self._judge_timeout is None:
            return await self._judge_batch(target, candidates)
        try:
            return await asyncio.wait_for(
                self._judge_batch(target, candidates),
                timeout=self._judge_timeout,
            )
        except TimeoutError:
            self.log.warning(
                "linking.judge_timeout",
                target_id=str(target.id),
                n_candidates=len(candidates),
                timeout_seconds=self._judge_timeout,
            )
            return [
                _LinkJudgment(
                    relation="none",
                    confidence=0.0,
                    rationale=f"judge_timeout_after_{self._judge_timeout}s",
                )
                for _ in candidates
            ]

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
        except TimeoutError:
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
        # G2: 発話内ペアは extraction が既にエッジ化済みなので linking では再判定しない
        # G6: claim↔claim はアクティブ窓内のみ照合 (evidence↔claim は窓で切らない)
        others = [
            n
            for n in store.nodes()
            if n.id != target.id
            and not _same_utterance(n, target)
            and self._in_link_window(n, target)
        ]
        if not others:
            return []

        # 未キャッシュのものをまとめて 1 リクエ���トで埋め込み
        uncached = [n for n in others if n.id not in self._embeddings]
        if uncached:
            vectors = await self.llm.embed([n.text for n in uncached], model=self._embedding_model)
            for n, v in zip(uncached, vectors, strict=True):
                self._embeddings[n.id] = v

        if self._top_k_per_source is not None:
            return self._select_per_source(target_vec, others)

        # --- Hybrid retrieval (embedding + BM25) ---
        # Embedding スコア
        emb_scores = [
            (n, cosine_similarity(target_vec, self._embeddings[n.id])) for n in others
        ]

        alpha = self._hybrid_alpha
        if alpha >= 1.0:
            # Pure embedding mode (従来互換)
            emb_scores.sort(key=lambda pair: pair[1], reverse=True)
            return [n for n, _ in emb_scores[: self._top_k]]

        # BM25 スコア
        query_tokens = _tokenize(target.text)
        doc_tokens_list = [_tokenize(n.text) for n in others]
        avg_dl = sum(len(t) for t in doc_tokens_list) / max(len(doc_tokens_list), 1)
        # DF 計算
        df: dict[str, int] = {}
        for tokens in doc_tokens_list:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1
        n_docs = len(others)
        bm25_scores = [
            bm25_score(query_tokens, doc_tokens, avg_dl, df, n_docs)
            for doc_tokens in doc_tokens_list
        ]

        # 正規化 (0-1) して混合
        emb_max = max((s for _, s in emb_scores), default=1.0) or 1.0
        bm25_max = max(bm25_scores, default=1.0) or 1.0

        hybrid_scored: list[tuple[Node, float]] = []
        for i, (node, emb_s) in enumerate(emb_scores):
            emb_norm = emb_s / emb_max
            bm25_norm = bm25_scores[i] / bm25_max
            combined = alpha * emb_norm + (1 - alpha) * bm25_norm
            hybrid_scored.append((node, combined))

        hybrid_scored.sort(key=lambda pair: pair[1], reverse=True)
        return [n for n, _ in hybrid_scored[: self._top_k]]

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

    def _in_link_window(self, candidate: Node, target: Node) -> bool:
        """候補を linking 対象にしてよいか (G6: claim↔claim のアクティブ窓絞り込み)。

        - evidence が絡むペア (evidence↔claim) は窓で切らない。古い文書/Web 知識と
          新しい主張のリンクは価値があるため、従来どおり embedding top-k 選抜に任せる。
        - 発話側どうし (claim/premise ↔ claim/premise) は、target の turn_index を
          現在ターンとしたアクティブ窓 (幅 active_window) の中だけを照合対象にする。
        """

        if candidate.node_type == "evidence" or target.node_type == "evidence":
            return True
        if candidate.source != "utterance" or target.source != "utterance":
            return True
        window_start = target.turn_index - self._active_window + 1
        return candidate.turn_index >= window_start

    def _maybe_assign_cluster(self, target: Node, store: GraphStore) -> None:
        """``target`` が既存 claim と十分に類似していれば同クラスタに合流させる。

        logic_review A2 の soft-merge。同義主張が言い直されるたびに別ノードになる
        問題を、非破壊 (ノードは残す) にクラスタ単位でまとめて指標を数えられるように
        する。claim ↔ claim のみ対象、``_select_candidates`` が cache 済みの embedding を
        再利用するので追加の埋め込み呼び出しは無い。
        """

        if target.node_type != "claim":
            return
        target_vec = self._embeddings.get(target.id)
        if target_vec is None:
            return

        best_id: UUID | None = None
        best_sim = 0.0
        for n in store.nodes():
            if n.id == target.id or n.node_type != "claim":
                continue
            vec = self._embeddings.get(n.id)
            if vec is None:
                continue
            sim = cosine_similarity(target_vec, vec)
            if sim > best_sim:
                best_sim = sim
                best_id = n.id

        if best_id is not None and best_sim >= self._cluster_threshold:
            rep = store.cluster_of(best_id)
            store.assign_cluster(target.id, rep)
            self.log.info(
                "linking.cluster_merge",
                target_id=str(target.id),
                cluster_rep=str(rep),
                similarity=round(best_sim, 3),
            )

    async def _ensure_embedding(self, node: Node) -> list[float]:
        if node.id in self._embeddings:
            return self._embeddings[node.id]
        vec = await self.llm.embed_one(node.text, model=self._embedding_model)
        self._embeddings[node.id] = vec
        return vec

    # --- 判定 ---------------------------------------------------------

    async def _judge_batch(
        self, target: Node, candidates: list[Node]
    ) -> list[_LinkJudgment]:
        """target と全候補の関係を 1 回の LLM 呼び出しでまとめて判定する。

        候補を ``[0] … [k-1]`` と番号付けして渡し、各候補 (B) と target (A) の
        関係を candidate ごとに返させる。判定の意味づけ・5 値 relation の向きは
        ``_judge_pair`` と同一 (A=target, B=candidate)。

        LLM が一部候補を省略したり順序が入れ替わっても、``index`` で対応付け、
        欠けた分は ``relation="none" / confidence=0`` で補完して candidate 数と
        アラインさせる (順序ズレ・欠落に頑健)。
        """

        lines = [
            "A (基準ノード):",
            f"  text: {target.text}",
            f"  type: {target.node_type}",
            f"  source: {target.source}",
            f"  author: {target.author}",
            "",
            "B 候補ノード群 (各 [i] について A との関係を 5 値で判定):",
        ]
        for i, cand in enumerate(candidates):
            lines.append(f"[{i}]")
            lines.append(f"  text: {cand.text}")
            lines.append(f"  type: {cand.node_type}")
            lines.append(f"  source: {cand.source}")
            lines.append(f"  author: {cand.author}")
        user_content = "\n".join(lines)

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]
        batch = await self.llm.chat_structured(
            messages,  # type: ignore[arg-type]
            response_format=_BatchJudgment,
            model=self._model_override,
        )

        # 既定 none で埋めてから index で上書き (欠落・順序ズレ・重複・範囲外に頑健)
        judgments = [
            _LinkJudgment(relation="none", confidence=0.0, rationale="")
            for _ in candidates
        ]
        for item in batch.judgments:
            if 0 <= item.index < len(candidates):
                judgments[item.index] = _LinkJudgment(
                    relation=item.relation,
                    confidence=item.confidence,
                    rationale=item.rationale,
                )
        return judgments

    async def _judge_pair(self, a: Node, b: Node) -> _LinkJudgment:
        user_content = (
            f"A:\n  text: {a.text}\n  type: {a.node_type}\n  source: {a.source}\n  author: {a.author}\n\n"
            f"B:\n  text: {b.text}\n  type: {b.node_type}\n  source: {b.source}\n  author: {b.author}"
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
            src_node, dst_node, relation = target, cand, "support"
        elif judgment.relation == "a_attacks_b":
            src_node, dst_node, relation = target, cand, "attack"
        elif judgment.relation == "b_supports_a":
            src_node, dst_node, relation = cand, target, "support"
        elif judgment.relation == "b_attacks_a":
            src_node, dst_node, relation = cand, target, "attack"
        else:  # pragma: no cover - 防御的
            return None

        # G1 (レビュー C-2): evidence↔claim の方向制約をコードで強制する。
        # 設計不変条件「evidence が常に src (事実が主張を支持/攻撃する)」を、
        # LLM の向き取り違えに対して正規化する。プロンプト任せにしない。
        src_is_evidence = src_node.node_type == "evidence"
        dst_is_evidence = dst_node.node_type == "evidence"
        if src_is_evidence and dst_is_evidence:
            # evidence↔evidence は設計上エッジを張らない (中立事実同士に立場はない)
            self._n_edges_dropped_evidence_pair += 1
            self.log.info(
                "linking.edge_dropped_evidence_pair",
                src_id=str(src_node.id),
                dst_id=str(dst_node.id),
            )
            return None
        if dst_is_evidence and not src_is_evidence:
            # claim→evidence 判定を evidence→claim に反転 (relation は保持)。
            # 意味論を「事実が主張を支持/攻撃する」に固定する。
            src_node, dst_node = dst_node, src_node
            self._n_edges_direction_normalized += 1
            self.log.info(
                "linking.edge_direction_normalized",
                evidence_id=str(src_node.id),
                claim_id=str(dst_node.id),
                relation=relation,
            )

        return Edge(
            src_id=src_node.id,
            dst_id=dst_node.id,
            relation=relation,
            confidence=judgment.confidence,
            rationale=judgment.rationale,
            created_by="linking",
        )


__all__ = [
    "LinkingAgent",
    "RetrievalQualityEntry",
    "RetrievalQualityLog",
    "bm25_score",
    "cosine_similarity",
]

"""ファシリテーションエージェント (中央調停者)。

研究計画書 §4.1 のファシリテーションエージェントに対応する実装。

責務:
  1. 統合議論グラフ全体を読みながら、**いつ・誰に・何を提示するか**を判断
  2. 介入が不要なときは黙る (= skip)
  3. 個別通知 (L1) と俯瞰サマリ (L2) を切り替える

設計上の選択:
  - 介入の要否判断 (decide_intervention) はグラフ状態のみに基づく
    (テキスト重複や round-robin の話者順番には依存しない)
  - 出力 ``InterventionDecision`` は配信チャネル非依存 (シミュレーション・対面・
    音声いずれでも、内容と宛先がそのまま使える)
  - L2 ブリーフの自然文整形は LLM を使うが、LLM が居ない / 失敗した場合の
    deterministic fallback も持つ
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from uuid import UUID

from das.agents.base import BaseAgent
from das.graph.schema import Node, NodeSource
from das.graph.store import GraphStore
from das.llm import OpenAIClient
from das.types import Utterance

Stage = Literal["diverge", "converge", "stalled"]
DecisionKind = Literal["skip", "l1", "l2"]


@dataclass(frozen=True)
class BiasReport:
    """グラフ全体の支持/攻撃の偏り報告。"""

    n_support: int
    n_attack: int
    dominant_side: Literal["support", "attack", "balanced"]
    weak_claims: list[Node] = field(default_factory=list)
    """攻撃を 2 件以上受けて支持が無い発話 claim ノード。"""

    over_supported_claims: list[Node] = field(default_factory=list)
    """支持を 2 件以上受けて反論が無い発話 claim ノード。"""

    @property
    def imbalance_ratio(self) -> float:
        """0 = 完全均衡, 1 = 完全に片寄り。"""

        total = self.n_support + self.n_attack
        if total == 0:
            return 0.0
        return abs(self.n_support - self.n_attack) / total


@dataclass(frozen=True)
class StageReport:
    """議論ステージの判定結果 (グラフ状態ベース)。"""

    stage: Stage
    n_recent_utterances: int
    new_claims_in_window: int
    """直近窓で追加された発話 claim ノード数。stalled 判定の主シグナル。"""

    new_attacks_in_window: int
    """直近窓で追加された attack エッジ数。"""

    speaker_diversity: float
    """直近窓のユニーク話者比 (0..1)。発散の補助指標。"""


@dataclass(frozen=True)
class InfoItem:
    """ファシリテーションが提示する 1 件の情報。"""

    relation: Literal["support", "attack"]
    target_text: str
    target_speaker: str | None
    source_text: str
    source_kind: NodeSource
    source_author: str | None
    confidence: float
    rationale: str = ""
    priority: float = 1.0
    """0..1 の表示優先度。偏り補正やステージ補正で調整される。"""

    reason: Literal["adjacent", "balance_correction", "stage_alignment"] = "adjacent"
    """この項目が選ばれた論理的理由 (介入の透明性)。"""


@dataclass(frozen=True)
class InterventionDecision:
    """ファシリテータの介入判断結果。配信チャネル非依存。"""

    kind: DecisionKind
    items: list[InfoItem] = field(default_factory=list)
    """L1 (個別通知) のときに提示する 1〜2 件の関連エッジ情報。"""

    brief: str = ""
    """L2 (俯瞰サマリ) のときに全員へ提示する自然文。"""

    addressed_to: str | None = None
    """L1 の通知対象話者名。L2 や skip では None (= 全員 / 該当なし)。"""

    reason: str = ""
    """この判断に至ったロジカルな理由 (透明性のため log にも記録)。"""


class FacilitationAgent(BaseAgent):
    """中央調停者。グラフ全体を読み、介入の要否と内容を返す。"""

    name = "facilitation"

    def __init__(
        self,
        llm: OpenAIClient | None = None,
        *,
        max_items: int = 2,
        recent_window: int = 4,
        bias_threshold: float = 0.4,
        stall_window: int = 4,
        stall_max_new_claims: int = 1,
        stall_max_new_attacks: int = 0,
        l2_min_interval: int = 5,
        diverge_diversity_threshold: float = 0.75,
    ) -> None:
        """
        Parameters:
          - ``max_items``: L1 で提示する最大件数 (研究計画書「1〜2 件」に従い既定 2)
          - ``recent_window``: stage 判定で見る直近発話数
          - ``bias_threshold``: imbalance_ratio がこの値を超えたら偏り
          - ``stall_window``: stalled 判定で見る直近発話数
          - ``stall_max_new_claims``: その窓で新 claim がこれ以下なら停滞シグナル
          - ``stall_max_new_attacks``: その窓で新 attack がこれ以下なら停滞シグナル
          - ``l2_min_interval``: L2 を出してから次の L2 まで最低何発話空けるか
          - ``diverge_diversity_threshold``: speaker_diversity がこれ以上なら発散判定
        """

        super().__init__(llm=llm)
        self._max_items = max_items
        self._recent_window = recent_window
        self._bias_threshold = bias_threshold
        self._stall_window = stall_window
        self._stall_max_new_claims = stall_max_new_claims
        self._stall_max_new_attacks = stall_max_new_attacks
        self._l2_min_interval = l2_min_interval
        self._diverge_diversity_threshold = diverge_diversity_threshold

        # 内部状態 (skip 判定用)
        self._last_decision_kind: DecisionKind | None = None
        self._n_utterances_at_last_decision: int = 0
        self._n_edges_at_last_decision: int = 0
        self._n_utterances_at_last_l2: int = -10**9  # 起動直後でも L2 が出せる

    def reset(self) -> None:
        """セッション切替時に内部状態をリセットする。"""

        self._last_decision_kind = None
        self._n_utterances_at_last_decision = 0
        self._n_edges_at_last_decision = 0
        self._n_utterances_at_last_l2 = -10**9

    # --- 偏り検知 (グラフ状態のみに依存) -------------------------------

    def detect_bias(self, store: GraphStore) -> BiasReport:
        """グラフ全体の支持/攻撃の偏りを集計する。"""

        edges = list(store.edges())
        n_support = sum(1 for e in edges if e.relation == "support")
        n_attack = sum(1 for e in edges if e.relation == "attack")

        per_node: dict[UUID, Counter[str]] = {}
        for edge in edges:
            counter = per_node.setdefault(edge.dst_id, Counter())
            counter[edge.relation] += 1

        weak: list[Node] = []
        over: list[Node] = []
        for node_id, counts in per_node.items():
            node = store.get_node(node_id)
            if node is None or node.source != "utterance":
                continue
            if counts.get("attack", 0) >= 2 and counts.get("support", 0) == 0:
                weak.append(node)
            elif counts.get("support", 0) >= 2 and counts.get("attack", 0) == 0:
                over.append(node)

        dominant: Literal["support", "attack", "balanced"]
        if n_support == 0 and n_attack == 0:
            dominant = "balanced"
        elif n_support > n_attack:
            dominant = "support"
        elif n_attack > n_support:
            dominant = "attack"
        else:
            dominant = "balanced"

        return BiasReport(
            n_support=n_support,
            n_attack=n_attack,
            dominant_side=dominant,
            weak_claims=weak,
            over_supported_claims=over,
        )

    # --- ステージ検知 (グラフ状態ベース) -------------------------------

    def detect_stage(
        self, transcript: list[Utterance], store: GraphStore | None = None
    ) -> StageReport:
        """発話の進み方からステージを推定する。

        ステージはテキスト重複ではなく **AF への新規追加レート** を見る:
          - stalled: 直近窓で新 claim も新 attack もほとんど追加されていない
          - diverge: 話者多様性が高く、新 claim が活発
          - converge: その他 (典型的な収束フェーズ)
        """

        recent = transcript[-self._recent_window :] if transcript else []
        n = len(recent)
        if n == 0:
            return StageReport(
                stage="diverge",
                n_recent_utterances=0,
                new_claims_in_window=0,
                new_attacks_in_window=0,
                speaker_diversity=0.0,
            )

        diversity = len({u.speaker for u in recent}) / n

        new_claims = 0
        new_attacks = 0
        if store is not None:
            new_claims, new_attacks = self._count_recent_additions(recent, store)

        stage: Stage
        # stalled はグラフ状態に依存するため store 必須
        if (
            store is not None
            and new_claims <= self._stall_max_new_claims
            and new_attacks <= self._stall_max_new_attacks
        ):
            stage = "stalled"
        elif diversity >= self._diverge_diversity_threshold and (
            store is None or new_claims >= 2
        ):
            # store 不明のときは話者多様性のみで暫定判定 (Live UI など)
            stage = "diverge"
        else:
            stage = "converge"

        return StageReport(
            stage=stage,
            n_recent_utterances=n,
            new_claims_in_window=new_claims,
            new_attacks_in_window=new_attacks,
            speaker_diversity=diversity,
        )

    def _count_recent_additions(
        self, recent_utts: list[Utterance], store: GraphStore
    ) -> tuple[int, int]:
        """直近発話に対応する新 claim ノード数 / 新 attack エッジ数。

        対応キーは ``timestamp`` を使う (turn_id はシミュレーション固有で
        対面では存在しない可能性があるため避ける)。
        """

        if not recent_utts:
            return 0, 0
        window_start: datetime = recent_utts[0].timestamp

        utterance_nodes_in_window: set[UUID] = set()
        n_claims = 0
        for node in store.nodes():
            if node.source != "utterance":
                continue
            if node.timestamp >= window_start:
                utterance_nodes_in_window.add(node.id)
                if node.node_type == "claim":
                    n_claims += 1

        if not utterance_nodes_in_window:
            return n_claims, 0

        n_attacks = sum(
            1
            for e in store.edges()
            if e.relation == "attack"
            and (e.src_id in utterance_nodes_in_window or e.dst_id in utterance_nodes_in_window)
        )
        return n_claims, n_attacks

    # --- 介入判断 (Stage 1: いつ介入するか) ----------------------------

    def decide_intervention(
        self, transcript: list[Utterance], store: GraphStore
    ) -> InterventionDecision:
        """グラフ状態から介入の要否と種類を決定する。

        判定優先度: SKIP → L2 → L1 (L2 優先で俯瞰、L1 はデフォルト) 。
        """

        n_utts = len(transcript)
        n_edges = sum(1 for _ in store.edges())

        # --- SKIP 判定 -----------------------------------------------------

        if n_utts == 0:
            decision = InterventionDecision(kind="skip", reason="履歴なし")
            self._record(decision, n_utts, n_edges)
            return decision

        # 直近の発話に対応するノードがまだ無い場合 (extraction 待ち) は黙る
        last_utt = transcript[-1]
        last_utt_nodes = self._nodes_for_utterance(last_utt, store)
        if not last_utt_nodes:
            decision = InterventionDecision(
                kind="skip", reason="最新発話のノード化が未完了"
            )
            self._record(decision, n_utts, n_edges)
            return decision

        # 前回介入してから新エッジが増えていなければ skip (連続介入の抑制)
        if (
            self._last_decision_kind in ("l1", "l2")
            and n_edges == self._n_edges_at_last_decision
            and n_utts == self._n_utterances_at_last_decision + 1
        ):
            decision = InterventionDecision(
                kind="skip",
                reason="直前介入後に新エッジ追加なし (連続介入の抑制)",
            )
            self._record(decision, n_utts, n_edges)
            return decision

        # --- L2 判定 (俯瞰サマリの必要性) -----------------------------

        bias = self.detect_bias(store)
        stage = self.detect_stage(transcript, store)

        # L2 を出してからの間隔 + 最低限の発話数を満たしている必要がある
        utts_since_last_l2 = n_utts - self._n_utterances_at_last_l2
        l2_eligible = (
            n_utts >= self._stall_window
            and utts_since_last_l2 >= self._l2_min_interval
        )

        l2_triggers: list[str] = []
        if l2_eligible:
            if stage.stage == "stalled":
                l2_triggers.append(
                    f"停滞 (直近 {stage.n_recent_utterances} 発話で新 claim "
                    f"{stage.new_claims_in_window} 件 / 新 attack {stage.new_attacks_in_window} 件)"
                )
            if (
                bias.imbalance_ratio > self._bias_threshold
                and len(bias.weak_claims) + len(bias.over_supported_claims) >= 1
            ):
                l2_triggers.append(
                    f"構造的偏り (imbalance={bias.imbalance_ratio:.2f}, "
                    f"weak={len(bias.weak_claims)}, over={len(bias.over_supported_claims)})"
                )

        if l2_triggers:
            brief = self._compose_l2_brief_or_fallback(
                transcript=transcript,
                store=store,
                bias=bias,
                stage=stage,
            )
            if brief:
                decision = InterventionDecision(
                    kind="l2",
                    brief=brief,
                    addressed_to=None,  # 全員
                    reason="; ".join(l2_triggers),
                )
                self._record(decision, n_utts, n_edges, is_l2=True)
                return decision

        # --- L1 判定 (個別通知のデフォルト経路) ------------------------

        items: list[InfoItem] = []
        seen: set[str] = set()
        for node in last_utt_nodes:
            for item in self._select_for_target(node, store, bias=bias, stage=stage):
                key = f"{item.relation}|{item.source_text}|{item.target_text}"
                if key in seen:
                    continue
                seen.add(key)
                items.append(item)

        items.sort(key=lambda it: it.priority, reverse=True)
        items = items[: self._max_items]

        if not items:
            decision = InterventionDecision(
                kind="skip", reason="最新発話に未提示の隣接エッジなし"
            )
            self._record(decision, n_utts, n_edges)
            return decision

        decision = InterventionDecision(
            kind="l1",
            items=items,
            addressed_to=last_utt.speaker,
            reason=(
                f"L1: 最新発話への隣接 {len(items)} 件 / "
                f"stage={stage.stage} bias={bias.imbalance_ratio:.2f}"
            ),
        )
        self._record(decision, n_utts, n_edges)
        return decision

    def _record(
        self,
        decision: InterventionDecision,
        n_utts: int,
        n_edges: int,
        *,
        is_l2: bool = False,
    ) -> None:
        self._last_decision_kind = decision.kind
        self._n_utterances_at_last_decision = n_utts
        self._n_edges_at_last_decision = n_edges
        if is_l2:
            self._n_utterances_at_last_l2 = n_utts
        self.log.info(
            "facilitation.decided",
            kind=decision.kind,
            addressed_to=decision.addressed_to,
            reason=decision.reason,
        )

    @staticmethod
    def _nodes_for_utterance(utt: Utterance, store: GraphStore) -> list[Node]:
        """ある発話から派生したノード群を取り出す。

        マッチ規則は冗長で頑健に:
          1. 抽出時に保存された ``metadata.turn_id`` (= シミュレーションのキー)
          2. ``timestamp + author`` の組 (= 対面・音声のフォールバック)

        どちらか一方でも一致すれば採用する。
        """

        matched: list[Node] = []
        for n in store.nodes():
            if n.source != "utterance":
                continue
            meta_turn_id = n.metadata.get("turn_id")
            if meta_turn_id is not None and meta_turn_id == utt.turn_id:
                matched.append(n)
                continue
            if n.timestamp == utt.timestamp and n.author == utt.speaker:
                matched.append(n)
        return matched

    # --- L2 ブリーフ生成 (LLM + deterministic fallback) -----------------

    async def compose_l2_brief(
        self,
        transcript: list[Utterance],
        store: GraphStore,
        *,
        bias: BiasReport | None = None,
        stage: StageReport | None = None,
    ) -> str:
        """L2 用の俯瞰サマリを LLM で生成する (失敗時は deterministic fallback)。

        ``decide_intervention`` 経由で呼ばれる場合は同期パスから
        ``_compose_l2_brief_or_fallback`` を経由するため、これは外部呼び出し
        (例: Live UI から手動でサマリを生成したい場合) のための公開 API。
        """

        bias = bias or self.detect_bias(store)
        stage = stage or self.detect_stage(transcript, store)
        if self.llm is None:
            return self._compose_l2_brief_deterministic(transcript, store, bias, stage)
        try:
            return await self._compose_l2_brief_llm(transcript, store, bias, stage)
        except Exception as exc:  # pragma: no cover - 防御的
            self.log.warning(
                "facilitation.l2_brief.llm_failed",
                error=str(exc),
            )
            return self._compose_l2_brief_deterministic(transcript, store, bias, stage)

    def _compose_l2_brief_or_fallback(
        self,
        *,
        transcript: list[Utterance],
        store: GraphStore,
        bias: BiasReport,
        stage: StageReport,
    ) -> str:
        """同期コンテキスト用。LLM は使わず deterministic で組み立てる。

        ``decide_intervention`` は同期 API として残したいので、こちらは
        deterministic 専用。LLM を使った整文をしたい呼び出し側は別パスで
        ``compose_l2_brief`` を await すること。
        """

        return self._compose_l2_brief_deterministic(transcript, store, bias, stage)

    def _compose_l2_brief_deterministic(
        self,
        transcript: list[Utterance],
        store: GraphStore,
        bias: BiasReport,
        stage: StageReport,
    ) -> str:
        """LLM-free の俯瞰サマリ。グラフ集計を自然文テンプレに流す。"""

        lines: list[str] = []
        lines.append(
            f"ここまでの整理: 支持 {bias.n_support} 件 / 反論 {bias.n_attack} 件、"
            f"ステージ={stage.stage}"
        )
        if bias.weak_claims:
            samples = ", ".join(
                f"『{n.text[:40]}{'…' if len(n.text) > 40 else ''}』" for n in bias.weak_claims[:2]
            )
            lines.append(f"未応答の反論を受けたままの主張: {samples}")
        if bias.over_supported_claims:
            samples = ", ".join(
                f"『{n.text[:40]}{'…' if len(n.text) > 40 else ''}』"
                for n in bias.over_supported_claims[:2]
            )
            lines.append(f"反論されないまま支持を集めている主張: {samples}")
        if stage.stage == "stalled":
            lines.append(
                "新しい論点が出ていません。具体例や数値で論点を一段深掘りするか、"
                "未応答の反論に応答してから先に進むことを検討してください。"
            )
        return "\n".join(lines)

    async def _compose_l2_brief_llm(
        self,
        transcript: list[Utterance],
        store: GraphStore,
        bias: BiasReport,
        stage: StageReport,
    ) -> str:
        """LLM で 2〜3 文の自然な俯瞰サマリに整える。"""

        det = self._compose_l2_brief_deterministic(transcript, store, bias, stage)
        recent_lines = "\n".join(
            f"- {u.speaker}: {u.text[:120]}{'…' if len(u.text) > 120 else ''}"
            for u in transcript[-self._recent_window :]
        )
        prompt = (
            "あなたは議論のファシリテーターです。以下のグラフ状態と直近発言を踏まえ、"
            "参加者全員に向けた俯瞰整理を 2〜3 文の自然な日本語で書いてください。"
            "新しい主張は加えず、既に出た論点・対立・未応答の反論を整理することに専念してください。\n\n"
            f"## グラフ状態\n{det}\n\n"
            f"## 直近の発言\n{recent_lines}\n\n"
            "出力フォーマット: 「ここまでの整理:」で始まる 2〜3 文。改行可。"
        )
        if self.llm is None:  # pragma: no cover - guarded by caller
            return det
        text = await self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return text.strip() or det

    # --- 提示候補の選定 (Stage 2: L1 の中身) ---------------------------

    def select_for_target(
        self,
        target_node: Node,
        store: GraphStore,
        transcript: list[Utterance],
    ) -> list[InfoItem]:
        """``target_node`` の発話に対する提示候補を優先度付きで返す (L1 内部用)。

        外部互換: 既存テストや UI から呼べるよう公開 API として残す。
        新コードからは ``decide_intervention`` 経由が望ましい。
        """

        bias = self.detect_bias(store)
        stage = self.detect_stage(transcript, store)
        items = self._select_for_target(target_node, store, bias=bias, stage=stage)
        items.sort(key=lambda i: i.priority, reverse=True)
        result = items[: self._max_items]
        self.log.info(
            "facilitation.selected",
            target_id=str(target_node.id),
            n_candidates=len(items),
            n_selected=len(result),
            bias_imbalance=round(bias.imbalance_ratio, 3),
            stage=stage.stage,
        )
        return result

    def _select_for_target(
        self,
        target_node: Node,
        store: GraphStore,
        *,
        bias: BiasReport,
        stage: StageReport,
    ) -> list[InfoItem]:
        """priority 計算を含む内部ヘルパ (sort/cut は呼び出し側)。"""

        items: list[InfoItem] = []
        seen: set[str] = set()
        for edge in store.neighbors(target_node.id, direction="in"):
            src = store.get_node(edge.src_id)
            if src is None or src.id == target_node.id:
                continue
            key = f"{edge.relation}|{src.id}"
            if key in seen:
                continue
            seen.add(key)

            priority = edge.confidence
            reason: Literal["adjacent", "balance_correction", "stage_alignment"] = (
                "adjacent"
            )

            # 偏り補正
            if bias.imbalance_ratio > self._bias_threshold:
                if edge.relation != bias.dominant_side:
                    priority = min(priority * 1.3, 1.0)
                    reason = "balance_correction"
                else:
                    priority *= 0.7  # 優勢側を抑制

            # ステージ補正
            if stage.stage == "stalled" and edge.relation == "attack":
                priority = min(priority * 1.2, 1.0)
                if reason == "adjacent":
                    reason = "stage_alignment"
            elif stage.stage == "diverge" and edge.relation == "support":
                priority *= 0.85

            items.append(
                InfoItem(
                    relation=edge.relation,
                    target_text=target_node.text,
                    target_speaker=target_node.author,
                    source_text=src.text,
                    source_kind=src.source,
                    source_author=src.author,
                    confidence=edge.confidence,
                    rationale=edge.rationale,
                    priority=max(0.0, min(priority, 1.0)),
                    reason=reason,
                )
            )
        return items


__all__ = [
    "BiasReport",
    "DecisionKind",
    "FacilitationAgent",
    "InfoItem",
    "InterventionDecision",
    "Stage",
    "StageReport",
]

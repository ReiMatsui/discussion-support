"""ファシリテーションエージェント (中央調停者)。

研究計画書 §4.1 のファシリテーションエージェントに対応する実装。

責務:
  1. 統合議論グラフ全体を読みながら、いつ何を参加者に提示するかを判断する中央調停者
  2. 参加者の最新発言を **支持・攻撃するノード** を取り出して提示
  3. 議論が一方の立場に **構造的に偏っている** 状態を検知し、反対側を補強
  4. 議論ステージ (発散・収束・停滞) に応じて介入方針を切り替え

設計上の選択:
  - 偏り検知 / ステージ検知は LLM 不要のヒューリスティック (低コスト・決定的)
  - 提示候補は ``InfoItem`` (priority/reason 付きの構造化情報) として返す
    (UI/Condition 側は last_items を読むだけで動く)
  - LLM クライアントは将来の拡張用に保持 (例: ステージ判定の精緻化)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

from das.agents.base import BaseAgent
from das.graph.schema import Node, NodeSource
from das.graph.store import GraphStore
from das.llm import OpenAIClient
from das.types import Utterance

Stage = Literal["diverge", "converge", "stalled"]


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
    """議論ステージの判定結果。"""

    stage: Stage
    n_recent_turns: int
    repetition_rate: float
    """直近ターンのテキスト重複率 (0..1)。stalled 判定の主要シグナル。"""

    speaker_diversity: float
    """直近ターンのユニーク話者比 (0..1)。発散の指標。"""


@dataclass(frozen=True)
class InfoItem:
    """ファシリテーションが提示する 1 件の情報。

    UI・分析・LLM プロンプト生成の共通データ型。``priority`` と ``reason`` は
    FacilitationAgent が選定理由を残すための追加フィールドで、隣接エッジ取得
    だけしか行わない経路では既定値で問題なく動く。
    """

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
    """この項目が選ばれた論理的理由。介入の透明性 (§4.3) のために残す。"""



class FacilitationAgent(BaseAgent):
    """中央調停者。グラフ全体を読み、提示候補を優先度付きで返す。"""

    name = "facilitation"

    def __init__(
        self,
        llm: OpenAIClient | None = None,
        *,
        max_items: int = 3,
        recent_window: int = 6,
        bias_threshold: float = 0.4,
        stage_repetition_threshold: float = 0.5,
        stage_diversity_threshold: float = 0.7,
    ) -> None:
        super().__init__(llm=llm)
        self._max_items = max_items
        self._recent_window = recent_window
        self._bias_threshold = bias_threshold
        self._stalled_rep_threshold = stage_repetition_threshold
        self._diverge_diversity_threshold = stage_diversity_threshold

    # --- 偏り検知 ---------------------------------------------------

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

    # --- ステージ検知 ----------------------------------------------

    def detect_stage(self, transcript: list[Utterance]) -> StageReport:
        """発話の進み方からステージを推定するヒューリスティック。

        - stalled: 直近ターンのテキスト重複率が高い (同じことを繰り返している)
        - diverge: 話者がバラけ、新規性が高い
        - converge: その他 (典型的な収束フェーズ)
        """

        recent = transcript[-self._recent_window :] if transcript else []
        n = len(recent)
        if n == 0:
            return StageReport(
                stage="diverge",
                n_recent_turns=0,
                repetition_rate=0.0,
                speaker_diversity=0.0,
            )

        unique_speakers = len({u.speaker for u in recent})
        diversity = unique_speakers / n

        rep_pairs = 0
        possible = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                a, b = recent[i].text, recent[j].text
                if len(a) < 5 or len(b) < 5:
                    continue
                possible += 1
                set_a, set_b = set(a), set(b)
                overlap = len(set_a & set_b) / max(len(set_a), len(set_b))
                if overlap > 0.6:
                    rep_pairs += 1
        rep_rate = rep_pairs / possible if possible > 0 else 0.0

        stage: Stage
        if rep_rate > self._stalled_rep_threshold:
            stage = "stalled"
        elif diversity >= self._diverge_diversity_threshold and rep_rate < 0.3:
            stage = "diverge"
        else:
            stage = "converge"

        return StageReport(
            stage=stage,
            n_recent_turns=n,
            repetition_rate=rep_rate,
            speaker_diversity=diversity,
        )

    # --- 提示候補の選定 (中央調停の本体) -----------------------------

    def select_for_target(
        self,
        target_node: Node,
        store: GraphStore,
        transcript: list[Utterance],
    ) -> list[InfoItem]:
        """``target_node`` の発話に対する提示候補を優先度付きで返す。

        手順:
          1. 隣接エッジ (target を支持/攻撃するノード) を集める
          2. 偏り検知の結果に応じて優先度を補正
             - 全体が attack 優勢のとき、target への support を強調
             - 全体が support 優勢のとき、target への attack を強調
          3. ステージに応じてさらに補正
             - stalled: attack を優先 (議論を再活性化)
             - diverge: support を控えめに (発散の収束を促さない)
             - converge: 通常
          4. priority 降順 / max_items でカット
        """

        bias = self.detect_bias(store)
        stage = self.detect_stage(transcript)

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


__all__ = [
    "BiasReport",
    "FacilitationAgent",
    "InfoItem",
    "Stage",
    "StageReport",
]

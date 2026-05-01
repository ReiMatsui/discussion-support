"""シミュレーション結果に対する構造指標。

M2.4 で本格的に拡張するが、3 条件比較ページで使う基本指標をここに置く:
  - 発話ターン数 / 文字数 / 平均文字数
  - 発言バランス (話者ごとの発話数、Gini 係数)
  - 統合議論グラフを伴う条件向け: ノード/エッジ数、支持/攻撃の比率

すべて純関数として書き、引数として transcript と (任意の) GraphStore を受ける。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from das.graph.store import GraphStore
from das.types import Utterance


@dataclass(frozen=True)
class TranscriptMetrics:
    """transcript だけから計算できる指標。"""

    n_turns: int
    n_chars_total: int
    avg_chars_per_turn: float
    speaker_turn_counts: dict[str, int]
    gini_speaker_balance: float


@dataclass(frozen=True)
class GraphMetrics:
    """``GraphStore`` から計算する指標 (FullProposal 条件向け)。"""

    n_nodes: int
    n_edges: int
    n_utterance_nodes: int
    n_document_nodes: int
    n_web_nodes: int
    n_support_edges: int
    n_attack_edges: int
    support_attack_ratio: float | None  # support / (support + attack), edges 0 なら None


def gini_coefficient(values: list[int]) -> float:
    """整数列の Gini 係数 (0 = 完全均等, 1 = 完全不均等)。"""

    if not values:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    n = len(values)
    if n == 1:
        return 0.0
    sorted_vals = sorted(values)
    cum = 0.0
    weighted_sum = 0.0
    for i, v in enumerate(sorted_vals, start=1):
        cum += v
        weighted_sum += i * v
    # 標準的な Gini 公式: G = (2 Σ i * x_i) / (n Σ x_i) - (n+1)/n
    return (2.0 * weighted_sum) / (n * total) - (n + 1) / n


def transcript_metrics(transcript: list[Utterance]) -> TranscriptMetrics:
    """transcript の基本指標をまとめて返す。"""

    counts: Counter[str] = Counter(u.speaker for u in transcript)
    chars = sum(len(u.text) for u in transcript)
    n_turns = len(transcript)
    avg = chars / n_turns if n_turns else 0.0
    gini = gini_coefficient(list(counts.values()))
    return TranscriptMetrics(
        n_turns=n_turns,
        n_chars_total=chars,
        avg_chars_per_turn=avg,
        speaker_turn_counts=dict(counts),
        gini_speaker_balance=gini,
    )


def graph_metrics(store: GraphStore) -> GraphMetrics:
    """``GraphStore`` の基本指標。"""

    nodes = list(store.nodes())
    edges = list(store.edges())
    n_support = sum(1 for e in edges if e.relation == "support")
    n_attack = sum(1 for e in edges if e.relation == "attack")
    total_typed = n_support + n_attack
    ratio: float | None = None
    if total_typed > 0:
        ratio = n_support / total_typed
    return GraphMetrics(
        n_nodes=len(nodes),
        n_edges=len(edges),
        n_utterance_nodes=sum(1 for n in nodes if n.source == "utterance"),
        n_document_nodes=sum(1 for n in nodes if n.source == "document"),
        n_web_nodes=sum(1 for n in nodes if n.source == "web"),
        n_support_edges=n_support,
        n_attack_edges=n_attack,
        support_attack_ratio=ratio,
    )


__all__ = [
    "GraphMetrics",
    "TranscriptMetrics",
    "gini_coefficient",
    "graph_metrics",
    "transcript_metrics",
]

"""議論の客観構造指標 (AF + transcript から決定的に計算)。

研究計画書 §5.1 「客観指標」と DQI / Social Laboratory の流れに沿った
**LLM 不要の議論質指標**。AF と transcript さえあれば計算できるので、
シミュレーション・対面・音声どのモダリティでも同じ意味で使える。

採用した指標:

  - **participation_gini**: 話者間の発話数の偏り (0=完全平等)
  - **speaker_share**: 話者ごとの発話比率
  - **avg_premises_per_claim**: 発話 claim あたりの premise/support 数
  - **pct_unsupported_claims**: 根拠が無い claim の割合
  - **response_rate**: 直前発話に応答した発話の割合
  - **pct_attacks_answered**: 反論を受けた話者がさらに反論し返した割合
  - **avg_argument_chain_length**: 支持/攻撃チェーンの平均長
  - **n_isolated_claims**: どの claim とも繋がっていない孤立 claim 数

これらは Steenbergen et al. の DQI における **正当化レベル / 平等性 / 尊重 (応答性)**
の操作化として位置付けられる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from das.graph.schema import Edge, Node
from das.graph.store import GraphStore
from das.types import Utterance


@dataclass(frozen=True)
class DiscussionStructuralMetrics:
    """1 セッション分の構造指標。"""

    n_utterances: int
    n_speakers: int

    # 平等性 (DQI: equality of participation)
    participation_gini: float
    speaker_share: dict[str, float] = field(default_factory=dict)

    # 正当化レベル (DQI: level of justification)
    n_utterance_claims: int = 0
    avg_premises_per_claim: float = 0.0
    pct_unsupported_claims: float = 0.0

    # 応答性 (DQI: respect / mutual engagement)
    response_rate: float = 0.0
    pct_attacks_answered: float = 0.0

    # 構造の深さ
    avg_argument_chain_length: float = 0.0
    max_argument_chain_length: int = 0
    n_isolated_claims: int = 0

    # 議論ノードへの細分参考
    n_total_nodes: int = 0
    n_total_edges: int = 0
    n_support_edges: int = 0
    n_attack_edges: int = 0


# --- 補助関数 ---------------------------------------------------------


def _gini(values: list[float]) -> float:
    """Gini 係数。0 = 完全平等, 1 = 一極集中。"""

    if not values:
        return 0.0
    n = len(values)
    if n == 1:
        return 0.0
    mean = sum(values) / n
    if mean == 0:
        return 0.0
    abs_diff_sum = sum(abs(a - b) for a in values for b in values)
    return abs_diff_sum / (2 * n * n * mean)


def _utterance_nodes(store: GraphStore) -> list[Node]:
    return [n for n in store.nodes() if n.source == "utterance"]


def _utterance_claims(store: GraphStore) -> list[Node]:
    return [n for n in _utterance_nodes(store) if n.node_type == "claim"]


def _incoming_edges(store: GraphStore, node_id: UUID) -> list[Edge]:
    return [e for e in store.edges() if e.dst_id == node_id]


def _outgoing_edges(store: GraphStore, node_id: UUID) -> list[Edge]:
    return [e for e in store.edges() if e.src_id == node_id]


def _node_belongs_to_speaker(node: Node, speaker: str) -> bool:
    return node.author == speaker


def _argument_chain_max_depth(
    store: GraphStore, root_id: UUID, visited: set[UUID] | None = None
) -> int:
    """root_id を target として支持/攻撃の連鎖を遡る BFS 最大深さ。"""

    if visited is None:
        visited = set()
    if root_id in visited:
        return 0
    visited.add(root_id)
    incoming = _incoming_edges(store, root_id)
    if not incoming:
        return 0
    return 1 + max(
        _argument_chain_max_depth(store, e.src_id, visited) for e in incoming
    )


# --- 主関数 ----------------------------------------------------------


def compute_structural_metrics(
    transcript: list[Utterance], store: GraphStore | None
) -> DiscussionStructuralMetrics:
    """``transcript + store`` から客観構造指標を計算する。"""

    n_utts = len(transcript)
    speakers = sorted({u.speaker for u in transcript})
    n_speakers = len(speakers)

    # 平等性
    counts_by_speaker: dict[str, int] = {s: 0 for s in speakers}
    for u in transcript:
        counts_by_speaker[u.speaker] = counts_by_speaker.get(u.speaker, 0) + 1
    speaker_share = {
        s: (counts_by_speaker[s] / n_utts) if n_utts else 0.0 for s in speakers
    }
    gini = _gini([float(v) for v in counts_by_speaker.values()])

    metrics = DiscussionStructuralMetrics(
        n_utterances=n_utts,
        n_speakers=n_speakers,
        participation_gini=gini,
        speaker_share=speaker_share,
    )

    if store is None:
        return metrics

    # 正当化レベル
    edges = list(store.edges())
    nodes = list(store.nodes())
    utt_claims = _utterance_claims(store)
    n_claims = len(utt_claims)

    n_supports_per_claim: list[int] = []
    n_unsupported = 0
    for claim in utt_claims:
        n_supp = sum(
            1 for e in edges if e.dst_id == claim.id and e.relation == "support"
        )
        n_supports_per_claim.append(n_supp)
        if n_supp == 0:
            n_unsupported += 1
    avg_premises = (
        sum(n_supports_per_claim) / n_claims if n_claims else 0.0
    )
    pct_unsupported = n_unsupported / n_claims if n_claims else 0.0

    # 応答性: 各発話に対応するノードのうち少なくとも 1 つが
    # 「過去の発話/文書ノード」へ向けてエッジを持つ割合
    utt_nodes_by_turn: dict[int, list[Node]] = {}
    for n in _utterance_nodes(store):
        tid = n.metadata.get("turn_id")
        if tid is not None:
            utt_nodes_by_turn.setdefault(int(tid), []).append(n)

    n_responding = 0
    for u in transcript:
        nodes_for_u = utt_nodes_by_turn.get(u.turn_id, [])
        responded = False
        for n in nodes_for_u:
            for e in _outgoing_edges(store, n.id):
                target = next((x for x in nodes if x.id == e.dst_id), None)
                if target is None:
                    continue
                if target.timestamp < n.timestamp:
                    responded = True
                    break
            if responded:
                break
        if responded:
            n_responding += 1
    response_rate = n_responding / n_utts if n_utts else 0.0

    # pct_attacks_answered: attack edge で、その target を持つ speaker が
    # 後に source の speaker に対して attack を返した割合
    n_attacks = 0
    n_attacks_answered = 0
    attack_edges = [e for e in edges if e.relation == "attack"]
    for e in attack_edges:
        src = next((x for x in nodes if x.id == e.src_id), None)
        dst = next((x for x in nodes if x.id == e.dst_id), None)
        if src is None or dst is None:
            continue
        if src.author is None or dst.author is None or src.author == dst.author:
            continue
        n_attacks += 1
        # dst の speaker が、後に src の speaker のノードに attack しているか
        for e2 in attack_edges:
            if e2.src_id == e.src_id and e2.dst_id == e.dst_id:
                continue
            src2 = next((x for x in nodes if x.id == e2.src_id), None)
            dst2 = next((x for x in nodes if x.id == e2.dst_id), None)
            if src2 is None or dst2 is None:
                continue
            if (
                src2.author == dst.author
                and dst2.author == src.author
                and src2.timestamp > src.timestamp
            ):
                n_attacks_answered += 1
                break
    pct_answered = n_attacks_answered / n_attacks if n_attacks else 0.0

    # 構造の深さ
    chain_lens = [_argument_chain_max_depth(store, c.id, set()) for c in utt_claims]
    avg_chain = sum(chain_lens) / len(chain_lens) if chain_lens else 0.0
    max_chain = max(chain_lens) if chain_lens else 0
    n_isolated = sum(
        1 for c in utt_claims
        if not _incoming_edges(store, c.id) and not _outgoing_edges(store, c.id)
    )

    return DiscussionStructuralMetrics(
        n_utterances=n_utts,
        n_speakers=n_speakers,
        participation_gini=gini,
        speaker_share=speaker_share,
        n_utterance_claims=n_claims,
        avg_premises_per_claim=avg_premises,
        pct_unsupported_claims=pct_unsupported,
        response_rate=response_rate,
        pct_attacks_answered=pct_answered,
        avg_argument_chain_length=avg_chain,
        max_argument_chain_length=max_chain,
        n_isolated_claims=n_isolated,
        n_total_nodes=len(nodes),
        n_total_edges=len(edges),
        n_support_edges=sum(1 for e in edges if e.relation == "support"),
        n_attack_edges=sum(1 for e in edges if e.relation == "attack"),
    )


def aggregate_structural_metrics(
    runs: list[DiscussionStructuralMetrics],
) -> dict:
    """複数ラン分の構造指標を平均する (集計表示用)。"""

    if not runs:
        return {}

    def _mean(attr: str) -> float:
        return sum(getattr(r, attr) for r in runs) / len(runs)

    return {
        "n_runs": len(runs),
        "participation_gini_mean": _mean("participation_gini"),
        "avg_premises_per_claim_mean": _mean("avg_premises_per_claim"),
        "pct_unsupported_claims_mean": _mean("pct_unsupported_claims"),
        "response_rate_mean": _mean("response_rate"),
        "pct_attacks_answered_mean": _mean("pct_attacks_answered"),
        "avg_argument_chain_length_mean": _mean("avg_argument_chain_length"),
        "n_isolated_claims_mean": _mean("n_isolated_claims"),
        "n_total_edges_mean": _mean("n_total_edges"),
    }


__all__ = [
    "DiscussionStructuralMetrics",
    "aggregate_structural_metrics",
    "compute_structural_metrics",
]

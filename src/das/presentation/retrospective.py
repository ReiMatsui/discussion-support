"""L4 議論後の振り返り (個別化)。

研究計画書 §4.4 L4 に対応。議論終了後、参加者ごとに以下を提示する:
  - あなたが立てた主張の一覧
  - あなたの主張に張られた支持・反論
  - **応答できなかった反論** (= 受けたが反撃していない attack)
  - あなたが他者に向けた攻撃と、その応答の有無
  - あなたが立場と論証的に対立したノード (= attack を出した先)

こちらは決定的な集計のみで構成され、LLM は使わない。L3 と違って参加者ごとに
個別化された情報になる点が要点。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from das.graph.schema import Edge, Node
from das.graph.store import GraphStore
from das.types import Utterance


@dataclass(frozen=True)
class IncomingAttack:
    """ある主張に張られた 1 件の反論。"""

    target: Node
    attacker: Node
    edge: Edge


@dataclass(frozen=True)
class OutgoingAttack:
    """自分が他者に向けた 1 件の反論。"""

    source: Node
    target: Node
    edge: Edge


@dataclass(frozen=True)
class ParticipantRetrospective:
    """1 参加者向けの振り返りデータ。"""

    speaker: str
    own_claims: list[Node] = field(default_factory=list)
    unanswered_attacks: list[IncomingAttack] = field(default_factory=list)
    answered_attacks: list[IncomingAttack] = field(default_factory=list)
    outgoing_attacks: list[OutgoingAttack] = field(default_factory=list)
    text_summary: str = ""


def _is_self_response(
    attacked_claim: Node, store: GraphStore, speaker: str
) -> bool:
    """``attacked_claim`` の所有者がその後 attack を出していれば応答とみなす。"""

    for edge in store.neighbors(attacked_claim.id, direction="out"):
        if edge.relation != "attack":
            continue
        src = store.get_node(edge.src_id)
        if src is not None and src.author == speaker:
            return True
    return False


def retrospective_for(
    speaker: str,
    store: GraphStore,
    transcript: list[Utterance],
) -> ParticipantRetrospective:
    """指定 ``speaker`` 向けの振り返りを集計する。"""

    own_claims = [
        n for n in store.nodes() if n.source == "utterance" and n.author == speaker
    ]

    unanswered: list[IncomingAttack] = []
    answered: list[IncomingAttack] = []
    for claim in own_claims:
        for edge in store.neighbors(claim.id, direction="in"):
            if edge.relation != "attack":
                continue
            attacker = store.get_node(edge.src_id)
            if attacker is None:
                continue
            item = IncomingAttack(target=claim, attacker=attacker, edge=edge)
            if _is_self_response(claim, store, speaker):
                answered.append(item)
            else:
                unanswered.append(item)

    outgoing: list[OutgoingAttack] = []
    for claim in own_claims:
        for edge in store.neighbors(claim.id, direction="out"):
            if edge.relation != "attack":
                continue
            target = store.get_node(edge.dst_id)
            if target is None:
                continue
            outgoing.append(OutgoingAttack(source=claim, target=target, edge=edge))

    n_claims = len(own_claims)
    summary_parts = [f"{speaker} さん: {n_claims} 件の発話 claim を残しました。"]
    if unanswered:
        summary_parts.append(
            f"そのうち {len(unanswered)} 件は他者から反論を受けたまま、応答していません。"
        )
    if answered:
        summary_parts.append(f"逆に {len(answered)} 件の反論には応答しています。")
    if outgoing:
        summary_parts.append(
            f"あなた自身も他者に対し {len(outgoing)} 件の反論を出しています。"
        )
    if not (unanswered or answered or outgoing):
        summary_parts.append("反論の応酬は記録されていません。")

    return ParticipantRetrospective(
        speaker=speaker,
        own_claims=own_claims,
        unanswered_attacks=unanswered,
        answered_attacks=answered,
        outgoing_attacks=outgoing,
        text_summary="".join(summary_parts),
    )


def retrospectives_by_speaker(
    store: GraphStore,
    transcript: list[Utterance],
) -> dict[str, ParticipantRetrospective]:
    """transcript に登場した全話者の振り返りを返す。"""

    speakers = sorted({u.speaker for u in transcript})
    return {s: retrospective_for(s, store, transcript) for s in speakers}


__all__ = [
    "IncomingAttack",
    "OutgoingAttack",
    "ParticipantRetrospective",
    "retrospective_for",
    "retrospectives_by_speaker",
]

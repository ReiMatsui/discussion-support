"""議論の合意検出 (研究計画書 §5.1 客観指標「合意形成までの時間」対応)。

LLM 駆動のペルソナ議論において「いつ合意/収束したか」を多シグナルで検出する。
完全な意見一致は出にくいので、以下の条件を組み合わせて近似する:

  1. **明示的合意フレーズ**: 直近 N ターンの発話に "賛成", "同意", "なるほど",
     "了解", "おっしゃる通り" などが含まれる割合が高い
  2. **新規 claim の停止**: 直近 M ターンで新しい claim ノードが出ていない
     (graph store が利用可能な場合のみ判定)
  3. **反論エッジの停止**: 直近 M ターンで新しい attack エッジが追加されていない

何れかの強いシグナルが立ったら ``consensus_reached=True`` を返す。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from das.graph.store import GraphStore
from das.types import Utterance

ConsensusSignal = Literal[
    "explicit_agreement",
    "new_claim_stalled",
    "no_new_attacks",
    "none",
]

# 「合意・同意」を示唆する日本語キーワード
_AGREEMENT_KEYWORDS: tuple[str, ...] = (
    "賛成",
    "同意",
    "なるほど",
    "了解",
    "おっしゃる通り",
    "確かに",
    "その通り",
    "納得",
    "受け入れ",
    "合意",
    "歩み寄り",
)

# 合意キーワードの直後に来ると「実は反論の前置き」と判定する逆接表現。
# 例: 「確かにコストはあります**が、**長期的には〜」「なるほど、**しかし**〜」
_REBUTTAL_CONJUNCTIONS: tuple[str, ...] = (
    "が、",
    "しかし",
    "ただし",
    "ただ、",
    "でも、",
    "けれど",
    "けれども",
    "ですが、",
    "ですが",
    "一方",
    "もっとも",
    "とはいえ",
    "ものの",
)

# 合意キーワードと逆接の距離 (文字数) の上限。これ以内に逆接が現れたら
# 「表面的な譲歩 → 反論」のパターンとみなして合意扱いしない。
_NEGATION_PROXIMITY_CHARS = 30


def _agreement_keyword_index(text: str) -> int:
    """テキスト内で最も早く出てくる合意キーワードの末尾位置 (見つからなければ -1)。"""

    earliest_end = -1
    for kw in _AGREEMENT_KEYWORDS:
        idx = text.find(kw)
        if idx == -1:
            continue
        end = idx + len(kw)
        if earliest_end == -1 or end < earliest_end:
            earliest_end = end
    return earliest_end


def _has_genuine_agreement(text: str) -> bool:
    """合意キーワードがあり、かつ直後に逆接が無いとき True。

    LLM 発話では「確かに〜が、…」のような **譲歩 → 反論** パターンが超頻出する
    ので、純粋な合意とこの「前置き型」を見分けるために逆接距離を見る。
    """

    end = _agreement_keyword_index(text)
    if end == -1:
        return False
    tail = text[end : end + _NEGATION_PROXIMITY_CHARS]
    return not any(c in tail for c in _REBUTTAL_CONJUNCTIONS)


@dataclass(frozen=True)
class ConsensusReport:
    """合意検出の結果。"""

    consensus_reached: bool
    signal: ConsensusSignal
    confidence: float
    """0..1 のシグナル強度。"""

    rationale: str = ""
    detected_at_turn: int | None = None
    fired_signals: list[ConsensusSignal] = field(default_factory=list)


def _explicit_agreement_score(transcript: list[Utterance], window: int) -> float:
    """直近 ``window`` ターンのうち、**真正な合意**フレーズを含む割合。

    「確かに〜が、」のような逆接の前置きは除外する。
    """

    recent = transcript[-window:] if transcript else []
    if not recent:
        return 0.0
    n_with_genuine = sum(1 for u in recent if _has_genuine_agreement(u.text))
    return n_with_genuine / len(recent)


def _new_claims_in_recent_turns(
    store: GraphStore, recent_turn_ids: set[int]
) -> int:
    """直近ターンに対応する新しい claim ノード数。"""

    count = 0
    for node in store.nodes():
        if node.source != "utterance" or node.node_type != "claim":
            continue
        if node.metadata.get("turn_id") in recent_turn_ids:
            count += 1
    return count


def _new_attacks_in_recent_turns(
    store: GraphStore, recent_turn_ids: set[int]
) -> int:
    """直近ターンの発話ノードを送信元 / 受信元に持つ attack エッジ数。"""

    if not recent_turn_ids:
        return 0
    nodes_in_window = {
        n.id
        for n in store.nodes()
        if n.source == "utterance"
        and n.metadata.get("turn_id") in recent_turn_ids
    }
    if not nodes_in_window:
        return 0
    return sum(
        1
        for e in store.edges()
        if e.relation == "attack"
        and (e.src_id in nodes_in_window or e.dst_id in nodes_in_window)
    )


def detect_consensus(
    transcript: list[Utterance],
    *,
    store: GraphStore | None = None,
    agreement_window: int = 3,
    agreement_threshold: float = 0.67,
    stall_window: int = 4,
    min_turns_before_consensus: int = 6,
) -> ConsensusReport:
    """``transcript`` (と任意で ``store``) から合意状態を検出する。

    パラメータ:
      - ``agreement_window``: 直近何ターンを見て合意フレーズを判定するか
      - ``agreement_threshold``: そのうち何割以上に合意キーワードが含まれれば
        合意と見なすか
      - ``stall_window``: 「新規 claim / 攻撃が止まった」を判定するターン窓
      - ``min_turns_before_consensus``: 合意判定を始める最小ターン数 (序盤の
        誤検出を避ける)
    """

    if not transcript or len(transcript) < min_turns_before_consensus:
        return ConsensusReport(
            consensus_reached=False,
            signal="none",
            confidence=0.0,
            rationale="ターン数が不足",
            detected_at_turn=None,
        )

    fired: list[ConsensusSignal] = []
    rationales: list[str] = []
    confidence = 0.0

    # 1. 明示的合意フレーズ
    score = _explicit_agreement_score(transcript, agreement_window)
    if score >= agreement_threshold:
        fired.append("explicit_agreement")
        rationales.append(
            f"直近 {agreement_window} ターンの {score:.0%} に合意キーワード"
        )
        confidence = max(confidence, score)

    # 2. graph 利用可能時の構造シグナル
    if store is not None:
        recent = transcript[-stall_window:]
        recent_ids = {u.turn_id for u in recent}

        new_claims = _new_claims_in_recent_turns(store, recent_ids)
        new_attacks = _new_attacks_in_recent_turns(store, recent_ids)

        if new_claims == 0 and stall_window > 0:
            fired.append("new_claim_stalled")
            rationales.append(f"直近 {stall_window} ターンで新規 claim なし")
            confidence = max(confidence, 0.7)

        if new_attacks == 0 and stall_window > 0:
            fired.append("no_new_attacks")
            rationales.append(f"直近 {stall_window} ターンで新規 attack なし")
            confidence = max(confidence, 0.6)

    # 合意成立条件:
    #   - explicit_agreement (逆接除外後の真正な合意キーワード) 単独 OK
    #   - もしくは構造シグナル 2 つ以上 (no_new_attacks + new_claim_stalled)
    # 構造シグナル 1 つ単独では成立させない (extraction 遅延などで誤検出する)。
    has_explicit = "explicit_agreement" in fired
    structural_count = sum(
        1 for s in fired if s in {"new_claim_stalled", "no_new_attacks"}
    )
    consensus = has_explicit or structural_count >= 2

    primary: ConsensusSignal = "none"
    if has_explicit:
        primary = "explicit_agreement"
    elif fired:
        primary = fired[0]

    return ConsensusReport(
        consensus_reached=consensus,
        signal=primary,
        confidence=confidence if consensus else 0.0,
        rationale="; ".join(rationales) if rationales else "シグナルなし",
        detected_at_turn=transcript[-1].turn_id if consensus else None,
        fired_signals=fired,
    )


__all__ = ["ConsensusReport", "ConsensusSignal", "detect_consensus"]

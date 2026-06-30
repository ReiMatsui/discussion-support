"""介入採否の一元化（FacilitationController）— Phase 0/1.

設計: docs/design/intervention_controller_redesign.md

このモジュールは「採否（arbitration）だけ」を担う層を提供する。
候補の抽出・fact検査・文案生成は **ここでは一切行わない**（§3）。
既存の checker（drift/fact/participation）が生成した候補を受け取り、
「どれを・今・言うか／黙るか」と urgency / deadline を裁定するだけ。

Phase 1 では shadow mode で並走させる。Controller の判断は
``intervention_review.jsonl`` にログするのみで、実際の発話採否は
従来ロジック（``_select_*_decision``）のまま変えない。

Phase 1 の Controller は **決定的**（LLM呼び出しなし）に実装する。
これにより shadow 並走がレイテンシ・コスト・挙動へ影響しない。
LLM 裁定への置換は Phase 2 以降で行う。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ._constants import (
    _FACTCHECK_COOLDOWN,
    _FACTCHECK_PENDING_TTL,
    _INTERVENTION_COOLDOWN,
    _INTERVENTION_PAUSE_COUNT,
    _INTERVENTION_PAUSE_DRIFT,
    _INTERVENTION_PAUSE_FACT,
    _INTERVENTION_PAUSE_RETRY,
    _INVITE_SILENCE,
    _STALL_COOLDOWN,
    _STALL_SILENCE,
)

# 採否で扱う候補種別。drift/fact/invite/summarize に加え、現行の
# 通常トリガー（count/silence/stall）と barge-in 再送（retry）、
# conversation も Phase1 shadow の比較対象として受け付ける。
Kind = Literal[
    "fact", "drift", "retry", "count", "silence", "stall",
    "invite", "summarize", "conversation",
]

InterruptPolicy = Literal["allow_barge_in", "wait_for_pause", "never_barge_in"]
Urgency = Literal["barge_in", "wait_for_pause", "low"]


@dataclass(frozen=True)
class InterventionCandidate:
    """checker が生成した「言う価値があるか」の一次判定結果（§3.1）.

    Controller はこの候補を採否するだけで、中身（brief/payload）の生成や
    検査はしない。``id`` は LLM 往復中に候補リストが変わっても照合できる安定ID。
    """

    id: str
    kind: Kind
    brief: str
    target_speaker: str | None = None
    confidence: float = 0.0
    created_at: float = 0.0
    expires_at: float = 0.0          # 0.0 = 無期限
    source_turn_ids: list[str] = field(default_factory=list)
    retryable: bool = False
    interrupt_policy: InterruptPolicy = "wait_for_pause"
    payload: dict = field(default_factory=dict)


@dataclass(frozen=True)
class InterventionLogEntry:
    """直近の介入1件の構造化ログ（時刻・種別・要旨・理由, §3.3 / Phase0）.

    Controller の cooldown 判断（直前に同種をやったばかりか）に使うほか、
    レビュー用の ``recent_interventions`` としても渡す。
    """

    at: float
    kind: str
    brief: str = ""
    reason: str = ""

    def as_dict(self) -> dict:
        return {"at": round(self.at, 3), "kind": self.kind,
                "brief": self.brief, "reason": self.reason}


@dataclass(frozen=True)
class FacilitationInput:
    """Controller の入力スナップショット（§3.1）.

    ロックを保持したまま LLM を呼ばないため、入力は確定済みスナップショット
    として渡す（§8.5）。Phase1 は決定的なので並行性は単純だが、将来の
    epoch ベース stale 判定に備えて ``snapshot_epoch`` を持つ。
    """

    candidates: tuple[InterventionCandidate, ...]
    recent_interventions: tuple[InterventionLogEntry, ...]
    silence_elapsed: float
    snapshot_epoch: int
    now: float
    # 任意（将来フェーズ）。Phase1 では未使用。
    silence_summarize: float | None = None
    cooldown: float = _INTERVENTION_COOLDOWN


@dataclass(frozen=True)
class FacilitationDecision:
    """Controller の出力（§3.2）.

    candidate_id=None は「黙る」。配列indexではなく id で選ぶ。
    """

    decision_id: str
    candidate_id: str | None
    urgency: Urgency
    valid_for_epoch: int
    deadline_ms: int
    suppressed: tuple[dict, ...]   # [{"candidate_id":..., "reason":...}]
    reason: str

    def as_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "candidate_id": self.candidate_id,
            "urgency": self.urgency,
            "valid_for_epoch": self.valid_for_epoch,
            "deadline_ms": self.deadline_ms,
            "suppressed": list(self.suppressed),
            "reason": self.reason,
        }


# 種別ごとの採否ポリシー（§3.3）。完全な単一 min_interval にはしない。
#   priority    : 小さいほど優先（現行 fact>drift>retry / count>silence>stall>invite に整合）
#   pause       : 発話の切れ目として必要な沈黙秒（floor 判定, §4）
#   cooldown    : 直前の同種介入からの最小間隔（しつこさ防止）
#   deadline_ms : この時間内に発話開始できなければ stale 破棄（§3.5）
#   urgency     : 既定の緊急度
@dataclass(frozen=True)
class _KindPolicy:
    priority: int
    pause: float
    cooldown: float
    deadline_ms: int
    urgency: Urgency


_KIND_POLICY: dict[str, _KindPolicy] = {
    "fact":     _KindPolicy(0, _INTERVENTION_PAUSE_FACT, _FACTCHECK_COOLDOWN, 1500, "wait_for_pause"),
    "drift":    _KindPolicy(1, _INTERVENTION_PAUSE_DRIFT, _INTERVENTION_COOLDOWN, 2000, "wait_for_pause"),
    "retry":    _KindPolicy(2, _INTERVENTION_PAUSE_RETRY, 0.0, 2000, "wait_for_pause"),
    "count":    _KindPolicy(3, _INTERVENTION_PAUSE_COUNT, 0.0, 2000, "wait_for_pause"),
    "silence":  _KindPolicy(4, 0.0, 0.0, 2000, "low"),
    "summarize": _KindPolicy(5, _STALL_SILENCE, _STALL_COOLDOWN, 2000, "low"),
    "stall":    _KindPolicy(5, _STALL_SILENCE, _STALL_COOLDOWN, 2000, "low"),
    "invite":   _KindPolicy(6, _INVITE_SILENCE, _INTERVENTION_COOLDOWN, 2000, "wait_for_pause"),
    "conversation": _KindPolicy(7, 0.0, 0.0, 2000, "low"),
}
_DEFAULT_POLICY = _KindPolicy(9, 1.0, _INTERVENTION_COOLDOWN, 2000, "low")


def _policy_for(kind: str) -> _KindPolicy:
    return _KIND_POLICY.get(kind, _DEFAULT_POLICY)


class FacilitationController:
    """採否を一元化する単一の裁定器（§3）.

    Phase 1 では shadow mode 専用の **決定的** 実装。候補を一括で見て
    「今どれを採るか／黙るか」「urgency」を返すだけ。抽出・fact検査・
    文案生成はしない。timeout 概念は無い（同期・決定的）が、戻り値の
    ``deadline_ms`` で「正しいが遅い介入」を後段が破棄できるようにする。
    """

    def __init__(self) -> None:
        self._seq = 0

    def arbitrate(self, inp: FacilitationInput) -> FacilitationDecision:
        """候補群を見て1つ採る／全部見送る。採否と緊急度の裁定のみ."""
        self._seq += 1
        decision_id = f"dec-{self._seq:05d}"
        suppressed: list[dict] = []

        eligible: list[tuple[int, InterventionCandidate]] = []
        for cand in inp.candidates:
            ok, reason = self._eligible(cand, inp)
            if ok:
                eligible.append((_policy_for(cand.kind).priority, cand))
            else:
                suppressed.append({"candidate_id": cand.id, "reason": reason})

        if not eligible:
            why = "候補なし" if not inp.candidates else "全候補を抑制（クールダウン/間待ち/期限切れ）"
            return FacilitationDecision(
                decision_id=decision_id, candidate_id=None, urgency="low",
                valid_for_epoch=inp.snapshot_epoch, deadline_ms=0,
                suppressed=tuple(suppressed), reason=why)

        # 優先度 → confidence の順で1件を採る（残りは抑制扱いで透明化）
        eligible.sort(key=lambda pc: (pc[0], -pc[1].confidence))
        _, chosen = eligible[0]
        for _, other in eligible[1:]:
            suppressed.append({"candidate_id": other.id,
                               "reason": f"優先度が低い（採択={chosen.kind}）"})

        policy = _policy_for(chosen.kind)
        urgency = self._urgency(chosen, policy, inp)
        return FacilitationDecision(
            decision_id=decision_id, candidate_id=chosen.id, urgency=urgency,
            valid_for_epoch=inp.snapshot_epoch, deadline_ms=policy.deadline_ms,
            suppressed=tuple(suppressed),
            reason=f"{chosen.kind} を採択（{chosen.brief[:30]}）")

    # ------------------------------------------------------------------
    def _eligible(self, cand: InterventionCandidate,
                  inp: FacilitationInput) -> tuple[bool, str]:
        """候補が「今」採れるか。採れない理由（抑制理由）も返す."""
        policy = _policy_for(cand.kind)
        # 期限切れ（§3.1 expires_at）
        if cand.expires_at and inp.now > cand.expires_at:
            return False, "期限切れ（鮮度を失った）"
        # 同種クールダウン（§3.3）
        if policy.cooldown > 0:
            last = self._last_same_kind(cand.kind, inp.recent_interventions)
            if last is not None and inp.now - last < policy.cooldown:
                return False, f"直前に同種介入済み（cooldown {policy.cooldown:.0f}s）"
        # floor / 間待ち（§4）。barge-in許可種別は pause を無視できる。
        if (cand.interrupt_policy != "allow_barge_in"
                and inp.silence_elapsed < policy.pause):
            return False, f"発話の切れ目待ち（必要 {policy.pause:.1f}s）"
        return True, ""

    @staticmethod
    def _last_same_kind(kind: str,
                        recent: tuple[InterventionLogEntry, ...]) -> float | None:
        times = [e.at for e in recent if e.kind == kind]
        return max(times) if times else None

    @staticmethod
    def _urgency(cand: InterventionCandidate, policy: _KindPolicy,
                 inp: FacilitationInput) -> Urgency:
        if cand.interrupt_policy == "allow_barge_in":
            return "barge_in"
        if cand.interrupt_policy == "never_barge_in":
            return "low"
        return policy.urgency


# 鮮度に依存する fact 候補の有効期限を作るためのヘルパー。
def fact_expires_at(queued_at: float) -> float:
    return queued_at + _FACTCHECK_PENDING_TTL


def confidence_score(label: object) -> float:
    """checker の confidence ラベル（high/medium/low）を数値化する."""
    return {"high": 0.9, "medium": 0.5, "low": 0.2}.get(str(label).lower(), 0.0)

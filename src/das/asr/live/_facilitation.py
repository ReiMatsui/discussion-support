"""介入採否の一元化（FacilitationController）.

設計: docs/design/intervention_controller_redesign.md

このモジュールは「採否（arbitration）だけ」を担う層を提供する。
候補の抽出・fact検査・文案生成は **ここでは一切行わない**（§3）。
既存の checker（drift/fact/participation）が生成した候補を受け取り、
「どれを・今・言うか／黙るか」と urgency / deadline を裁定するだけ。

Phase2 以降、この Controller が実際の発話採否を駆動する（固定優先順位の
``_select_*_decision`` を置換）。採否の経緯は ``intervention_review.jsonl`` に
記録され、なぜ話したか／なぜ黙ったかを追える。

Controller は **決定的**（LLM呼び出しなし）に実装する。同期・低遅延で採否を返し、
戻り値の ``deadline_ms`` で「正しいが遅い介入」を後段が破棄できるようにする。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ._constants import (
    _AGENT_CONV_SILENCE,
    _FACTCHECK_COOLDOWN,
    _FACTCHECK_PENDING_TTL,
    _INTERVENTION_COOLDOWN,
    _INTERVENTION_PAUSE_COUNT,
    _INTERVENTION_PAUSE_DRIFT,
    _INTERVENTION_PAUSE_FACT,
    _INTERVENTION_PAUSE_MANUAL,
    _INTERVENTION_PAUSE_RETRY,
    _INVITE_SILENCE,
    _MANUAL_CALL_COOLDOWN,
)

# 採否で扱う候補種別。現行 checker が生成する fact/drift/retry/count/silence/
# invite/conversation を受け付ける。summarize は設計（§3）上の将来kindで、
# まだ候補生成器を持たない（policy 未定義なら _DEFAULT_POLICY にフォールバック）。
# 注: stall（介入不要後のデッドエア一押し）は Phase3 で廃止した。
Kind = Literal[
    "fact", "manual", "drift", "retry", "count", "silence",
    "invite", "summarize", "conversation",
]

InterruptPolicy = Literal["allow_barge_in", "wait_for_pause", "never_barge_in"]
Urgency = Literal["barge_in", "wait_for_pause", "low"]

# 抑制理由の機械可読コード。worker 側の後処理（drift破棄・invite消費など）は
# このコードだけで分岐する。表示文（reason）は UI/ログ専用で、文言変更が
# 挙動に影響してはならない（H4: 文字列プロトコルの禁止）。
SuppressionCode = Literal[
    "expired",                      # expires_at を過ぎた（鮮度喪失）
    "partner_busy",                 # パートナー発話中
    "echo_window",                  # エコーウィンドウ中
    "awaiting_drift_confirmation",  # 脱線の確認回数待ち
    "cooldown_global",              # 直前のあらゆる介入から間隔不足
    "cooldown_kind",                # 直前の同種介入から間隔不足
    "awaiting_pause",               # 発話の切れ目（pause）待ち
    "same_as_last_invited",         # 直前と同じ相手への連続声かけ
    "lower_priority",               # 採択可能だが他候補を優先
]


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
    として渡す（§8.5）。Controller は決定的なので並行性は単純だが、将来の
    epoch ベース stale 判定に備えて ``snapshot_epoch`` を持つ。
    """

    candidates: tuple[InterventionCandidate, ...]
    recent_interventions: tuple[InterventionLogEntry, ...]
    silence_elapsed: float
    snapshot_epoch: int
    now: float
    cooldown: float = _INTERVENTION_COOLDOWN
    # --- 物理コンテキスト（floor / barge-in 層, §4） ---
    # barge-in 層では partner 発話中・エコー残響中は誰も差し込まない。
    partner_busy: bool = False
    in_echo_window: bool = False
    # drift / invite は「直前の介入から一定間隔」を見る（種別横断の global cooldown）。
    last_intervention_at: float = 0.0
    # 脱線は確認回数（drift_confirmations）に達してから採る。0 ならゲートしない。
    required_drift_confirmations: int = 0
    # fact の同種クールダウンは fast lane の鮮度設定をそのまま使う（live 定数を注入）。
    fact_cooldown: float | None = None


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
    # [{"candidate_id":..., "code": SuppressionCode, "reason": 表示文}]
    suppressed: tuple[dict, ...]
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
#   priority    : 小さいほど優先（fact>drift>retry / count>silence>invite に整合）
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
    # "kind"  : 直前の「同種」介入からの間隔を見る（fact/count/silence）
    # "global": 直前の「あらゆる」介入からの間隔を見る（drift/invite。会話を頻繁に止めない）
    cooldown_scope: str = "kind"


# 注: summarize（将来kind）は候補生成器が未実装のため policy を持たない。
# 実装時にここへ追加する。それまでは _DEFAULT_POLICY にフォールバックする。
_KIND_POLICY: dict[str, _KindPolicy] = {
    "fact":     _KindPolicy(0, _INTERVENTION_PAUSE_FACT, _FACTCHECK_COOLDOWN, 1500, "wait_for_pause"),
    # manual: ユーザーが明示的に呼んだので基本尊重。ただし直前に明確な fact 補正が
    # あればそちらを優先。global cooldown は受けず（kind scope）、連打だけ抑える。
    "manual":   _KindPolicy(1, _INTERVENTION_PAUSE_MANUAL, _MANUAL_CALL_COOLDOWN, 3000, "wait_for_pause"),
    "drift":    _KindPolicy(2, _INTERVENTION_PAUSE_DRIFT, _INTERVENTION_COOLDOWN, 2000, "wait_for_pause", "global"),
    "retry":    _KindPolicy(3, _INTERVENTION_PAUSE_RETRY, 0.0, 2000, "wait_for_pause"),
    "count":    _KindPolicy(4, _INTERVENTION_PAUSE_COUNT, 0.0, 2000, "wait_for_pause"),
    "silence":  _KindPolicy(5, 0.0, 0.0, 2000, "low"),
    "invite":   _KindPolicy(6, _INVITE_SILENCE, _INTERVENTION_COOLDOWN, 2000, "wait_for_pause", "global"),
    "conversation": _KindPolicy(7, _AGENT_CONV_SILENCE, 0.0, 2000, "low"),
}
_DEFAULT_POLICY = _KindPolicy(9, 1.0, _INTERVENTION_COOLDOWN, 2000, "low")


def policy_for(kind: str) -> _KindPolicy:
    """種別ごとの採否ポリシー（pause/cooldown/優先度）の唯一の出所（M8）.

    dispatch 側のタイミングログもこの表を参照し、pause 値の重複管理をしない。
    """
    return _KIND_POLICY.get(kind, _DEFAULT_POLICY)


class FacilitationController:
    """採否を一元化する単一の裁定器（§3）.

    **決定的** 実装。候補を一括で見て「今どれを採るか／黙るか」「urgency」を
    返すだけ。抽出・fact検査・文案生成はしない。timeout 概念は無い（同期・
    決定的）が、戻り値の ``deadline_ms`` で「正しいが遅い介入」を後段が破棄
    できるようにする。Phase2 以降、この採否が実際の発話を駆動する。
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
            ok, code, reason = self._eligible(cand, inp)
            if ok:
                eligible.append((policy_for(cand.kind).priority, cand))
            else:
                suppressed.append(
                    {"candidate_id": cand.id, "code": code, "reason": reason})

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
                               "code": "lower_priority",
                               "reason": f"優先度が低い（採択={chosen.kind}）"})

        policy = policy_for(chosen.kind)
        urgency = self._urgency(chosen, policy, inp)
        return FacilitationDecision(
            decision_id=decision_id, candidate_id=chosen.id, urgency=urgency,
            valid_for_epoch=inp.snapshot_epoch, deadline_ms=policy.deadline_ms,
            suppressed=tuple(suppressed),
            reason=f"{chosen.kind} を採択（{chosen.brief[:30]}）")

    # ------------------------------------------------------------------
    def _eligible(self, cand: InterventionCandidate,
                  inp: FacilitationInput) -> tuple[bool, str, str]:
        """候補が「今」採れるか。``(ok, code, 表示文)`` を返す.

        code は機械可読な :data:`SuppressionCode`。worker の後処理はこのコード
        だけで分岐し、表示文はログ/UI専用とする（文言の推敲が挙動を変えない）。

        判定順序は legacy のフロア条件に揃える（§4）:
          期限切れ → 物理フロア(発話中/エコー) → 脱線確認 → クールダウン →
          間待ち → 連続声かけ。
        連続声かけ(same_as_last_invited)は最後に見る。これにより「間やクールダウンが
        満たされて初めて『同じ人だから今回は見送る(skip_invite)』」を再現できる。
        """
        policy = policy_for(cand.kind)
        # 期限切れ（§3.1 expires_at）= 古い判断の破棄（§8.5）
        if cand.expires_at and inp.now > cand.expires_at:
            return False, "expired", "期限切れ（鮮度を失った）"
        # 物理フロア（§4）: barge-in 許可種別以外は、発話中・エコー残響中は待つ。
        if cand.interrupt_policy != "allow_barge_in":
            if inp.partner_busy:
                return False, "partner_busy", "パートナー発話中で待機"
            if inp.in_echo_window:
                return False, "echo_window", "エコーウィンドウ中で待機"
        # 脱線の確認待ち（連続検出で初めて採る）
        if (cand.kind == "drift" and inp.required_drift_confirmations > 0
                and int(cand.payload.get("drift_count", 0))
                < inp.required_drift_confirmations):
            return False, "awaiting_drift_confirmation", (
                f"脱線判定の確認待ち "
                f"({int(cand.payload.get('drift_count', 0))}/"
                f"{inp.required_drift_confirmations})")
        # クールダウン（§3.3）。kind別 / global を共通engineで扱う。
        if policy.cooldown_scope == "global":
            if (inp.last_intervention_at
                    and inp.now - inp.last_intervention_at < inp.cooldown):
                return False, "cooldown_global", (
                    f"直前の介入から間隔不足（cooldown {inp.cooldown:.0f}s）")
        else:
            cd = policy.cooldown
            if cand.kind == "fact" and inp.fact_cooldown is not None:
                cd = inp.fact_cooldown
            if cd > 0:
                last = self._last_same_kind(cand.kind, inp.recent_interventions)
                if last is not None and inp.now - last < cd:
                    return False, "cooldown_kind", (
                        f"直前に同種介入済み（cooldown {cd:.0f}s）")
        # floor / 間待ち（§4）。barge-in許可種別は pause を無視できる。
        pause_required = float(cand.payload.get("pause_required", policy.pause))
        if (cand.interrupt_policy != "allow_barge_in"
                and inp.silence_elapsed < pause_required):
            return False, "awaiting_pause", (
                f"発話の切れ目待ち（必要 {pause_required:.1f}s）")
        # 連続声かけ（間・クールダウンを満たした上で、同じ人なら今回は見送る）
        if cand.payload.get("same_as_last_invited"):
            return False, "same_as_last_invited", "直前と同じ参加者への連続声かけ"
        return True, "", ""

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

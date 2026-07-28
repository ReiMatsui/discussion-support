"""main()から抽出されたワーカー関数群.

ログ接頭辞の規約（Phase 3 R4）:
  # [state]   ... エージェントの状態遷移（RESPONDING/SPEAKING/INTERRUPTED/IDLE等）
  # [trigger] ... ファシリテーターのトリガー理由（drift/retry/summarize/silence/invite/skip）
  # [drift]   ... 並列ドリフト（脱線）検出の動作
  # [diag]    ... 定期的な状態ダンプ・スキップ理由などの診断
"""
from __future__ import annotations

import collections
import contextlib
import datetime
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ._session_state import SessionState
    from .stt import STTBackend

from ._constants import (
    _ACK_CHIME_ENABLED,
    _AGENDA_MIN_UTTS,
    _AGENDA_RETRY_SEC,
    _AGENDA_WINDOW,
    _AGENT_DEBATE_SILENCE,
    _BACKCHANNEL_RE,
    _DRIFT_CHECK_INTERVAL,
    _DRIFT_CHECK_WINDOW,
    _DRIFT_PENDING_TTL,
    _DRIFT_WARMUP,
    _FACTCHECK_CHECK_SEC,
    _FACTCHECK_COOLDOWN,
    _FACTCHECK_MAX_RETRIES,
    _FACTCHECK_PENDING_TTL,
    _INTERRUPT_MIN_CHARS,
    _INTERVENTION_COOLDOWN,
    _INVITE_CHECK_SEC,
    _INVITE_QUIET_RATIO,
    _INVITE_WARMUP,
    _MANUAL_CALL_MAX_CHARS,
    _MANUAL_CALL_TTL,
    _PARTIAL_FLOOR_MAX_AGE,
    _STRUCTURING_WINDOW,
    _TRIAGE_BACKLOG_MAX,
    _TRIAGE_CONTEXT_WINDOW,
    _TRIAGE_MAX_RETRIES,
    _TRIAGE_MIN_CHARS,
    AGENT_SPEAKER,
    SR,
)
from ._facilitation import (
    FacilitationController,
    FacilitationDecision,
    FacilitationInput,
    InterventionCandidate,
    InterventionLogEntry,
    confidence_score,
    fact_expires_at,
    policy_for,
)
from ._participation import (
    participation_share_key,
    participation_share_label,
    participation_stats,
    quietest_participation_share,
)
from ._speaker_policy import (
    intervention_records,
    intervention_speaker_name,
    reliable_human_records,
    triage_records,
)
from ._ui import _print_line


def _effective_silence(state: SessionState, now: float,
                       last_utt_time: list[float]) -> float:
    """フロア判定用の沈黙経過秒（F3）.

    誰かの発話が今まさに転写されている（``partial_text`` が非空でかつ直近に更新
    されている）間は「フロアは埋まっている」とみなして沈黙 0 を返す。人間が発話の
    途中で置く1秒程度の自然な間を「フロアが空いた」と誤認して介入が発話に被さるのを
    防ぐ。partial が ``_PARTIAL_FLOOR_MAX_AGE`` 秒以上変化していなければ stale として
    無視する（partial がクリアされずに固着した場合の保険）。
    """
    with state.state_lock:
        partial = state.partial_text
        changed = state._last_partial_change
    if partial and (now - changed) < _PARTIAL_FLOOR_MAX_AGE:
        return 0.0
    return now - last_utt_time[0]


def _log_voice_call_diag(state: SessionState, *, text: str,
                         request: str | None, reason: str | None = None) -> None:
    """音声呼びかけの検出を ``.interventions.jsonl`` に診断として残す.

    triage 分類が依頼を検出した発話だけが対象なので低頻度（ログ洪水にならない）。
    replay 等の既存ローダーは type=trigger/delivery しか読まないため無害。
    state が未対応（テストの FakeState 等）なら no-op。
    """
    write = getattr(state, "write_intervention_event", None)
    if not callable(write):
        return
    with contextlib.suppress(Exception):
        write({
            "type": "voice_call_diag",
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "detected": request is not None,
            "text": text[:80],
            "request": request,
            "ignored_reason": reason,
        })


def _intervention_event_metadata(
    state: SessionState,
    *,
    recent_limit: int = 5,
    timing: dict | None = None,
) -> dict:
    """介入の事後レビューに必要な最小コンテキストを作る."""
    with state.state_lock:
        utterances = [
            {
                "speaker": intervention_speaker_name(state, r),
                "text": str(r.get("text", "")),
                "ms": r.get("ms"),
                "end_ms": r.get("end_ms"),
            }
            for r in state.records
            if "speaker" in r and r.get("text")
        ]
    with state.topics_lock:
        topics = [
            {"topic": str(t.get("topic", "")), "speaker": str(t.get("speaker", ""))}
            for t in state.topics[:5]
        ]
    agent = getattr(state, "agent", None)
    metadata = {
        "mode": getattr(agent, "mode", None),
        "proactivity": getattr(state, "proactivity_name", None),
        "turn_count": len(utterances),
        "recent_utterances": utterances[-recent_limit:],
        "topics": topics,
    }
    if timing:
        metadata["timing"] = timing
    return metadata


def _log_intervention_event(
    state: SessionState,
    reason: str,
    detail: str = "",
    *,
    timing: dict | None = None,
) -> None:
    add_event = getattr(state, "add_intervention_event", None)
    if callable(add_event):
        add_event(reason, detail, metadata=_intervention_event_metadata(state, timing=timing))


def _intervention_enabled(state: SessionState) -> bool:
    return bool(getattr(state, "intervention_enabled", True))


def _set_manual_status(state: SessionState, status: str, **kw) -> None:
    """手動呼び出しのUIステータスを更新する（state 未対応なら no-op）."""
    setter = getattr(state, "set_manual_call_status", None)
    if callable(setter):
        with contextlib.suppress(Exception):
            setter(status, **kw)


def _is_backchannel(text: str) -> bool:
    """相槌 (「そうですね」「なるほど」等) か判定する (空文字も割り込み対象外扱い)。

    ファシリテーター/パートナー両方の自動割り込み判定を共通化するヘルパー (T7)。
    相槌でAIの発話をキャンセルしないための除外に使う。
    """
    t = text.strip()
    return not t or bool(_BACKCHANNEL_RE.match(t))


def _as_bool(value: Any) -> bool:
    """LLM 出力の真偽値を頑健に正規化する (T9-1)。

    JSON パースで bool になるのが正だが、LLM が文字列 ``"false"`` / ``"no"`` 等を
    返すと ``bool("false")`` が True になる。文字列は明示的に真値語だけ True にする。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y")
    return bool(value)


# AF ベース介入候補の TTL（H1 フェーズ4）。af_l1 はアクティブ窓と整合、af_l2 は長め。
_AF_L1_PENDING_TTL = 45.0
_AF_L2_PENDING_TTL = 90.0


@dataclass
class _PendingInterventions:
    """agent_worker が一元調停する未処理介入要求."""

    drift_reason: str | None = None
    drift_count: int = 0
    last_drift_request_at: float = 0.0
    last_drift_request_wall_at: str | None = None
    facts: collections.deque[dict] = field(
        default_factory=lambda: collections.deque(maxlen=5))
    invite: str | None = None
    manual_call: dict | None = None
    summarize: dict[str, Any] | None = None
    # AF ベース介入（af_l1/af_l2）。AF ランタイム有効時のみ _run_af_checker が積む。
    af: dict[str, Any] | None = None

    def drain(self, state: SessionState, *, now: float) -> None:
        """各監視ワーカーのキューを回収し、保留状態に反映する."""
        while True:
            try:
                self.drift_reason = state.drift_requests.get_nowait()
                if now - self.last_drift_request_at > 20.0:
                    self.drift_count = 0
                self.last_drift_request_at = now
                self.last_drift_request_wall_at = datetime.datetime.now().isoformat(
                    timespec="seconds")
                self.drift_count += 1
            except queue.Empty:
                break
        while True:
            try:
                self.invite = state.invite_requests.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                fact = state.factcheck_requests.get_nowait()
                fact.setdefault("_queued_at", now)
                fact.setdefault("_queued_wall_at", datetime.datetime.now().isoformat(
                    timespec="seconds"))
                self.facts.append(fact)
            except queue.Empty:
                break
        manual_q = getattr(state, "manual_call_requests", None)
        if manual_q is not None:
            while True:
                try:
                    # 複数回押されても最新の依頼だけを保持する。
                    self.manual_call = manual_q.get_nowait()
                    self.manual_call.setdefault("created_at", now)
                except queue.Empty:
                    break
        summarize_q = getattr(state, "summarize_requests", None)
        if summarize_q is not None:
            while True:
                try:
                    # 最新の整理要求だけを保持する（採択されるまで drain で維持）。
                    self.summarize = summarize_q.get_nowait()
                    self.summarize.setdefault("created_at", now)
                except queue.Empty:
                    break
        af_q = getattr(state, "af_requests", None)
        if af_q is not None:
            while True:
                try:
                    # 最新の AF 介入候補だけを保持する（AF 有効時のみ積まれる）。
                    self.af = af_q.get_nowait()
                    self.af.setdefault("created_at", now)
                except queue.Empty:
                    break

    def drop_stale_manual(self, *, now: float) -> dict | None:
        """会話タイミングを外した古い手動呼び出しを破棄する（TTL）.

        破棄した場合はその payload を返す（expired の観測ログ用）。
        """
        if self.manual_call is None:
            return None
        age = now - float(self.manual_call.get("created_at", now))
        if age > _MANUAL_CALL_TTL:
            print(f"# [trigger] skip: 古い手動呼び出しを破棄（{age:.0f}秒経過）",
                  flush=True)
            dropped = self.manual_call
            self.manual_call = None
            return dropped
        return None

    def clear_manual(self) -> None:
        self.manual_call = None

    def clear_af(self) -> None:
        self.af = None

    def drop_stale_af(self, *, now: float) -> None:
        """タイミングを外した古い AF 介入候補を破棄する（TTL, 種別別）."""
        if self.af is None:
            return
        ttl = _AF_L2_PENDING_TTL if self.af.get("kind") == "af_l2" else _AF_L1_PENDING_TTL
        age = now - float(self.af.get("created_at", now))
        if age > ttl:
            print(f"# [trigger] skip: 古い AF 介入候補を破棄（{age:.0f}秒経過）", flush=True)
            self.af = None

    def clear_all(self) -> None:
        """介入オフ/モードオフ時に、worker内で保持した候補も破棄する."""
        self.drift_reason = None
        self.drift_count = 0
        self.last_drift_request_at = 0.0
        self.last_drift_request_wall_at = None
        self.facts.clear()
        self.invite = None
        self.manual_call = None
        self.summarize = None
        self.af = None

    def drop_stale_facts(self, *, now: float) -> None:
        """会話タイミングを外した古い事実補正を破棄する."""
        while self.facts:
            age = now - float(self.facts[0].get("_queued_at", now))
            if age <= _FACTCHECK_PENDING_TTL:
                return
            stale = self.facts.popleft()
            print(f"# [trigger] skip: 古い事実補正を破棄 {stale.get('correction', '')}",
                  flush=True)

    def drop_stale_drift(self, *, now: float) -> None:
        """確認待ちのまま古くなった脱線候補を破棄する（TTL）.

        drift は確認回数（drift_confirmations）に達するまで採択されない。
        1回だけ検出されて会話が自然に本題へ戻った場合、候補が無期限に残ると
        全介入レーンの飢餓を招くため、寿命を過ぎたら忘れる。
        """
        if self.drift_reason is None:
            return
        age = now - self.last_drift_request_at
        if age > _DRIFT_PENDING_TTL:
            print(f"# [trigger] skip: 古い脱線候補を破棄（{age:.0f}秒経過）",
                  flush=True)
            self.clear_drift()

    def clear_drift(self) -> None:
        self.drift_reason = None
        self.drift_count = 0


@dataclass(frozen=True)
class _BargeInDecision:
    reason: str
    fact: dict | None = None
    drift_reason: str | None = None
    manual: dict | None = None


@dataclass(frozen=True)
class _NormalTriggerDecision:
    reason: str
    detail: str = ""
    invite_target: str | None = None
    drift_reason: str | None = None
    summary_focus: str | None = None
    af_text: str | None = None  # AF ベース介入の提示文（H1 フェーズ4）


def _intervention_timing_metadata(
    *,
    kind: str,
    now: float,
    silence_elapsed: float,
    pause_required: float,
    queued_at: float | None = None,
    queued_wall_at: str | None = None,
    policy: str = "floor_pause",
    hold_to_release_ms: float | None = None,
) -> dict:
    """介入の自然さ/遅延レビュー用タイミング情報を作る."""
    timing = {
        "kind": kind,
        "policy": policy,
        "pause_required_sec": round(pause_required, 3),
        "pause_actual_sec": round(silence_elapsed, 3),
    }
    if queued_at is not None:
        timing["candidate_wait_sec"] = round(max(0.0, now - queued_at), 3)
    if queued_wall_at:
        timing["candidate_created_at"] = queued_wall_at
    if hold_to_release_ms is not None:
        timing["hold_to_release_ms"] = round(hold_to_release_ms, 1)
    return timing


def _recent_agent_texts(state: SessionState, n: int = 4) -> list[str]:
    """直近にファシリテーターが実際に発話したテキスト（新しい順でない末尾n件）.

    trigger のコンテキストに渡し、生成側で「同じ内容の介入の繰り返し」を
    防ぐために使う（duplicate_content の第2層）。records は膨らむため
    後方からの走査で打ち切る。
    """
    texts: list[str] = []
    with state.state_lock:
        for r in reversed(state.records):
            if r.get("speaker") == AGENT_SPEAKER and r.get("text"):
                texts.append(str(r["text"]))
                if len(texts) >= n:
                    break
    return list(reversed(texts))


def _build_candidates(
    pending: _PendingInterventions,
    agent,
    *,
    now: float,
    silence_summarize: float | None = None,
    partner_present: bool = False,
    last_invited: str | None = None,
) -> list[InterventionCandidate]:
    """保留中の介入要求を Controller 入力用の候補へ変換する（読み取り専用）.

    pending のキュー/deque は **pop しない**。あくまで現在の候補スナップショットを
    作るだけで、状態を変えない。fact 候補は deque 先頭の1件だけを対象にする。
    """
    cands: list[InterventionCandidate] = []
    mode = getattr(agent, "mode", "facilitator")

    if pending.facts:
        f = pending.facts[0]
        correction = str(f.get("correction") or "").strip()
        if correction:
            queued_at = float(f.get("_queued_at", now))
            cands.append(InterventionCandidate(
                id=f"fact-{int(queued_at * 1000)}",
                kind="fact",
                brief=correction,
                confidence=confidence_score(f.get("confidence")),
                created_at=queued_at,
                expires_at=fact_expires_at(queued_at),
                interrupt_policy="wait_for_pause",
                retryable=False,
                payload={"correction": correction, "fact": f},
            ))

    if pending.manual_call:
        m = pending.manual_call
        request = str(m.get("request") or "").strip()
        created = float(m.get("created_at", now))
        cands.append(InterventionCandidate(
            id="manual",
            kind="manual",
            brief=request or "直近の議論整理",
            created_at=created,
            expires_at=created + _MANUAL_CALL_TTL,
            interrupt_policy="wait_for_pause",
            retryable=True,
            payload={"request": request, "source": str(m.get("source") or "ui")},
        ))

    if pending.drift_reason:
        drift_created = float(pending.last_drift_request_at or now)
        cands.append(InterventionCandidate(
            id="drift",
            kind="drift",
            brief=str(pending.drift_reason),
            created_at=drift_created,
            expires_at=drift_created + _DRIFT_PENDING_TTL,
            interrupt_policy="wait_for_pause",
            retryable=True,
            payload={"drift_count": pending.drift_count},
        ))

    pi = getattr(agent, "_pending_intervention", None)
    if pi:
        cands.append(InterventionCandidate(
            id="retry",
            kind="retry",
            brief=str(pi.get("delivered", "")),
            created_at=float(pi.get("created_at", now)),
            interrupt_policy="wait_for_pause",
            retryable=True,
        ))

    # summarize 抑止規則 (設計 88f9a78): pending に af_l2 が保留されている間は
    # summarize 候補を生成しない (af_l2 が整理介入を代表する)。priority は変えない。
    # af 候補は --af 有効時しか存在しないため、ルールベースモードの挙動は不変。
    _af_l2_pending = bool(pending.af) and str(pending.af.get("kind") if pending.af else "") == "af_l2"
    if mode != "conversation" and pending.summarize and not _af_l2_pending:
        s = pending.summarize
        focus = str(s.get("focus") or "").strip()
        created = float(s.get("created_at", now))
        cands.append(InterventionCandidate(
            id="summarize",
            kind="summarize",
            brief=focus or "議論の整理",
            created_at=created,
            interrupt_policy="wait_for_pause",
            retryable=True,
            payload={"focus": focus},
        ))

    # AF ベース介入（H1 フェーズ4）。AF ランタイム有効時のみ pending.af が入る。
    if mode != "conversation" and pending.af:
        a = pending.af
        af_kind = str(a.get("kind") or "af_l1")
        af_created = float(a.get("created_at", now))
        af_ttl = _AF_L2_PENDING_TTL if af_kind == "af_l2" else _AF_L1_PENDING_TTL
        cands.append(InterventionCandidate(
            id=af_kind,
            kind=af_kind,  # type: ignore[arg-type]
            brief=str(a.get("brief") or ""),
            target_speaker=a.get("target_speaker"),
            created_at=af_created,
            expires_at=af_created + af_ttl,
            interrupt_policy="wait_for_pause",
            retryable=True,
            payload={"af_text": str(a.get("af_text") or ""), "kind": af_kind},
        ))

    if mode == "conversation" and getattr(agent, "pending_count", 0) > 0:
        cands.append(InterventionCandidate(
            id="conversation",
            kind="conversation",
            brief=f"{agent.pending_count}発話が蓄積",
            created_at=now,
            interrupt_policy="wait_for_pause",
        ))

    # 沈黙要約の閾値: プロファイルの無効化設定 (silence_summarize is None) を尊重する。
    # controlled は「沈黙要約なし」の設計なので、Partner 同席でも沈黙候補を出さない。
    # 有効な場合のみ、Partner 会話を邪魔しない意図で debate 閾値と大きい方を採る。
    if silence_summarize is None:
        silence_thresh = None
    elif partner_present:
        silence_thresh = max(silence_summarize, _AGENT_DEBATE_SILENCE)
    else:
        silence_thresh = silence_summarize
    if (mode != "conversation"
            and silence_thresh is not None
            and getattr(agent, "pending_count", 0) > 0):
        cands.append(InterventionCandidate(
            id="silence",
            kind="silence",
            brief=f"沈黙要約候補（必要{float(silence_thresh):.1f}秒）",
            created_at=now,
            interrupt_policy="wait_for_pause",
            payload={"pause_required": float(silence_thresh)},
        ))

    if pending.invite:
        invite_payload = {}
        if pending.invite == last_invited:
            invite_payload["same_as_last_invited"] = True
        cands.append(InterventionCandidate(
            id=f"invite-{pending.invite}",
            kind="invite",
            brief=f"{pending.invite}さんに声かけ",
            target_speaker=str(pending.invite),
            created_at=now,
            interrupt_policy="wait_for_pause",
            payload=invite_payload,
        ))

    return cands


def _candidate_brief(c: InterventionCandidate) -> dict:
    """レビューログ用の候補サマリ（読みやすい最小限の dict）."""
    brief = {
        "id": c.id,
        "kind": c.kind,
        "brief": c.brief[:60],
        "confidence": round(c.confidence, 3),
        "target_speaker": c.target_speaker,
    }
    if c.kind == "manual":
        # manual だけ追跡フィールドを足す（ui/voice の不発検証用）。既存形式は不変。
        brief["source"] = str(c.payload.get("source") or "ui")
        brief["request"] = str(c.payload.get("request") or "")
        brief["queued_at"] = c.created_at
    return brief


def _legacy_decision_brief(reason: str, detail: str = "") -> dict:
    """レビューログの ``legacy_decision`` 欄用に、採択された判断を最小dict化する.

    注: フィールド名 ``legacy_decision`` は Phase1（shadow で従来判断と突合していた
    頃）の名残。Phase2 以降はここに「実際に採択した Controller 判断」が入る。
    後方互換のためキー名は据え置く。
    """
    return {
        "reason": reason,
        "detail": detail,
        "at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


class _InterventionReviewRecorder:
    """採否の経緯を ``intervention_review.jsonl`` に記録する（観測性）.

    ``record()``  : 実際に dispatch へ使った Controller 判断をそのまま記録する
                    （Phase2 以降の主経路。採択・抑制・latency が追える）。
    ``evaluate()``: hold/echo/partner 待機などで dispatch しない局面向けに、
                    現時点の候補で採否を再計算してログする（物理コンテキストは
                    渡さない簡易評価）。

    採否（採択候補＋抑制集合）が前回から変化した時だけ書き出し、0.25秒ループでの
    ログ洪水を避ける。state が review ログ非対応（テストの FakeState 等）なら no-op。
    """

    def __init__(self) -> None:
        self._controller = FacilitationController()
        self._last_fingerprint: tuple | None = None
        self._warned_write_failure = False

    def evaluate(
        self,
        state: SessionState,
        *,
        pending: _PendingInterventions,
        agent,
        now: float,
        silence_elapsed: float,
        epoch: int,
        recent_interventions: list[InterventionLogEntry],
        legacy: dict | None,
        silence_summarize: float | None = None,
        partner_present: bool = False,
        last_invited: str | None = None,
    ) -> None:
        add_review = getattr(state, "add_intervention_review", None)
        if not callable(add_review):
            return  # review ログ非対応 → 記録は完全に no-op にする
        candidates = _build_candidates(
            pending, agent, now=now,
            silence_summarize=silence_summarize,
            partner_present=partner_present,
            last_invited=last_invited,
        )
        t0 = time.perf_counter()
        decision = self._controller.arbitrate(FacilitationInput(
            candidates=tuple(candidates),
            recent_interventions=tuple(recent_interventions),
            silence_elapsed=silence_elapsed,
            snapshot_epoch=epoch,
            now=now,
        ))
        latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        fingerprint = (
            decision.candidate_id,
            decision.reason,
            tuple(sorted(s["candidate_id"] for s in decision.suppressed)),
            tuple(sorted((c.id, c.kind, c.brief) for c in candidates)),
            (legacy or {}).get("reason"),
            (legacy or {}).get("detail"),
        )
        if fingerprint == self._last_fingerprint:
            return  # 採否に変化なし → 記録しない
        self._last_fingerprint = fingerprint
        try:
            add_review({
                # dispatched=False: これは hold/echo/partner 等で発話しない局面の
                # 「もし採るなら」再評価（物理コンテキスト無し）。controller_decision の
                # candidate_id は実際には dispatch されていない what-if 値（分析時に注意）。
                "dispatched": False,
                "candidates": [_candidate_brief(c) for c in candidates],
                "legacy_decision": legacy,
                "controller_decision": decision.as_dict(),
                "silence_elapsed_sec": round(silence_elapsed, 3),
                "latency_ms": latency_ms,
                "epoch": epoch,
            })
        except Exception as exc:
            if not self._warned_write_failure:
                self._warned_write_failure = True
                print(f"# [diag] intervention review log failed: {exc}", flush=True)

    def record(
        self,
        state: SessionState,
        *,
        candidates: list[InterventionCandidate],
        decision: FacilitationDecision,
        silence_elapsed: float,
        epoch: int,
        legacy: dict | None,
        latency_ms: float = 0.0,
    ) -> None:
        """実際にdispatchへ使ったController判断をreviewログへ記録する.

        Phase2では採否が実挙動を駆動するため、ログ用に再評価すると物理コンテキスト
        （echo/partner/cooldown等）の差でズレる。adapterが返した decision をそのまま
        記録する。
        """
        add_review = getattr(state, "add_intervention_review", None)
        if not callable(add_review):
            return
        fingerprint = (
            decision.candidate_id,
            decision.reason,
            tuple(sorted(s["candidate_id"] for s in decision.suppressed)),
            tuple(sorted((c.id, c.kind, c.brief) for c in candidates)),
            (legacy or {}).get("reason"),
            (legacy or {}).get("detail"),
        )
        if fingerprint == self._last_fingerprint:
            return
        self._last_fingerprint = fingerprint
        try:
            add_review({
                # dispatched=True: 実際に dispatch へ使った採否。controller_decision の
                # candidate_id が実発話に対応する（legacy_decision.reason も実 kind）。
                "dispatched": True,
                "candidates": [_candidate_brief(c) for c in candidates],
                "legacy_decision": legacy,
                "controller_decision": decision.as_dict(),
                "silence_elapsed_sec": round(silence_elapsed, 3),
                "latency_ms": round(latency_ms, 3),
                "epoch": epoch,
            })
        except Exception as exc:
            if not self._warned_write_failure:
                self._warned_write_failure = True
                print(f"# [diag] controller review log failed: {exc}", flush=True)


# 物理レーン分割（§4）。barge-in は echo/partner ガード前に評価し、
# 通常トリガーはフロア返却後に評価する。Controller はそれぞれのレーンの
# 候補集合から「採否」だけを決める（固定優先順位の置換, Phase2）。
_BARGEIN_KINDS = ("fact", "manual", "drift", "retry")
# af_l1/af_l2 は wait_for_pause なので通常レーン。AF 無効時は候補自体が出ない。
# Phase3: stall は廃止（Speaker から「介入不要」判断を外したため）。
_NORMAL_KINDS = ("summarize", "silence", "invite", "conversation", "af_l1", "af_l2")


def _suppressed_for(
    decision: FacilitationDecision,
    *,
    candidate_id: str,
    codes: tuple[str, ...],
) -> bool:
    """Controllerの抑制に、特定候補×特定コードの組が含まれるかを調べる.

    後処理の分岐は機械可読コード（SuppressionCode）のみで行い、表示文
    （reason）には依存しない（H4: 文言変更が挙動を変えないため）。
    """
    return any(
        s.get("candidate_id") == candidate_id and s.get("code") in codes
        for s in decision.suppressed
    )


def _controller_barge_in_decision(
    controller: FacilitationController,
    *,
    pending: _PendingInterventions,
    agent,
    state: SessionState,
    now: float,
    last_fact_at: float,
    last_intervention_at: float,
    silence_elapsed: float,
    partner_busy: bool,
    in_echo_window: bool,
    cooldown: float,
    recent_interventions: list[InterventionLogEntry],
    silence_summarize: float | None,
    last_invited: str | None,
    epoch: int,
) -> tuple[_BargeInDecision, FacilitationDecision | None, list[InterventionCandidate], float]:
    """barge-in レーンの採否を Controller に委ねる（固定優先順位の置換）.

    既存checkerが作った候補(fact/drift/retry)を入力にし、Controller が
    「今どれを採るか／黙るか」を決める。戻り値は既存 dispatch がそのまま使える
    ``_BargeInDecision`` へ逆変換したもの。物理タイミング（pause/partner/echo）は
    Controller の eligibility が判定する（§4）。
    """
    pending.drop_stale_facts(now=now)  # fast lane: 古い事実補正は破棄（鮮度維持）
    pending.drop_stale_drift(now=now)  # 確認待ちのまま古い脱線候補は破棄（TTL）
    expired_manual = pending.drop_stale_manual(now=now)  # 古い手動呼び出しは破棄（TTL）
    if expired_manual is not None:
        # 「呼んだのに反応しなかった」を後から追えるよう trigger ログに残す。
        request = str(expired_manual.get("request") or "").strip()
        _log_intervention_event(
            state, "manual_call_expired", request or "直近の議論整理",
            timing={"kind": "manual",
                    "source": str(expired_manual.get("source") or "ui"),
                    "request": request,
                    "queued_at": float(expired_manual.get("created_at", now)),
                    "candidate_wait_sec": round(
                        now - float(expired_manual.get("created_at", now)), 3),
                    "outcome": "expired"})
        _set_manual_status(state, "expired",
                           detail=f"{_MANUAL_CALL_TTL:.0f}秒以内に間が取れず破棄")
    all_cands = _build_candidates(
        pending, agent, now=now, silence_summarize=silence_summarize,
        partner_present=False, last_invited=last_invited)
    cands = [c for c in all_cands if c.kind in _BARGEIN_KINDS]
    if not cands:
        return _BargeInDecision("none"), None, [], 0.0
    required = int(state.proactivity.get("drift_confirmations", 1))
    t0 = time.perf_counter()
    decision = controller.arbitrate(FacilitationInput(
        candidates=tuple(cands),
        recent_interventions=tuple(recent_interventions),
        silence_elapsed=silence_elapsed,
        snapshot_epoch=epoch,
        now=now,
        cooldown=cooldown,
        partner_busy=partner_busy,
        in_echo_window=in_echo_window,
        last_intervention_at=last_intervention_at,
        required_drift_confirmations=required,
        fact_cooldown=_FACTCHECK_COOLDOWN,
    ))
    latency_ms = (time.perf_counter() - t0) * 1000
    if pending.manual_call is not None and decision.candidate_id != "manual":
        # 手動呼び出しが保留された理由をUIステータスへ（待機中の可視化）。
        held_reason = next(
            (str(s.get("reason", "")) for s in decision.suppressed
             if s.get("candidate_id") == "manual"), "")
        _set_manual_status(
            state, "waiting", detail=held_reason,
            wait_sec=now - float(pending.manual_call.get("created_at", now)))
    if _suppressed_for(
        decision,
        candidate_id="drift",
        codes=("cooldown_global", "expired", "duplicate_content"),
    ):
        # クールダウン中の脱線は「今の脱線」への対応時機を逸しており、
        # 期限切れは鮮度を失っている。同一内容（duplicate_content）は既に
        # 伝えた脱線を蒸し返すだけになる。いずれも保持し続けず忘れる。
        pending.clear_drift()
    if decision.candidate_id is None:
        # 候補はあるが今は採らない → 保持して次の機会を待つ（hold）。
        return _BargeInDecision("hold"), decision, cands, latency_ms
    chosen = next((c for c in cands if c.id == decision.candidate_id), None)
    if chosen is None:
        # Controller/LLM が候補に無い ID を返した → クラッシュせず今回は見送る (T5)。
        print(f"# [diag] controller: 不正な candidate_id={decision.candidate_id}"
              f"（バージイン見送り）", flush=True)
        return _BargeInDecision("hold"), decision, cands, latency_ms
    if chosen.kind == "fact":
        return (_BargeInDecision("fact", fact=chosen.payload.get("fact")),
                decision, cands, latency_ms)
    if chosen.kind == "manual":
        return (_BargeInDecision("manual", manual=chosen.payload),
                decision, cands, latency_ms)
    if chosen.kind == "drift":
        return (_BargeInDecision("drift", drift_reason=chosen.brief),
                decision, cands, latency_ms)
    return _BargeInDecision("retry"), decision, cands, latency_ms


def _controller_normal_decision(
    controller: FacilitationController,
    *,
    pending: _PendingInterventions,
    agent,
    now: float,
    silence_elapsed: float,
    silence_summarize: float | None,
    partner_present: bool,
    last_intervention_at: float,
    cooldown: float,
    last_invited: str | None,
    recent_interventions: list[InterventionLogEntry],
    epoch: int,
) -> tuple[_NormalTriggerDecision, FacilitationDecision | None, list[InterventionCandidate], float]:
    """通常トリガーレーンの採否を Controller に委ねる（固定優先順位の置換）.

    フロア返却後に呼ばれるため partner/echo は通過済み。count/silence/invite/
    conversation の候補から Controller が採否を決め、既存 dispatch 用の
    ``_NormalTriggerDecision`` に逆変換する。
    """
    pending.drop_stale_af(now=now)  # 古い AF 介入候補は破棄（TTL）
    all_cands = _build_candidates(
        pending, agent, now=now, silence_summarize=silence_summarize,
        partner_present=partner_present, last_invited=last_invited)
    cands = [c for c in all_cands if c.kind in _NORMAL_KINDS]
    if not cands:
        return _NormalTriggerDecision("none"), None, [], 0.0
    t0 = time.perf_counter()
    decision = controller.arbitrate(FacilitationInput(
        candidates=tuple(cands),
        recent_interventions=tuple(recent_interventions),
        silence_elapsed=silence_elapsed,
        snapshot_epoch=epoch,
        now=now,
        cooldown=cooldown,
        partner_busy=False,
        in_echo_window=False,
        last_intervention_at=last_intervention_at,
        required_drift_confirmations=0,
    ))
    latency_ms = (time.perf_counter() - t0) * 1000
    if _suppressed_for(decision, candidate_id="summarize",
                       codes=("duplicate_content",)):
        # 同一焦点の整理は既に実施済み。保持し続けると毎tick同じ抑制が続く
        # だけなので忘れる（新しい焦点の要求は drain で置き換わる）。
        pending.summarize = None
    if decision.candidate_id is None:
        # 連続声かけ抑制（同じ人を続けて誘わない）→ skip_invite で invite を消費。
        for s in decision.suppressed:
            if (s["candidate_id"].startswith("invite-")
                    and s.get("code") == "same_as_last_invited"):
                return (_NormalTriggerDecision(
                    "skip_invite", invite_target=pending.invite), decision, cands, latency_ms)
        return _NormalTriggerDecision("none"), decision, cands, latency_ms
    chosen = next((c for c in cands if c.id == decision.candidate_id), None)
    if chosen is None:
        # Controller/LLM が候補に無い ID を返した → クラッシュせず今回は見送る (T5)。
        print(f"# [diag] controller: 不正な candidate_id={decision.candidate_id}"
              f"（通常レーン見送り）", flush=True)
        return _NormalTriggerDecision("none"), decision, cands, latency_ms
    if chosen.kind == "conversation":
        return (_NormalTriggerDecision("conversation", f"沈黙{silence_elapsed:.1f}秒"),
                decision, cands, latency_ms)
    if chosen.kind == "summarize":
        return (_NormalTriggerDecision(
            "summarize", chosen.brief, summary_focus=str(chosen.payload.get("focus", ""))),
            decision, cands, latency_ms)
    if chosen.kind == "silence":
        thresh = float(chosen.payload.get("pause_required", 0.0))
        return (_NormalTriggerDecision(
            "silence", f"{silence_elapsed:.1f}>{thresh:.1f}秒"), decision, cands, latency_ms)
    if chosen.kind == "invite":
        return (_NormalTriggerDecision(
            "invite", f"{chosen.target_speaker}さんに声かけ",
            invite_target=chosen.target_speaker), decision, cands, latency_ms)
    if chosen.kind in ("af_l1", "af_l2"):
        return (_NormalTriggerDecision(
            chosen.kind, chosen.brief,
            invite_target=chosen.target_speaker,
            af_text=str(chosen.payload.get("af_text", ""))),
            decision, cands, latency_ms)
    return _NormalTriggerDecision("none"), decision, cands, latency_ms


@dataclass(frozen=True)
class _NormalSpec:
    """通常介入の発火手順のうち、**種別ごとに違う部分だけ**を持つ表.

    発火の手順そのもの（timing算出→表示→イベント記録→trigger→消費→記帳）は
    全種別で同じで、違うのはここにある5項目だけ。従来はこの手順が種別ごとに
    丸ごと書き写されており（110行の if-elif 連鎖）、枝を足すたびに記帳を
    書き忘れる事故が実際に2件起きた（handoff §22.1 の1と2）。手順を1本にし、
    差分を表に落とすことで、新しい種別は1行足すだけで済む。

    trigger: agent.trigger に渡す引数名の並び。ここから派生して決まるもの:
      - "invite_target" を渡す種別は「誰を誘ったか」を記帳する
      - "af_presentation" を渡す種別は AF ランタイムへ受容計測を通知する
    pause_from: timing の pause_required の出所。"policy"＝種別ポリシーの pause、
      "silence_summarize"＝実行時の沈黙要約しきい値（silence だけ実行時値）。
    policy: timing に載せる方針ラベル。None なら timing も表示も出さない
      （conversation は会話モードの応答であり「介入」の体裁を取らない）。
    print_limit: 表示時に detail を切り詰める長さ（af は本文が長いため40）。
    """

    trigger: tuple[str, ...]
    consume: str | None = None
    pause_from: str = "policy"
    policy: str | None = None
    print_limit: int | None = None


_NORMAL_SPECS: dict[str, _NormalSpec] = {
    "conversation": _NormalSpec(trigger=()),
    "summarize": _NormalSpec(
        trigger=("topics", "summary_focus", "recent_agent_texts"),
        consume="summarize", policy="structuring_value"),
    "silence": _NormalSpec(
        trigger=("topics", "recent_agent_texts"),
        pause_from="silence_summarize", policy="silence_summary"),
    "invite": _NormalSpec(
        trigger=("topics", "invite_target", "recent_agent_texts"),
        consume="invite", policy="participation_pause"),
    "af_l1": _NormalSpec(
        trigger=("topics", "af_presentation", "invite_target"),
        consume="af", policy="af_intervention", print_limit=40),
    "af_l2": _NormalSpec(
        trigger=("topics", "af_presentation", "invite_target"),
        consume="af", policy="af_intervention", print_limit=40),
}


def _run_agenda_detector(state: SessionState, oai_key: str, oai_model: str):
    """会議冒頭の発話から議題を1回推定してシードする（S3, --topic未指定時）.

    既に論点があれば（明示シード or 抽出済み）何もしない。十分な発話が
    たまったらLLMで議題を推定し、成功したら seed_topic して終了する。
    判断できなければ一定間隔で再試行し、論点が現れたら（topic_worker等が
    先に論点を作ったら）役目を終えて停止する。
    """
    from das.asr.live._bootstrap import detect_agenda as _detect_agenda

    _last_attempt = 0.0
    _known_epoch = getattr(state, "meeting_epoch", 0)  # 会議リセット検知用 (T3)
    while not state.stop.is_set():
        time.sleep(2)
        if not oai_key:
            return
        if not _intervention_enabled(state):
            continue
        with state.topics_lock:
            if state.topics:
                return  # 既に議題/論点あり → 役目終了
        with state.state_lock:
            epoch = state.meeting_epoch
            talk_rs = intervention_records([
                r for r in state.records
                if "speaker" in r and r.get("text")
                and r.get("speaker") != AGENT_SPEAKER
            ])
        # 会議リセット (epoch 変化) で再試行スロットルを戻す (T3)。旧会議の
        # 発話を新会議の議題としてシードしないよう、下でも副作用直前に再確認する。
        if epoch != _known_epoch:
            _known_epoch = epoch
            _last_attempt = 0.0
        if len(talk_rs) < _AGENDA_MIN_UTTS:
            continue
        if time.monotonic() - _last_attempt < _AGENDA_RETRY_SEC:
            continue
        _last_attempt = time.monotonic()
        utts = [{"speaker": intervention_speaker_name(state, r), "text": r["text"]}
                for r in talk_rs[:_AGENDA_WINDOW]]
        agenda = _detect_agenda(utts, oai_key, oai_model)
        # シード (副作用) の直前で epoch 再確認: 検出中にリセットが起きたら破棄する。
        if agenda and state.meeting_epoch == epoch:
            state.seed_topic(agenda, speaker="議題(自動)")
            _print_line(f"# 議題を自動検出してシード: {agenda}")
            return


def _run_topic_worker(state: SessionState, oai_key: str, oai_model: str):
    """論点抽出のバックグラウンドワーカー（モジュールレベル関数）."""
    from das.asr.live._bootstrap import extract_topics as _extract_topics

    while not state.stop.is_set():
        time.sleep(3)
        if not oai_key:
            continue
        if not _intervention_enabled(state):
            continue
        with state.state_lock:
            epoch = state.meeting_epoch
            talk_rs = intervention_records([
                r for r in state.records if "speaker" in r and r.get("text")
            ])
        n = len(talk_rs)
        if n - state.topic_cursor < state._TOPIC_TRIGGER:
            continue
        window = talk_rs[max(0, n - state._TOPIC_WINDOW):]
        utts = [{"speaker": intervention_speaker_name(state, r), "text": r["text"]}
                for r in window]
        with state.topics_lock:
            existing = [t["topic"] for t in state.topics]
        new_topics = _extract_topics(utts, existing, oai_key, oai_model)
        # 副作用の直前で epoch を再確認（H2）: リセットを跨いだら結果を破棄。
        with state.state_lock:
            if state.meeting_epoch != epoch:
                continue
            state.topic_cursor = n
        if new_topics:
            ms = window[-1].get("ms")
            with state.topics_lock:
                for t in new_topics:
                    if isinstance(t, dict) and "topic" in t:
                        state.topics.append({"topic": t["topic"],
                                             "speaker": t.get("speaker", "?"),
                                             "ms": ms})
            state.save()
            for t in new_topics:
                if isinstance(t, dict) and "topic" in t:
                    _print_line(f"# 💡論点: {t['topic']}（{t.get('speaker', '?')}）")


def _run_drift_checker(state: SessionState, oai_key: str, oai_model: str):
    """脱線検出のバックグラウンドワーカー（並列監視）.

    _run_topic_worker が抽出した論点(state.topics)を使い、
    直近の発話が論点からズレていないかを軽量モデルで高頻度チェック。

    人間・パートナー双方の発話をチェック対象に含める。
    パートナーが脱線に付き合っている状態も検出するため。

    R2: このワーカーは trigger() を呼ばない。脱線を検出したら理由を
    state.drift_requests キューに積むだけ。実際のトリガーは
    _run_agent_worker が一元的に行う（トリガー経路の単一化）。
    """
    from das.asr.live._bootstrap import check_drift as _check_drift

    _diag_tick = 0
    while not state.stop.is_set():
        time.sleep(1)
        _diag_tick += 1
        agent = state.agent
        if not oai_key or agent is None or not agent.enabled:
            continue
        if not _intervention_enabled(state):
            continue
        if agent.mode == "conversation":
            continue

        # 論点がまだなければスキップ
        with state.topics_lock:
            _has_topics = bool(state.topics)
            topics = list(state.topics) if _has_topics else []
        if not _has_topics:
            if _diag_tick % 30 == 0:
                print("# [drift] 待機中: 論点未抽出", flush=True)
            continue
        # ファシリテーター以外の全発話をカウント＆チェック対象にする
        with state.state_lock:
            epoch = state.meeting_epoch
            talk_rs = intervention_records([
                r for r in state.records
                if "speaker" in r and r.get("text")
                and r.get("speaker") != AGENT_SPEAKER
            ])
        n = len(talk_rs)
        # ウォームアップ: 会議開始直後の挨拶などで誤検出しないよう猶予を置く（Fix 11）
        if n < _DRIFT_WARMUP:
            if _diag_tick % 30 == 0:
                print(f"# [drift] ウォームアップ中: {n}/{_DRIFT_WARMUP}発話", flush=True)
            continue
        if n - state.drift_cursor < _DRIFT_CHECK_INTERVAL:
            continue
        # 直近の発話を取得
        window = talk_rs[max(0, n - _DRIFT_CHECK_WINDOW):]
        utts = [{"speaker": intervention_speaker_name(state, r), "text": r["text"]}
                for r in window]
        # cursor 書き戻し（副作用）の直前で epoch 確認（H2）: リセット跨ぎを破棄。
        with state.state_lock:
            if state.meeting_epoch != epoch:
                continue
            state.drift_cursor = n
        print(f"# [drift] チェック実行: {len(utts)}発話, "
              f"cursor={n}, topics={len(topics)}件", flush=True)
        # 脱線判定
        result = _check_drift(utts, topics, oai_key, oai_model)
        if result.get("drift"):
            reason = result.get("reason", "")
            # キュー投入（副作用）の直前でも epoch を確認し、リセット後の
            # 新会議に古い脱線要求が混ざらないようにする（H2）。
            with state.state_lock:
                if state.meeting_epoch != epoch:
                    continue
            _print_line(f"# 🔀 脱線検出: {reason}")
            # R2: trigger()は呼ばず、要求をキューに積む。agent_workerが裁定する。
            state.drift_requests.put(reason)
            print("# [drift] → 介入要求をキューに投入", flush=True)


def _play_ack_chime() -> None:
    """音声呼びかけを受け取ったことを短いチャイムで即時に伝える（H）.

    STT確定→triage→pause→生成で音声応答まで3〜7秒かかる間、話者は「聞こえたか」が
    分からず言い直して二重呼び出しになる。150ms程度の減衰サイン波（880Hz・控えめ音量）
    を鳴らして「聞こえた」を伝える。音は必須機能ではないので失敗は握りつぶし、triage
    ループを止めないよう daemon スレッドで再生する。
    """
    if not _ACK_CHIME_ENABLED:
        return

    def _play() -> None:
        with contextlib.suppress(Exception):
            import sounddevice as sd
            dur = 0.15
            t = np.linspace(0, dur, int(SR * dur), endpoint=False)
            envelope = np.exp(-t * 12.0)                 # なめらかな減衰
            wave = (0.2 * envelope
                    * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)
            sd.play(wave, SR)
            sd.wait()

    threading.Thread(target=_play, daemon=True).start()


def _run_triage_worker(state: SessionState, oai_key: str,
                       oai_model: str) -> None:
    """確定発話ごとに1回だけ表層分類し、record に ``triage`` 注釈を付ける（H6/M2）.

    fact prefilter の正規表現群と音声呼びかけの regex 検出を置き換える。
    ローカルで判定するのは機械的に安全なゲート（最小文字数・相槌）だけで、
    意味判定（事実断定か / ファシリテーターへの依頼か）は文脈付きの軽量 LLM
    分類に一本化する。結果は ``record["triage"]`` に格納され、fact checker が
    消費する。ファシリテーターへの依頼を検出したら手動呼び出しキューに積む
    （UIボタンと同じ経路）。

    復帰時の誤発火防止と遅延の有界化:
      - 介入オフ / conversation モード中は、未処理分を LLM を呼ばずに一括で
        負注釈（``skipped=intervention_off``）してカーソルを最新まで進める。
        こうしないと、復帰時に溜まった過去発話を順に分類し、数分前の呼びかけを
        「今」の manual_call として誤発火させてしまう（TTL が無効化される）。
      - 1 tick で処理するのは最大 ``_TRIAGE_BACKLOG_MAX`` 件。これを超える古い
        バックログは分類せず ``skipped=backlog`` で飛ばし、最新発話の呼びかけ
        遅延を有界に保つ。
    """
    from das.asr.live._bootstrap import classify_utterance as _classify

    _retry_counts: dict[int, int] = {}
    _backlog_warned = False
    _known_epoch = getattr(state, "meeting_epoch", 0)  # 会議リセット検知用 (T3)
    while not state.stop.is_set():
        time.sleep(0.25)
        agent = state.agent
        if not oai_key or agent is None or not agent.enabled:
            continue
        with state.state_lock:
            epoch = state.meeting_epoch
        # 会議リセット (epoch 変化) で index ベースのローカル状態を破棄する (T3)。
        # 旧会議の retry カウントが新会議の同 index 発話に誤適用されるのを防ぐ。
        if epoch != _known_epoch:
            _known_epoch = epoch
            _retry_counts.clear()
            _backlog_warned = False
        with state.state_lock:
            # 呼びかけ検出は話者同一性に依存しないため、未確定話者も含む
            # triage_records でフィルタする（修正5）。triage_cursor はこの
            # フィルタ後リストへのインデックスで、fact_cursor(intervention_records
            # ベース) とはリストが異なるが、fact checker は record の triage キーを
            # 直接読むため整合する（インデックスは共有しない）。
            talk_rs = triage_records([
                r for r in state.records
                if "speaker" in r and r.get("text")
                and r.get("speaker") != AGENT_SPEAKER
            ])
        n = len(talk_rs)
        cursor = state.triage_cursor
        if cursor >= n:
            continue

        # 介入オフ / conversation モード: LLM を呼ばず未処理分を一括で負注釈し、
        # カーソルを n まで進める。復帰時に溜まった過去発話を再分類して古い
        # 呼びかけを誤発火させないため（問題1）。キューには何も積まない。
        if not _intervention_enabled(state) or agent.mode == "conversation":
            with state.state_lock:
                if state.meeting_epoch != epoch:
                    continue
                for c in talk_rs[cursor:n]:
                    c["triage"] = {"factual_claim": False,
                                   "facilitator_request": "",
                                   "skipped": "intervention_off"}
                state.triage_cursor = n
            continue

        # バックログ上限を超える古い分は分類せずスキップ（遅延を有界化, 問題2）。
        backlog = n - cursor
        if backlog > _TRIAGE_BACKLOG_MAX:
            drop = backlog - _TRIAGE_BACKLOG_MAX
            with state.state_lock:
                if state.meeting_epoch != epoch:
                    continue
                for c in talk_rs[cursor:cursor + drop]:
                    c["triage"] = {"factual_claim": False,
                                   "facilitator_request": "",
                                   "skipped": "backlog"}
                cursor += drop
                state.triage_cursor = cursor
            if not _backlog_warned:
                print(f"# [triage] backlog {backlog}件: 古い{drop}件を分類せず"
                      f"スキップ（遅延を有界化, 上限{_TRIAGE_BACKLOG_MAX}）",
                      flush=True)
                _backlog_warned = True
        else:
            _backlog_warned = False

        # 残りを連続処理（tick あたり最大 _TRIAGE_BACKLOG_MAX 件）。
        processed = 0
        while processed < _TRIAGE_BACKLOG_MAX and cursor < n:
            if state.stop.is_set():
                return
            idx = cursor
            r = talk_rs[idx]
            text = str(r.get("text") or "").strip()
            # 相槌は triage_records が除外済み（未確定話者は呼びかけ検出のため
            # 含まれる, 修正5）。ここで残る機械的ゲートは「極端に短い発話」のみ
            # （LLM を呼ぶ価値がない, コスト0）。
            if len(text) < _TRIAGE_MIN_CHARS:
                annotation = {"factual_claim": False, "facilitator_request": ""}
            else:
                context = talk_rs[max(0, idx - _TRIAGE_CONTEXT_WINDOW):idx]
                utts = [
                    {"speaker": intervention_speaker_name(state, c),
                     "text": c["text"]}
                    for c in context
                ]
                utts.append({
                    "speaker": intervention_speaker_name(state, r),
                    "text": text,
                })
                result = _classify(utts, oai_key, oai_model)
                if result.get("retryable_error"):
                    tries = _retry_counts.get(idx, 0) + 1
                    _retry_counts[idx] = tries
                    if tries <= _TRIAGE_MAX_RETRIES:
                        print(f"# [triage] retry: LLM分類の一時失敗 "
                              f"{tries}/{_TRIAGE_MAX_RETRIES}", flush=True)
                        break  # 同一発話を次tickで再試行（カーソルは進めない）
                    print("# [triage] skip: LLM分類の失敗が続いたため対象発話を"
                          "スキップ", flush=True)
                    annotation = {"factual_claim": False,
                                  "facilitator_request": ""}
                else:
                    annotation = {
                        "factual_claim": _as_bool(result.get("factual_claim")),
                        "facilitator_request": str(
                            result.get("facilitator_request") or ""
                        ).strip()[:_MANUAL_CALL_MAX_CHARS],
                    }
            _retry_counts.pop(idx, None)
            # 注釈書き込み・cursor 書き戻し（副作用）の直前で epoch 確認（H2）。
            with state.state_lock:
                if state.meeting_epoch != epoch:
                    break
                r["triage"] = annotation
                state.triage_cursor = idx + 1
            cursor = idx + 1
            processed += 1
            request = str(annotation.get("facilitator_request") or "")
            # facilitate モードのみ音声呼びかけを扱う。converse（パートナー有り）
            # では通常応答に任せ、専用経路は使わない（二重応答の回避, Phase2）。
            if request and state.partner is None:
                # キュー投入（副作用）の直前でも epoch 確認（H2）。
                with state.state_lock:
                    if state.meeting_epoch != epoch:
                        break
                state.manual_call_requests.put({
                    "request": request,
                    "source": "voice",
                    "created_at": time.monotonic(),
                    "created_wall_at": datetime.datetime.now()
                    .isoformat(timespec="seconds"),
                })
                print(f"# [voice] ファシリテーター呼びかけ検出: {request}",
                      flush=True)
                _log_voice_call_diag(state, text=text, request=request)
                _set_manual_status(state, "queued", source="voice",
                                   request=request)
                # 「聞こえた」を即時に伝えるアック音（H）。UI由来のボタン呼び出しは
                # UIに既にフィードバックがあるため鳴らさない（voice経路だけ）。
                _play_ack_chime()


def _run_fact_checker(state: SessionState, oai_key: str, oai_model: str):
    """明確な事実誤りだけを短く補正する要求を積む.

    脱線や発話量とは別ルートにする。候補の選別は triage 注釈
    （``record["triage"]["factual_claim"]``, _run_triage_worker が付与）に従い、
    明確な誤りかどうかは文脈付きの LLM 判定に任せる。
    採用するのは high confidence の訂正だけ。
    """
    from das.asr.live._bootstrap import check_fact_correction as _check_fact

    _last_check = 0.0
    _recent_corrections: list[tuple[float, str]] = []
    _retry_counts: dict[int, int] = {}
    _known_epoch = getattr(state, "meeting_epoch", 0)  # 会議リセット検知用 (T3)
    while not state.stop.is_set():
        time.sleep(0.25)
        agent = state.agent
        if not oai_key or agent is None or not agent.enabled:
            continue
        if not _intervention_enabled(state):
            continue
        if agent.mode == "conversation":
            continue
        with state.state_lock:
            epoch = state.meeting_epoch
        # 会議リセット (epoch 変化) で index ベースの retry と重複補正履歴を破棄する (T3)。
        # 旧会議の補正が新会議の補正を dedup で握りつぶすのを防ぐ。
        if epoch != _known_epoch:
            _known_epoch = epoch
            _retry_counts.clear()
            _recent_corrections = []
        with state.state_lock:
            talk_rs = intervention_records([
                r for r in state.records
                if "speaker" in r and r.get("text")
                and r.get("speaker") != AGENT_SPEAKER
            ])
        n = len(talk_rs)
        if n <= state.fact_cursor:
            continue
        next_idx = state.fact_cursor
        candidate = None
        while next_idx < n:
            r = talk_rs[next_idx]
            triage = r.get("triage")
            if triage is None:
                break  # triage 分類待ち。分類済みの位置までで止まり、次tickで再開
            if triage.get("factual_claim"):
                candidate = r
                break
            next_idx += 1
        if candidate is None:
            # cursor 書き戻し（副作用）の直前で epoch 確認（H2）。
            with state.state_lock:
                if state.meeting_epoch == epoch:
                    state.fact_cursor = next_idx
            continue
        now = time.monotonic()
        if now - _last_check < _FACTCHECK_CHECK_SEC:
            continue
        _last_check = now
        context = talk_rs[max(0, next_idx - 3):next_idx]
        utts = [
            {"speaker": intervention_speaker_name(state, r), "text": r["text"]}
            for r in context
        ]
        utts.append({
            "speaker": intervention_speaker_name(state, candidate),
            "text": candidate["text"],
        })
        result = _check_fact(utts, oai_key, oai_model)
        if result.get("retryable_error"):
            tries = _retry_counts.get(next_idx, 0) + 1
            _retry_counts[next_idx] = tries
            if tries <= _FACTCHECK_MAX_RETRIES:
                print(f"# [fact] retry: LLM判定の一時失敗 {tries}/{_FACTCHECK_MAX_RETRIES}",
                      flush=True)
                continue
            print("# [fact] skip: LLM判定の失敗が続いたため対象発話をスキップ", flush=True)
        # cursor 書き戻し（副作用）の直前で epoch 確認（H2）: リセット跨ぎを破棄。
        with state.state_lock:
            if state.meeting_epoch != epoch:
                continue
            state.fact_cursor = next_idx + 1
        _retry_counts.pop(next_idx, None)
        # 採用するのは high confidence の訂正だけ (docstring と一致)。confidence が
        # 欠落・不正値なら安全側で発火しない。低確度の訂正を対面議論に流さない。
        _fact_confidence = str(result.get("confidence") or "").strip().lower()
        if _as_bool(result.get("should_correct")) and _fact_confidence == "high":
            correction = str(result.get("correction") or "").strip()
            if correction:
                norm = re.sub(r"[\s、。,.，．!！?？]+", "", correction).lower()
                _recent_corrections = [
                    (t, c) for t, c in _recent_corrections if now - t < 90.0
                ]
                if any(c == norm for _, c in _recent_corrections):
                    print("# [fact] skip: 重複する補正", flush=True)
                    continue
                # キュー投入（副作用）の直前でも epoch 確認（H2）。
                with state.state_lock:
                    if state.meeting_epoch != epoch:
                        continue
                _recent_corrections.append((now, norm))
                result["_queued_at"] = time.monotonic()
                _print_line(f"# ✅ 事実補正候補: {correction}")
                state.factcheck_requests.put(result)
                print("# [fact] → 補正要求をキューに投入", flush=True)


def _run_participation_checker(state: SessionState, oai_key: str, oai_model: str):
    """発話量の偏りを監視し、発言の少ない人への声かけ要求を積む（S4）.

    決定的に算出した参加度（時間シェア/回数）で「明らかに静かな人がいるか」を
    事前ゲートし、いる時だけ軽量LLMに最終判断（誰に声をかけるか）を委ねる。
    invite=true なら対象話者名を state.invite_requests に積む。実際の発話タイミングは
    _run_agent_worker が「沈黙の間」で裁定する（人間を割り込まない）。
    """
    from das.asr.live._bootstrap import check_participation as _check

    _skip = (AGENT_SPEAKER, "パートナー")
    _last_check = 0.0
    while not state.stop.is_set():
        time.sleep(1)
        agent = state.agent
        if not oai_key or agent is None or not agent.enabled:
            continue
        if not _intervention_enabled(state):
            continue
        if agent.mode == "conversation":
            continue
        with state.state_lock:
            talk_rs = intervention_records([
                r for r in state.records
                if "speaker" in r and r.get("text")
                and r.get("speaker") not in _skip
            ])
        if len(talk_rs) < _INVITE_WARMUP:
            continue
        if time.monotonic() - _last_check < _INVITE_CHECK_SEC:
            continue
        reliable_rs = reliable_human_records(talk_rs)
        stats = participation_stats(reliable_rs, exclude_speakers=_skip)
        if len(stats) < 2:
            continue  # 信頼できる参加者が2人未満なら声かけの意味がない
        # 事前ゲート: 公平シェアの_INVITE_QUIET_RATIO未満の人がいる時だけLLMを呼ぶ
        equal = 1.0 / len(stats)
        if quietest_participation_share(stats) >= equal * _INVITE_QUIET_RATIO:
            continue
        _last_check = time.monotonic()
        now_ms = max((d["last_end_ms"] for d in stats.values()
                      if d["last_end_ms"] is not None), default=None)
        participation = []
        valid_invite_targets: set[str] = set()
        share_key = participation_share_key(stats)
        share_label = participation_share_label(share_key)
        for sp, d in stats.items():
            silent = ((now_ms - d["last_end_ms"]) / 1000.0
                      if now_ms is not None and d["last_end_ms"] is not None else 0.0)
            speaker_name = state.disp_name(sp)
            valid_invite_targets.add(speaker_name)
            participation.append({"speaker": speaker_name,
                                  "time_share": d["time_share"],
                                  "participation_share": d[share_key],
                                  "participation_share_label": share_label,
                                  "turns": d["turns"], "silent_sec": silent})
        window = talk_rs[-_DRIFT_CHECK_WINDOW:]
        utts = [{"speaker": intervention_speaker_name(state, r), "text": r["text"]}
                for r in window]
        result = _check(participation, utts, oai_key, oai_model)
        if result.get("invite") and result.get("speaker"):
            target = result["speaker"]
            if target not in valid_invite_targets:
                print(f"# [invite] skip: 信頼できる参加者名ではない target={target}",
                      flush=True)
                continue
            _print_line(f"# 🙋 声かけ候補: {target}（{result.get('reason', '')}）")
            state.invite_requests.put(target)
            print("# [invite] → 声かけ要求をキューに投入", flush=True)


def _run_structuring_checker(state: SessionState, oai_key: str,
                             oai_model: str) -> None:
    """整理介入の価値判定チェッカー（C3, count の無条件介入を置換）.

    「N発話たまったら無条件に介入」をやめ、pending_count が trigger_n に達し、
    かつ前回判定時から発話が進んでいる時だけ、LLMに「今、短い整理の介入が議論に
    価値を足すか」を判定させる。intervene=true のときだけ state.summarize_requests
    に focus を積む。実際の発話タイミングは _run_agent_worker が裁定する。

    「なぜ黙ったか」の追跡は本研究の核なので、false 判定もログに1行残す。
    """
    from das.asr.live._bootstrap import check_summary_value as _check

    _last_judged_count = 0
    while not state.stop.is_set():
        time.sleep(1)
        agent = state.agent
        if not oai_key or agent is None or not agent.enabled:
            continue
        if not _intervention_enabled(state):
            continue
        if agent.mode == "conversation":
            continue
        pending_count = getattr(agent, "pending_count", 0)
        trigger_n = getattr(agent, "trigger_n", 0)
        if trigger_n <= 0 or pending_count < trigger_n:
            # 蓄積が閾値未満 = trigger/リセットで消費された。次に閾値へ達したとき
            # 再判定できるよう高水位マークを戻す（さもないと介入は一度きりになる）。
            _last_judged_count = 0
            continue
        # 同じ蓄積量での再判定の連打を防ぐ（発話が進んだ時だけ判定する）。
        if pending_count <= _last_judged_count:
            continue
        _last_judged_count = pending_count
        with state.state_lock:
            epoch = state.meeting_epoch
            talk_rs = intervention_records([
                r for r in state.records
                if "speaker" in r and r.get("text")
                and r.get("speaker") != AGENT_SPEAKER
            ])
        with state.topics_lock:
            topics = list(state.topics) if state.topics else []
        window = talk_rs[-_STRUCTURING_WINDOW:]
        utts = [{"speaker": intervention_speaker_name(state, r), "text": r["text"]}
                for r in window]
        result = _check(utts, topics, oai_key, oai_model)
        if not result.get("intervene"):
            # 「なぜ黙ったか」を追跡できるよう、見送りも1行残す。
            print("# [structuring] skip: 介入価値なし", flush=True)
            continue
        focus = str(result.get("focus") or "").strip()
        # キュー投入（副作用）の直前で epoch 再確認（H2）: リセット跨ぎを破棄。
        with state.state_lock:
            if state.meeting_epoch != epoch:
                continue
        _print_line(f"# 🧭 整理介入の価値あり: {focus or '（焦点なし）'}")
        state.summarize_requests.put({"focus": focus})
        print("# [structuring] → 整理介入の要求をキューに投入", flush=True)


def _af_l1_presentation(decision: Any) -> str:
    """AF L1 decision の items を関係ラベル付き提示文に整文する（H1 フェーズ4）."""
    label = {"support": "支持", "attack": "反論"}
    who = decision.addressed_to
    head = f"{who}さんの先ほどの発言に対する関連情報:" if who else "関連情報:"
    lines = [head]
    for it in decision.items:
        tag = label.get(it.relation, "参考")
        lines.append(f"- [{tag}] {it.source_text}")
    return "\n".join(lines)


# af_l2 再発火ガード: 前回 af_l2 以降にグラフへ新規発話ノードがこれだけ追加される
# まで次の af_l2 を出さない (cooldown とは別の「状態が十分変化したか」条件)。
_AF_L2_MIN_NEW_NODES = 4
# 同種理由バックオフの上限。同じ理由タイプで状態が改善しないまま再発火が続くほど
# 必要新規ノード数を倍化する (4→8→16)。理由タイプが変わるか応答エッジ検出でリセット。
# 「無視されている介入を同じ調子で繰り返さない」(B4)。
_AF_L2_MAX_NEW_NODES = 16
_AF_L2_MAX_LEVEL = 2  # 4 * 2**2 = 16 で上限に達する


def _af_l2_reason_type(reason: str) -> str:
    """af_l2 の理由文字列を粗い理由タイプに分類する (同種理由バックオフ用)。"""
    if "停滞" in reason:
        return "stalled"
    if "未応答" in reason or "偏り" in reason:
        return "bias"
    return "other"


def _af_checker_tick(
    state: SessionState, facil: Any, presented: set[str],
    af_gate: dict[str, Any] | None = None,
) -> int:
    """AF checker の 1 周分。af 候補を state.af_requests に積み、積んだ件数を返す。

    ``facil`` は :class:`FacilitationAgent`。``presented`` は提示済み source_text 集合
    (呼び出し側が meeting 世代ごとに保持)。``af_gate`` は af_l2 再発火ガードの状態
    ({"last_l2_node_count": int, "last_l2_reason_type": str, "l2_required_nodes": int,
    "last_l2_resp_count": int})。テスト容易性のため 1 周を関数化してある。
    """
    from das.types import Utterance

    if af_gate is None:
        af_gate = {}

    runtime = getattr(state, "af_runtime", None)
    if runtime is None:
        return 0  # AF 無効 (既定): 何もしない
    agent = state.agent
    if agent is None or getattr(agent, "mode", "facilitator") == "conversation":
        return 0
    epoch = state.meeting_epoch
    with state.state_lock:
        talk_rs = intervention_records([
            r for r in state.records
            if "speaker" in r and r.get("text")
            and r.get("speaker") != AGENT_SPEAKER
        ])
    if not talk_rs:
        return 0
    transcript = [
        Utterance(turn_id=i + 1, speaker=intervention_speaker_name(state, r),
                  text=str(r["text"]))
        for i, r in enumerate(talk_rs)
    ]
    try:
        store = runtime.store
        decision = facil.decide_intervention(transcript, store)
        if decision.kind == "l1":
            decision = facil.apply_l1_value_gate(
                decision, store, transcript, presented_source_texts=presented)
    except Exception as exc:  # pragma: no cover - 防御的
        print(f"# [af] decide error: {exc}", flush=True)
        return 0
    if decision.kind == "skip":
        return 0
    # 副作用（キュー投入）の直前で epoch 再確認（H2）。
    with state.state_lock:
        if state.meeting_epoch != epoch:
            return 0
    if decision.kind == "l1":
        for it in decision.items:
            presented.add(it.source_text.strip())
        state.af_requests.put({
            "kind": "af_l1",
            "brief": decision.reason,
            "af_text": _af_l1_presentation(decision),
            "target_speaker": decision.addressed_to,
        })
        print(f"# [af] → af_l1 候補を投入（{len(decision.items)}件）", flush=True)
        return 1
    # af_l2 再発火ガード + 同種理由バックオフ (課題2)。
    # 前回 af_l2 以降に新規発話ノードが「必要数」追加されるまで出さない。必要数は、
    # 同じ理由タイプで状態が改善しない (応答エッジも増えない) まま再発火が続くほど
    # 倍化する (4→8→16)。理由タイプが変わるか、応答エッジが増えたらベース(4)に戻す。
    n_utt = sum(1 for node in store.nodes() if node.source == "utterance")
    reason_type = _af_l2_reason_type(decision.reason)
    n_resp = len(getattr(runtime, "_response_edges", []) or [])
    last = af_gate.get("last_l2_node_count")
    last_type = af_gate.get("last_l2_reason_type")
    last_resp = int(af_gate.get("last_l2_resp_count", 0))
    level = int(af_gate.get("l2_backoff_level", 0))
    # 理由タイプが変わった / 応答エッジが増えた (介入が届いた) → バックオフをリセット。
    reset = last is None or reason_type != last_type or n_resp > last_resp
    if reset:
        level = 0
    required = min(_AF_L2_MIN_NEW_NODES * (2 ** level), _AF_L2_MAX_NEW_NODES)
    if last is not None and n_utt - last < required:
        print(
            f"# [af] af_l2 skip: 前回af_l2以降の新規発話ノード不足 "
            f"({n_utt - last}/{required}, 理由={reason_type}, lv={level})",
            flush=True,
        )
        return 0
    # 発火。同種理由・未改善のまま再発火した場合だけ、次回の必要数を倍化する。
    next_level = 0 if reset else min(level + 1, _AF_L2_MAX_LEVEL)
    af_gate["last_l2_node_count"] = n_utt
    af_gate["last_l2_reason_type"] = reason_type
    af_gate["last_l2_resp_count"] = n_resp
    af_gate["l2_backoff_level"] = next_level
    state.af_requests.put({
        "kind": "af_l2",
        "brief": decision.reason,
        "af_text": decision.brief,
        "target_speaker": None,
    })
    print(f"# [af] → af_l2 候補を投入 (理由={reason_type}, 必要数={required})", flush=True)
    return 1


class _AfEarlyGenGate:
    """af 介入の生成先行・再生ゲートの状態機械 (フェーズ6, **af 限定**).

    毎ループ :meth:`tick` を呼ぶ。時計・沈黙・Controller の採否状態を引数で受けるので、
    フェイク時計で状態遷移を単体テストできる。summarize 等ルールベース種別には一切
    関与しない (モード方針: 対象は af_l1/af_l2 のみ)。

    ``status`` は Controller から得た af 候補の採否状態 (三層分離の「WHEN」を委譲):
      - ``"deliver"`` : cooldown/arbitration/フロアを通過し **今** 配信してよい (pause 成立)
      - ``"hold"``    : ``awaiting_pause`` のみで抑制 = 採択見込みだが間待ち → 生成先行の対象
      - ``"none"``    : af 候補なし / cooldown・期限切れ・他候補優先などで見送り

    遷移:
      - 未 hold & status=hold & agent フリー & 沈黙>=0.3 → trigger(hold) で生成先行
      - 未 hold & status=deliver & agent フリー & 沈黙>=0.3 → 即時 trigger (取り込み遅延で
        pause 通過後に候補が来たケース。生成先行の余地なし。hold_to_release は付かない)
      - hold 中 & status=deliver (フロア成立) → release_playback で一斉再生
      - hold 中 & 新規確定発話 (会話が動いた) → cancel_held で破棄 (リトライにしない)
      - hold 中 & 保持時間が上限超過 → cancel_held (フロアが返らないまま抱え込まない, B4)
    """

    EARLY_GEN_SILENCE = 0.3
    MAX_HOLD_SEC = 8.0  # フロアが返らないまま生成先行を抱え込む上限 (安全弁)

    def __init__(self) -> None:
        self._holding = False
        self._held_kind: str | None = None
        self._held_af: dict[str, Any] | None = None
        self._held_since: float | None = None
        self._last_release_ms: float | None = None

    @property
    def is_holding(self) -> bool:
        return self._holding

    @property
    def last_release_ms(self) -> float | None:
        """直近の release で計測した hold→再生の所要 ms (未 release なら None)。"""
        return self._last_release_ms

    def reset(self) -> None:
        self._holding = False
        self._held_kind = None
        self._held_af = None
        self._held_since = None

    def _fire(self, agent: Any, af: dict[str, Any], *, hold: bool, topics: Any) -> None:
        agent.trigger(
            topics=topics,
            af_presentation=str(af.get("af_text") or ""),
            invite_target=af.get("target_speaker"),
            hold_playback=hold,
        )

    def tick(
        self, *, agent: Any, af: dict[str, Any] | None, status: str,
        silence: float, new_utterance: bool, agent_busy: bool,
        now: float | None = None, topics: Any = None,
    ) -> str:
        """1 ループ分の処理。行った操作名 (trigger/deliver/release/cancel/holding/none) を返す。"""
        if self._holding:
            # フロア成立前に新規確定発話が来たら、生成先行を破棄する (リトライにしない)。
            if new_utterance:
                agent.cancel_held()
                self.reset()
                return "cancel"
            # 候補が消えた (TTL 失効・応答済みなど) → 破棄。
            if not af:
                agent.cancel_held()
                self.reset()
                return "cancel"
            if status == "deliver":  # フロア成立 → 貯めた音声を一斉再生
                agent.release_playback()
                self._last_release_ms = getattr(agent, "last_hold_to_release_ms", None)
                self.reset()
                return "release"
            # フロアが返らないまま抱え込まない (安全弁)。
            if (now is not None and self._held_since is not None
                    and now - self._held_since > self.MAX_HOLD_SEC):
                agent.cancel_held()
                self.reset()
                return "cancel"
            return "holding"
        # 未 hold。生成先行は agent フリー & 沈黙が最小値以上のときだけ検討する。
        if agent_busy or silence < self.EARLY_GEN_SILENCE or not af:
            return "none"
        if status == "hold":  # 採択見込み・間待ち → 生成先行 (hold)
            self._fire(agent, af, hold=True, topics=topics)
            self._holding = True
            self._held_kind = str(af.get("kind") or "af_l1")
            self._held_af = af
            self._held_since = now
            return "trigger"
        if status == "deliver":  # 既に pause 成立 → 生成先行の余地なく即時配信
            self._fire(agent, af, hold=False, topics=topics)
            self._held_kind = str(af.get("kind") or "af_l1")
            return "deliver"
        return "none"


def _af_gate_status(
    controller: FacilitationController,
    pending: _PendingInterventions,
    agent: Any,
    *,
    now: float,
    silence_elapsed: float,
    recent_interventions: list[InterventionLogEntry],
    cooldown: float,
    last_intervention_at: float,
    epoch: int,
    partner_busy: bool,
    in_echo_window: bool,
) -> tuple[str, dict[str, Any] | None]:
    """af 候補の採否状態を Controller に問い、ゲート駆動用の status を返す (WHEN を委譲).

    三層分離: WHAT/WHOM は AF (decide_intervention)、WHEN は Controller (cooldown/
    arbitration/フロア) が決める。ゲートは再生タイミングだけを担う。ここで Controller
    の ``arbitrate`` を af 候補にかけ、

      - 採択 (candidate_id が af)         → ``"deliver"`` (pause 成立・今出してよい)
      - 何も採らず af が awaiting_pause のみ → ``"hold"``    (採択見込み・間待ち)
      - それ以外 (cooldown/期限切れ/他優先) → ``"none"``

    を返す。af 候補が無い / conversation モードなら ``("none", None)``。af 候補は
    ``--af`` 有効時しか積まれないので、ルールベース経路には影響しない。
    """
    if not pending.af or getattr(agent, "mode", "facilitator") == "conversation":
        return "none", None
    cands = [c for c in _build_candidates(pending, agent, now=now)
             if c.kind in ("af_l1", "af_l2")]
    if not cands:
        return "none", None
    af_cand = cands[0]
    decision = controller.arbitrate(FacilitationInput(
        candidates=tuple(cands),
        recent_interventions=tuple(recent_interventions),
        silence_elapsed=silence_elapsed,
        snapshot_epoch=epoch,
        now=now,
        cooldown=cooldown,
        partner_busy=partner_busy,
        in_echo_window=in_echo_window,
        last_intervention_at=last_intervention_at,
        required_drift_confirmations=0,
    ))
    if decision.candidate_id == af_cand.id:
        return "deliver", pending.af
    if _suppressed_for(decision, candidate_id=af_cand.id, codes=("awaiting_pause",)):
        return "hold", pending.af
    return "none", pending.af


def _run_af_checker(state: SessionState, *, interval: float = 3.0) -> None:
    """AF ベース介入の候補生成ワーカー（H1 フェーズ4, AF ランタイム有効時のみ）.

    ``state.af_runtime`` (run_af_runtime がセット) が育てた AF を数秒周期で読み、
    ``decide_intervention`` + L1 価値ゲートで介入候補を決めて ``state.af_requests``
    に積む。trigger は行わず、既存の _run_agent_worker が Controller 採否を通す。
    AF ランタイムが無い (既定 OFF) なら何もしない = ルールベース挙動は不変。
    """
    from das.agents.facilitation import FacilitationAgent

    facil = FacilitationAgent(llm=None)  # decide_intervention は LLM を呼ばない (決定的)
    presented: set[str] = set()
    af_gate: dict[str, Any] = {}
    epoch = state.meeting_epoch
    while not state.stop.is_set():
        time.sleep(interval)
        if not _intervention_enabled(state):
            continue
        if state.meeting_epoch != epoch:
            epoch = state.meeting_epoch
            presented = set()
            af_gate = {}
            facil.reset()
        _af_checker_tick(state, facil, presented, af_gate)


def _on_agent_text_factory(state: SessionState):
    """ファシリテーター発言コールバックを生成."""
    def _on_agent_text(text: str):
        from das.asr.live import ON_UTTERANCE

        text = text.strip()
        with state.state_lock:
            state.records.append({"ms": None, "end_ms": None,
                                  "speaker": AGENT_SPEAKER, "text": text})
            state.color_of(AGENT_SPEAKER)
        if ON_UTTERANCE is not None:
            with contextlib.suppress(Exception):
                ON_UTTERANCE("ファシリテーター", text)
        # 観測: trigger → 発話開始の遅延（§3.5 予算検証用）。属性が無い agent でも安全。
        timing = None
        speak_latency = getattr(state.agent, "_last_speak_latency_ms", None)
        if speak_latency is not None:
            timing = {"speak_start_latency_ms": speak_latency}
        state.add_facilitator_delivery_event(text, timing=timing)
        _print_line(f"\x1b[96m[ファシリテーター]\x1b[0m: {text}")
        state.save()
    return _on_agent_text


def _run_facilitator_event_worker(state: SessionState, on_text):
    """ファシリテーター発言の副作用を受信スレッドから切り離して処理する.

    agentの受信スレッドが、議事録のファイルI/Oや partner の WebSocket 送信で
    ブロックしないよう、副作用はこのワーカーに委譲する（state.fac_events 経由）。
    イベントは FIFO で処理され、順序（speech_start→utterance）は保たれる。
    state.partner / state.simulator は動的参照（実行中の接続/切断に追従, F3）。
    """
    while not state.stop.is_set():
        try:
            kind, text = state.fac_events.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            if kind == "utterance" and text is not None:
                on_text(text)
                if "介入不要" not in text:
                    p = state.partner
                    if p is not None and p._connected:
                        p.interrupt()
                        p.inject_context("ファシリテーター", text)
                    sim = state.simulator
                    if sim is not None:
                        sim.inject_facilitator(text)
            elif kind == "speech_start":
                p = state.partner
                if p is not None and p._connected and (p.ai_speaking or p._responding):
                    p.interrupt()
        except Exception as e:
            print(f"# ファシリテーターイベント処理エラー: {e}", flush=True)


def _connect_agent(state: SessionState, on_text):
    """ファシリテーターAgentのコールバック設定・接続・ワーカー起動."""
    agent = state.agent
    if agent is None:
        return
    # 受信スレッドはイベントを積むだけ。副作用は専用ワーカーで処理（受信ブロック回避）。
    agent.on_ai_utterance = lambda text: state.fac_events.put(("utterance", text))

    def _agent_speech_start() -> None:
        # AI再生区間を開き（P2-1）、従来のイベント通知も行う。
        state.note_ai_speech_start("agent")
        state.fac_events.put(("speech_start", None))

    agent.on_speech_start = _agent_speech_start
    agent.on_speech_end = lambda: state.note_ai_speech_end("agent")
    threading.Thread(target=_run_facilitator_event_worker,
                     args=(state, on_text), daemon=True).start()

    agent.connect()
    threading.Thread(target=_run_agent_worker, args=(state,),
                     daemon=True).start()
    print(f"# AI Agent: mode={agent.mode} voice={agent.voice}"
          f" trigger={agent.trigger_n}（ブラウザから変更可能）", flush=True)


def _on_partner_text_factory(state: SessionState):
    """Partner発言コールバックを生成."""
    def _on_partner_text(text: str):
        with state.state_lock:
            state.records.append({"ms": None, "end_ms": None,
                                  "speaker": "パートナー", "text": text.strip()})
            state.color_of("パートナー")
        _print_line(f"\x1b[93mパートナー\x1b[0m: {text.strip()}")
        state.save()
        state._last_utt_time[0] = time.monotonic()
        if state.agent is not None:
            state.agent.feed("パートナー", text, trigger_count=False)
    return _on_partner_text


# ---------------------------------------------------------------------------
# 実行中のモード切替（F3）
# ---------------------------------------------------------------------------

def _attach_partner(state: SessionState):
    """AIパートナーを生成・接続して state.partner にセットする（会話モード）."""
    if state.partner is not None:
        return  # 既に接続済み
    cfg = state._partner_cfg or {}
    if not cfg.get("api_key"):
        _print_line("# 会話モードにできません（OPENAI_API_KEYが未設定）")
        return
    from das.asr.live.agents._partner import ConversationPartner
    p = ConversationPartner(api_key=cfg["api_key"],
                            voice=cfg.get("voice") or "echo",
                            topic=cfg.get("topic") or "")
    if state.tracker is not None:
        p.set_tracker(state.tracker)
    p.on_ai_utterance = _on_partner_text_factory(state)
    p.connect()
    state.partner = p
    _print_line("# モード: AIと会話（パートナーを接続）")


def _detach_partner(state: SessionState):
    """AIパートナーを切断して state.partner を None にする."""
    p = state.partner
    if p is None:
        return
    # 切断でテキストエコー防御が即消えるのを防ぐ（P2-4）。直前の応答テキストを
    # TTL 内だけ退役エコー参照に残す。声紋 __PARTNER__ は tracker 側に残るため対応不要。
    with contextlib.suppress(Exception):
        state.add_retired_echo_texts(list(getattr(p, "_recent_ai_texts", [])))
    state.partner = None  # 先にNoneにして利用側(動的参照)を即座に切り離す
    with contextlib.suppress(Exception):
        p.close()
    _print_line("# パートナーを切断しました")


def set_session_mode(state: SessionState, mode: str) -> dict:
    """セッションモードを切り替える（transcribe / converse / facilitate）.

    エージェントのon/offとパートナーの接続/切断をまとめて行う。
    戻り値: {"ok": bool, "mode": str} or {"ok": False, "error": str}
    """
    if mode not in ("transcribe", "converse", "facilitate"):
        return {"ok": False, "error": f"未知のモード: {mode}"}
    if state.agent is None:
        return {"ok": False,
                "error": "AIエージェントが無効です（--agentで起動してください）"}
    if mode == "transcribe":
        state.agent.apply_config(mode="off")
        _detach_partner(state)
    elif mode == "facilitate":
        state.agent.apply_config(mode="facilitator")
        _detach_partner(state)
    else:  # converse
        state.agent.apply_config(mode="facilitator")
        _attach_partner(state)
    state.save()
    return {"ok": True, "mode": state.session_mode()}


def _run_agent_worker(state: SessionState):
    """バックグラウンドでAI応答のトリガーを管理（ターンテイキング）.

    自然な会話のフロア交代を模倣:
      - 人間のターン: 発話を即座にfeed、沈黙で譲渡 → AIがtrigger
      - AIのターン: 応答を再生。人間の実質的な発話で自動interrupt
      - AIターン終了: フロアを人間に返す（沈黙タイマーをリセット）
    """
    agent = state.agent
    _last_utt_time = state._last_utt_time
    _was_in_echo = state._was_in_echo
    _diag_tick = 0
    _last_intervention_at = 0.0  # 直近の介入時刻（脱線介入のクールダウン用）
    _pending = _PendingInterventions()
    _last_fact_at = 0.0
    _last_invited: str | None = None    # 直近に声をかけた相手（連続回避）
    _last_agent_reconnect_at = 0.0
    # --- 採否Controller: 固定優先順位に代わり最終採否を担当する ---
    # 物理タイミング（floor/barge-in）と fact fast lane は維持しつつ、
    # 「どの候補を今採るか／黙るか」を Controller が一元裁定する。
    _controller = FacilitationController()
    # af 生成先行・再生ゲート (フェーズ6, --af 有効時のみ作動する。af 限定)。
    _af_gate = _AfEarlyGenGate()
    _af_held_text = ""
    _af_held_kind = "af_l1"
    # 生成先行(hold)中の af が指名している相手。release 時に「誰を誘ったか」を
    # 記録するために保持する（保持しないと同じ人を連続で指名しうる）。
    _af_held_target: str | None = None
    # 採否の経緯（採択/抑制/latency）を intervention_review.jsonl へ記録する。
    _review = _InterventionReviewRecorder()
    # maxlen はクールダウン照会に加えて同一内容抑止（duplicate_content, 10分窓）
    # の照会範囲を兼ねる。介入は最短でも十数秒間隔なので 30 件で窓を十分覆う。
    _recent_interventions: collections.deque[InterventionLogEntry] = (
        collections.deque(maxlen=30))

    def _note_intervention(at: float, kind: str, detail: str = "",
                           *, invite_target: str | None = None) -> None:
        """実際に発火した介入を記録する（発火の副作用はここに一本化する）.

        記録するのは3つ:
          - 直近履歴 `_recent_interventions`（同一内容抑止 duplicate_content 用）
          - `_last_intervention_at`（global scope の cooldown の時計）
          - `_last_invited`（同じ人への連続声かけの抑止。指名した時だけ）

        **時計と声かけ相手をここで進めるのは、分岐ごとの更新漏れを構造的に防ぐため**。
        発火分岐は種別ごとの巨大な if-elif 連鎖になっており、実際に漏れが2件起きた
        （2026-07-25 の監査）:
          - `conversation` だけ `_last_intervention_at` を更新しておらず、会話モードで
            AIが応答した直後に、mode で gate されない `invite` 候補が古い時刻を基準に
            global cooldown を通過して続けざまに喋っていた
          - `af_l1`/`af_l2` は `invite_target` を渡して実際に人を指名するのに
            `_last_invited` を更新せず、同じ人への連続指名の抑止が効いていなかった
        以後、新しい発火分岐を足すときは `_note_intervention` を呼ぶだけでよい。
        """
        nonlocal _last_intervention_at, _last_invited
        _recent_interventions.append(
            InterventionLogEntry(at=at, kind=kind, brief=detail))
        _last_intervention_at = at
        if invite_target:
            _last_invited = invite_target

    def _fire_normal(decision: _NormalTriggerDecision, silence_elapsed: float,
                     silence_summarize) -> None:
        """通常介入を1件発火する（種別によらず手順は同じ。差分は _NORMAL_SPECS）.

        手順: timing算出 → 表示 → イベント記録 → agent.trigger → 種別ごとの
        後始末（AF受容計測・pending消費）→ 記帳。記帳は _note_intervention に
        一本化してあるので、cooldown の時計と「誰を誘ったか」は種別を問わず
        必ず更新される（handoff §22.1 の再発防止）。
        """
        kind = decision.reason
        if kind == "skip_invite":
            _pending.invite = None   # 同じ人を連続では誘わない（発話はしない）
            return
        spec = _NORMAL_SPECS.get(kind)
        if spec is None:
            return                   # "none" 等、発火しない判断
        timing = None
        if spec.policy is not None:
            pause = (float(silence_summarize or 0.0)
                     if spec.pause_from == "silence_summarize"
                     else policy_for(kind).pause)
            timing = _intervention_timing_metadata(
                kind=kind, now=time.monotonic(), silence_elapsed=silence_elapsed,
                pause_required=pause, policy=spec.policy)
            shown = (decision.detail[:spec.print_limit] if spec.print_limit
                     else decision.detail)
            print(f"# [trigger] {kind}: {shown}", flush=True)
        _log_intervention_event(state, kind, decision.detail, timing=timing)
        available = {
            "topics": _topics,
            "summary_focus": decision.summary_focus,
            "invite_target": decision.invite_target,
            "af_presentation": decision.af_text,
            "recent_agent_texts": _recent_agent_texts(state),
        }
        agent.trigger(**{k: available[k] for k in spec.trigger})
        if "af_presentation" in spec.trigger and decision.af_text:
            # 受容計測 (フェーズ5): 配信した af 介入を AF ランタイムに記録する。
            _af_rt = getattr(state, "af_runtime", None)
            if _af_rt is not None:
                with contextlib.suppress(Exception):
                    _af_rt.note_intervention(kind, decision.af_text)
        if spec.consume == "af":
            _pending.clear_af()
        elif spec.consume == "summarize":
            _pending.summarize = None   # 採択したら消費（drainで再取得しない）
        elif spec.consume == "invite":
            _pending.invite = None
        _note_intervention(
            time.monotonic(), kind, decision.detail,
            invite_target=(decision.invite_target
                           if "invite_target" in spec.trigger else None))

    while not state.stop.is_set():
        time.sleep(0.25)
        _diag_tick += 1
        _af_new_utt = False  # このtickで新規確定発話が来たか (af 生成先行のcancel用)
        partner = state.partner  # 動的参照: 実行中のパートナー接続/切断に追従（F3）
        if agent is None or not agent._connected or not agent.enabled:
            if agent is None or not agent.enabled:
                _pending.clear_all()
            if agent is not None and agent.enabled and not agent._connected:
                now = time.monotonic()
                if now - _last_agent_reconnect_at >= 5.0:
                    _last_agent_reconnect_at = now
                    print("# AI Agent: 再接続を試みます", flush=True)
                    with contextlib.suppress(Exception):
                        agent.connect()
            if _diag_tick % 20 == 0:
                print(f"# [diag] _agent_worker skip: agent={agent is not None}"
                      f" conn={agent._connected if agent else '?'}"
                      f" enabled={agent.enabled if agent else '?'}", flush=True)
            continue
        # 積極性プロファイル（S5）: 介入クールダウンと沈黙要約の閾値
        _cooldown = state.proactivity.get("cooldown", _INTERVENTION_COOLDOWN)
        _silence_summarize = state.proactivity.get("silence_summarize")
        _enabled = _intervention_enabled(state)
        with state.state_lock:
            meeting_epoch = state.meeting_epoch
            _skip = {AGENT_SPEAKER, "パートナー"}
            talk_rs = [r for r in state.records
                       if "speaker" in r and r.get("text")
                       and r.get("speaker") not in _skip]
        n = len(talk_rs)
        if n > state.agent_cursor:
            # raw スライス（話者未確定も含む）: 割り込み判定はこちらを使う。
            # 「遮る場面ほど帰属が壊れ、止まらない」問題への対処（C1）。
            _raw_new = talk_rs[state.agent_cursor:]
            new_records = intervention_records(_raw_new)
            # 発話供給（副作用）の直前で epoch 確認（H2）。リセットを跨いだら、
            # 古い会議の発話を新しい agent に流さないようこのtickを破棄する。
            with state.state_lock:
                if state.meeting_epoch != meeting_epoch:
                    continue
            if new_records:
                _last_utt_time[0] = time.monotonic()
                _af_new_utt = True  # フロア成立前なら af 生成先行を cancel する材料
            if _enabled:
                # 音声呼びかけの検出は _run_triage_worker（LLM分類）が担う。
                # ここでは発話をエージェントに供給するだけ。
                for r in new_records:
                    agent.feed(intervention_speaker_name(state, r),
                               r.get("text", ""))
            # cursor 書き戻しは epoch 再確認と同一 lock で atomic に行う（H2）。
            with state.state_lock:
                if state.meeting_epoch != meeting_epoch:
                    continue
                state.agent_cursor = n
            # --- 自動割り込み ---
            # 発話の存在は話者未確定でも確実なので raw スライスで判定する。
            # ファシリテーター/パートナーとも相槌 (_is_backchannel) は割り込みに使わない
            # (T7: 長めの相槌でファシリテーター発話がキャンセルされるのを防ぐ)。
            _raw_texts = [str(r.get("text", "")) for r in _raw_new]
            _human_spoke = any(len(t.strip()) > _INTERRUPT_MIN_CHARS
                               and not _is_backchannel(t)
                               for t in _raw_texts)
            if _human_spoke and agent.ai_speaking:
                agent.interrupt()
            if partner is not None and (partner.ai_speaking or partner._responding):
                _real_utterances = [t.strip() for t in _raw_texts
                                    if not _is_backchannel(t)]
                if _real_utterances:
                    partner.interrupt()
                    for i, utt in enumerate(_real_utterances):
                        is_last = (i == len(_real_utterances) - 1)
                        partner.inject_context(
                            "人間", utt,
                            request_response=is_last)
        if not _enabled:
            if (_pending.manual_call is not None
                    or not state.manual_call_requests.empty()):
                _set_manual_status(state, "cancelled", detail="介入オフのため破棄")
            _pending.clear_all()
            for q in (state.drift_requests, state.invite_requests,
                      state.factcheck_requests, state.manual_call_requests,
                      state.summarize_requests):
                while True:
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break
            if _diag_tick % 20 == 0:
                print("# [diag] agent: intervention disabled", flush=True)
            continue
        # --- ファシリテーター優先 ---
        if (partner is not None
                and (partner.ai_speaking or partner._responding)
                and agent is not None
                and agent.ai_speaking):
            partner.interrupt()
        # --- drift_checker/participation_checkerからの要求を回収（R2/S4） ---
        # busyでも取りこぼさないよう、キューは毎ループ必ずdrainして最新を保持する。
        _pending.drain(state, now=time.monotonic())

        # --- af 生成先行・再生ゲート (フェーズ6, --af 有効時のみ・af 限定) ---
        # AF が無効 (既定) ならこのブロックは丸ごとスキップされ、ルールベースの
        # 採否・trigger 経路は一切変わらない (モード方針)。summarize には関与しない。
        # WHEN は Controller に委譲 (_af_gate_status)。ゲートは再生タイミングだけ担い、
        # deliver/release/cancel の時点でだけ pending.af を消費する (hold 中は保持し続け、
        # Controller の判定が pause 成立で "deliver" に変わるのを待つ)。
        if getattr(state, "af_runtime", None) is not None and _enabled:
            try:
                _af_now = time.monotonic()
                with state.topics_lock:  # topics 読み出しは lock 取得で統一 (T9-5)
                    _af_topics = list(state.topics) if state.topics else None
                _af_partner_busy = bool(
                    partner is not None and (partner.ai_speaking or partner._responding))
                _af_status, _af_payload = _af_gate_status(
                    _controller, _pending, agent,
                    now=_af_now,
                    silence_elapsed=_effective_silence(state, _af_now, _last_utt_time),
                    recent_interventions=list(_recent_interventions),
                    cooldown=_cooldown,
                    last_intervention_at=_last_intervention_at,
                    epoch=state.agent_cursor,
                    partner_busy=_af_partner_busy,
                    in_echo_window=bool(getattr(agent, "in_echo_window", False)),
                )
                _af_action = _af_gate.tick(
                    agent=agent,
                    af=_af_payload,
                    status=_af_status,
                    silence=_effective_silence(state, _af_now, _last_utt_time),
                    new_utterance=_af_new_utt,
                    agent_busy=bool(agent._responding or agent.ai_speaking),
                    now=_af_now,
                    topics=_af_topics,
                )
                if _af_action == "trigger":  # 生成先行(hold)開始 — pending.af は保持
                    _held = _af_payload or {}
                    _af_held_text = str(_held.get("af_text") or "")
                    _af_held_kind = str(_held.get("kind") or "af_l1")
                    _af_held_target = _held.get("target_speaker") or None
                    _log_intervention_event(state, _af_held_kind, "af 生成先行(hold)")
                elif _af_action == "deliver":  # 取り込み遅延で pause 通過後に来た → 即時配信
                    _held = _af_payload or {}
                    _af_kd = str(_held.get("kind") or "af_l1")
                    _af_tx = str(_held.get("af_text") or "")
                    _afrt = state.af_runtime
                    if _afrt is not None and _af_tx:
                        with contextlib.suppress(Exception):
                            _afrt.note_intervention(_af_kd, _af_tx)
                    _pending.clear_af()  # 消費
                    _note_intervention(_af_now, _af_kd, "af 即時配信",
                                       invite_target=_held.get("target_speaker"))
                    _log_intervention_event(
                        state, _af_kd, "af 即時配信",
                        timing=_intervention_timing_metadata(
                            kind=_af_kd, now=_af_now,
                            silence_elapsed=_effective_silence(state, _af_now, _last_utt_time),
                            pause_required=policy_for(_af_kd).pause,
                            policy="af_intervention"))
                elif _af_action == "release":  # フロア成立 → 生成先行分を一斉再生
                    _afrt = state.af_runtime
                    if _afrt is not None and _af_held_text:
                        with contextlib.suppress(Exception):
                            _afrt.note_intervention(_af_held_kind, _af_held_text)
                    _pending.clear_af()  # 消費
                    _note_intervention(_af_now, _af_held_kind, "af release",
                                       invite_target=_af_held_target)
                    _log_intervention_event(
                        state, _af_held_kind, "af release",
                        timing=_intervention_timing_metadata(
                            kind=_af_held_kind, now=_af_now,
                            silence_elapsed=_effective_silence(state, _af_now, _last_utt_time),
                            pause_required=policy_for(_af_held_kind).pause,
                            policy="af_intervention",
                            hold_to_release_ms=_af_gate.last_release_ms))
                    _af_held_text = ""
                elif _af_action == "cancel":  # フロアが返らず/会話が動いた → 破棄
                    _pending.clear_af()  # 消費 (リトライにしない, B4)
                    _af_held_text = ""
                    print("# [af] 生成先行を破棄 (フロア未成立/新規発話)", flush=True)
            except Exception as _afe:  # pragma: no cover - 防御的
                print(f"# [af] early-gen error: {_afe}", flush=True)

        # --- 最優先のバージイン（ガードバイパス）:
        # ①事実補正 ②脱線介入 ③中断介入のリトライ ---
        # agentがfree(応答中でなく発話中でもない)になった瞬間に、エコーウィンドウ・
        # パートナー発話・沈黙閾値を無視してトリガーする。会話が活発でも取りこぼさない。
        # trigger()の呼び出しはこの _run_agent_worker に一元化されている（R2）。
        if not agent._responding and not agent.ai_speaking:
            _now = time.monotonic()
            # F3: アクティブな partial 中はフロア占有として沈黙 0 に倒す。
            _silence_elapsed = _effective_silence(state, _now, _last_utt_time)
            _partner_busy = bool(partner is not None
                                 and (partner.ai_speaking or partner._responding))
            _bargein_topics = None
            if agent.mode != "conversation":
                with state.topics_lock:
                    _bargein_topics = list(state.topics) if state.topics else None
            try:
                decision, _ctrl_barge, _barge_cands, _ctrl_latency_ms = (
                    _controller_barge_in_decision(
                        _controller,
                        pending=_pending,
                        agent=agent,
                        state=state,
                        now=_now,
                        last_fact_at=_last_fact_at,
                        last_intervention_at=_last_intervention_at,
                        silence_elapsed=_silence_elapsed,
                        partner_busy=_partner_busy,
                        in_echo_window=bool(agent.in_echo_window),
                        cooldown=_cooldown,
                        recent_interventions=list(_recent_interventions),
                        silence_summarize=_silence_summarize,
                        last_invited=_last_invited,
                        epoch=state.agent_cursor,
                    )
                )
            except Exception as exc:
                # Controllerは決定的（LLMなし）なので、ここに来るのはバグのみ。
                # 別実装へfallbackすると障害時に挙動が静かに変わるため（M1）、
                # このtickは何もせず、次のtickで再評価する。
                print(f"# [diag] controller barge-in error（このtickは見送り）: {exc}",
                      flush=True)
                continue
            # 古い判断の破棄（§8.5）: 裁定後に新しい発話で世代がずれたら採らない。
            if (decision.reason not in ("none", "hold")
                    and _ctrl_barge is not None
                    and _ctrl_barge.valid_for_epoch != state.agent_cursor):
                print("# [trigger] skip: stale decision (epoch changed)", flush=True)
                continue
            if decision.reason != "none":
                # 候補がある限り _ctrl_barge は必ず存在する（reason!="none" ⇒ 候補あり）。
                if decision.reason == "fact" and decision.fact is not None:
                    _legacy = _legacy_decision_brief(
                        "fact", str(decision.fact.get("correction") or "").strip())
                elif decision.reason == "manual" and decision.manual is not None:
                    _legacy = _legacy_decision_brief(
                        "manual_call",
                        str(decision.manual.get("request") or "").strip()
                        or "直近の議論整理")
                elif decision.reason == "drift" and decision.drift_reason is not None:
                    _legacy = _legacy_decision_brief("drift", decision.drift_reason)
                elif decision.reason == "retry":
                    _legacy = _legacy_decision_brief(
                        "retry", "中断された介入を再送")
                else:
                    _legacy = _legacy_decision_brief(decision.reason)
                if _ctrl_barge is not None:
                    _review.record(
                        state,
                        candidates=_barge_cands,
                        decision=_ctrl_barge,
                        silence_elapsed=_silence_elapsed,
                        epoch=state.agent_cursor,
                        legacy=_legacy,
                        latency_ms=_ctrl_latency_ms,
                    )
            # hold は「barge-in レーンに今すぐ話せる候補が無い」だけを意味する。
            # continue で通常レーン（count/silence/invite）ごと止めると、確認待ち
            # の drift 候補1件で他の介入が無期限に飢餓する（C1）。none と同様に
            # フォールスルーし、通常レーンは自身の pause/cooldown で判断させる。
            if decision.reason == "fact" and decision.fact is not None:
                correction = str(decision.fact.get("correction") or "").strip()
                timing = _intervention_timing_metadata(
                    kind="fact",
                    now=_now,
                    silence_elapsed=_silence_elapsed,
                    pause_required=policy_for("fact").pause,
                    queued_at=float(decision.fact.get("_queued_at", _now)),
                    queued_wall_at=str(decision.fact.get("_queued_wall_at") or ""),
                    policy="fact_freshness_pause",
                )
                print(f"# [trigger] fact: {correction}", flush=True)
                _log_intervention_event(state, "fact", correction, timing=timing)
                agent.trigger(topics=_bargein_topics,
                              fact_correction=decision.fact,
                              retry_intervention=False)
                _pending.facts.popleft()
                _last_fact_at = time.monotonic()
                _last_intervention_at = _last_fact_at
                _note_intervention(_last_fact_at, "fact", correction)
                continue
            if decision.reason == "manual" and decision.manual is not None:
                manual = decision.manual
                request = str(manual.get("request") or "").strip()
                detail = request or "直近の議論整理"
                _queued_payload = _pending.manual_call or {}
                _queued_at = float(_queued_payload.get("created_at", _now))
                timing = _intervention_timing_metadata(
                    kind="manual",
                    now=_now,
                    silence_elapsed=_silence_elapsed,
                    pause_required=policy_for("manual").pause,
                    queued_at=_queued_at,
                    queued_wall_at=str(
                        _queued_payload.get("created_wall_at") or ""),
                    policy="manual_call_pause",
                )
                print(f"# [trigger] manual_call: {detail}", flush=True)
                _log_intervention_event(
                    state, "manual_call", detail,
                    timing={**timing, "source": manual.get("source", "ui"),
                            "request": request, "queued_at": _queued_at,
                            "outcome": "selected"})
                _set_manual_status(state, "dispatched", detail=detail,
                                   wait_sec=_now - _queued_at)
                agent.trigger(topics=_bargein_topics, manual_request=manual)
                _pending.clear_manual()
                _last_intervention_at = time.monotonic()
                _note_intervention(_last_intervention_at, "manual", detail)
                continue
            if decision.reason == "drift" and decision.drift_reason is not None:
                timing = _intervention_timing_metadata(
                    kind="drift",
                    now=_now,
                    silence_elapsed=_silence_elapsed,
                    pause_required=policy_for("drift").pause,
                    queued_at=_pending.last_drift_request_at or None,
                    queued_wall_at=_pending.last_drift_request_wall_at,
                    policy="drift_confirmation_pause",
                )
                print(f"# [trigger] drift: 脱線介入「{decision.drift_reason}」",
                      flush=True)
                _log_intervention_event(
                    state, "drift", decision.drift_reason, timing=timing)
                agent.trigger(topics=_bargein_topics,
                              drift_reason=decision.drift_reason,
                              recent_agent_texts=_recent_agent_texts(state))
                _pending.clear_drift()
                _last_intervention_at = time.monotonic()
                _note_intervention(_last_intervention_at, "drift", decision.drift_reason)
                continue
            if decision.reason == "retry":
                pending_intervention = agent._pending_intervention or {}
                timing = _intervention_timing_metadata(
                    kind="retry",
                    now=_now,
                    silence_elapsed=_silence_elapsed,
                    pause_required=policy_for("retry").pause,
                    queued_at=float(pending_intervention.get("created_at", _now)),
                    policy="retry_extra_pause",
                )
                print("# [trigger] retry: 中断された介入を再送（ガードバイパス）",
                      flush=True)
                _log_intervention_event(
                    state, "retry", "中断された介入を再送", timing=timing)
                agent.trigger(topics=_bargein_topics, is_retry=True)
                _last_intervention_at = time.monotonic()
                _note_intervention(_last_intervention_at, "retry", "中断された介入を再送")
                continue
        # エコーウィンドウ中はtriggerしない
        if agent is not None and agent.in_echo_window:
            _was_in_echo[0] = True
            _review_now = time.monotonic()
            _review.evaluate(
                state,
                pending=_pending,
                agent=agent,
                now=_review_now,
                silence_elapsed=_review_now - _last_utt_time[0],
                epoch=state.agent_cursor,
                recent_interventions=list(_recent_interventions),
                legacy=_legacy_decision_brief("hold", "echo_window"),
                silence_summarize=_silence_summarize,
                partner_present=partner is not None,
                last_invited=_last_invited,
            )
            continue
        # Partnerが発話中はtriggerしない
        if partner is not None and (partner.ai_speaking or partner._responding):
            _was_in_echo[0] = True
            _review_now = time.monotonic()
            _review.evaluate(
                state,
                pending=_pending,
                agent=agent,
                now=_review_now,
                silence_elapsed=_review_now - _last_utt_time[0],
                epoch=state.agent_cursor,
                recent_interventions=list(_recent_interventions),
                legacy=_legacy_decision_brief("hold", "partner_busy"),
                silence_summarize=_silence_summarize,
                partner_present=partner is not None,
                last_invited=_last_invited,
            )
            continue
        # --- フロア返却 ---
        if _was_in_echo[0]:
            _was_in_echo[0] = False
            _last_utt_time[0] = time.monotonic()
        # モード別トリガー判定
        if _diag_tick % 20 == 0:
            _elapsed = time.monotonic() - _last_utt_time[0]
            print(f"# [diag] agent: mode={agent.mode} pending={agent.pending_count}"
                  f" trigger_n={agent.trigger_n} responding={agent._responding}"
                  f" silence={_elapsed:.1f}s echo={agent.in_echo_window}"
                  f" partner_talk={partner.ai_speaking if partner else '?'}", flush=True)
        # --- 論点一覧を取得（facilitatorモードのみ） ---
        _topics = None
        if agent.mode != "conversation":
            with state.topics_lock:
                _topics = list(state.topics) if state.topics else None
        # --- モード別トリガー判定 ---
        # （中断された介入のリトライは上のガードバイパス節に集約済み）
        # F3: アクティブな partial 中はフロア占有として沈黙 0 に倒す。
        _silence_elapsed = _effective_silence(state, time.monotonic(), _last_utt_time)
        try:
            normal_decision, _ctrl_normal, _normal_cands, _ctrl_latency_ms = (
                _controller_normal_decision(
                    _controller,
                    pending=_pending,
                    agent=agent,
                    now=time.monotonic(),
                    silence_elapsed=_silence_elapsed,
                    silence_summarize=_silence_summarize,
                    partner_present=partner is not None,
                    last_intervention_at=_last_intervention_at,
                    cooldown=_cooldown,
                    last_invited=_last_invited,
                    recent_interventions=list(_recent_interventions),
                    epoch=state.agent_cursor,
                )
            )
        except Exception as exc:
            # barge-in 側と同じ方針: 別実装へfallbackせず、このtickは見送る（M1）。
            print(f"# [diag] controller normal error（このtickは見送り）: {exc}",
                  flush=True)
            continue
        # 古い判断の破棄（§8.5）。
        if (normal_decision.reason not in ("none", "skip_invite")
                and _ctrl_normal is not None
                and _ctrl_normal.valid_for_epoch != state.agent_cursor):
            print("# [trigger] skip: stale normal decision (epoch changed)", flush=True)
            continue
        _normal_detail = normal_decision.detail
        if normal_decision.reason == "skip_invite":
            _normal_detail = f"{normal_decision.invite_target}さんへの連続声かけを抑制"
        _legacy_normal = _legacy_decision_brief(normal_decision.reason, _normal_detail)
        if _ctrl_normal is not None:
            _review.record(
                state,
                candidates=_normal_cands,
                decision=_ctrl_normal,
                silence_elapsed=_silence_elapsed,
                epoch=state.agent_cursor,
                legacy=_legacy_normal,
                latency_ms=_ctrl_latency_ms,
            )
        else:
            _review.evaluate(
                state,
                pending=_pending,
                agent=agent,
                now=time.monotonic(),
                silence_elapsed=_silence_elapsed,
                epoch=state.agent_cursor,
                recent_interventions=list(_recent_interventions),
                legacy=_legacy_normal,
                silence_summarize=_silence_summarize,
                partner_present=partner is not None,
                last_invited=_last_invited,
            )
        _fire_normal(normal_decision, _silence_elapsed, _silence_summarize)


def _run_stdin_commands(state: SessionState):
    """標準入力からの話者リネーム・統合コマンドを処理."""
    while not state.stop.is_set():
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        mfix = re.match(r"^\s*fix\s+(\S+)\s*=\s*(\S+)\s*$", line)
        m = re.match(r"^\s*(\S+?)\s*=\s*(.+?)\s*$", line)
        if mfix:
            src, dst = state.key_of(mfix.group(1)), state.key_of(mfix.group(2))
            if state.tracker is not None:
                state.tracker.remap(src, dst)
            state.rekey(src, dst)
            state.add_sys(None, f"{state.disp_name(src)} を {state.disp_name(dst)} に統合（手動fix）")
            state.save()
            _print_line(f"# {state.disp_name(src)} を {state.disp_name(dst)} に統合しました（過去の発言も修正済み）")
        elif m:
            label, name = m.group(1), m.group(2)
            if state.tracker is not None:
                old = state.tracker.enroll(label, name)
                if old is None:
                    _print_line(f"# 話者{label}の音声がまだ足りません（1秒以上話してから再実行）")
                    continue
                state.rekey(old, name)
                state.add_sys(None, f"「{name}」の声を登録（次回の会議から自動表示）")
                state.save()
                _print_line(f"# {name} の声を登録しました（過去の発言も置換、次回の会議から自動表示）")
            else:
                state.set_display_name("#" + label, name)
                state.save()
                _print_line(f"# 話者{label} → {name}（過去の発言も置換済み）")
        elif line.strip():
            _print_line("# 名前登録はブラウザUIを推奨。ターミナル操作: 「1=名前」/「fix 2=1」/ Ctrl+Cで終了")


def _run_from_mic(state: SessionState, device):
    """マイクからPCMを読み取り audio_q に送信."""
    import sounddevice as sd
    agent = state.agent

    def cb(indata, frames, t, status):
        pcm = (np.clip(indata[:, 0], -1, 1) * 32767).astype("<i2").tobytes()
        state.audio_q.put(pcm)
        partner = state.partner  # 動的参照: 実行中の接続/切断に追従（F3）
        if (partner is not None and partner._connected
                and not partner.in_echo_window
                and not (agent is not None and agent.in_echo_window)):
            partner.feed_audio(pcm)
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        device=device, callback=cb, blocksize=int(SR * 0.1)):
        while not state.stop.is_set():
            time.sleep(0.1)
    state.audio_q.put(None)


def _load_wav_mono_16k(path: str) -> "np.ndarray":
    """音声ファイルをモノラル float32 の SR(16kHz) 配列で読む.

    従来は librosa を使っていたが、librosa はどの依存グループにも宣言されて
    おらず --wav が常に ModuleNotFoundError で死んでいた。主用途（本システムが
    録音した transcripts/*.wav の再入力）は PCM WAV なので標準ライブラリ wave で
    依存ゼロで読み、レート違いは線形補間で SR に合わせる。PCM 以外の形式は
    torchaudio へのフォールバックを試みるが、環境によっては動かない
    （torchaudio 2.9+ のデコードは torchcodec 必須で、未導入だと ImportError）
    ため、失敗時は PCM WAV への変換手順を示して明確に終了する
    （2026-07-15 レビュー F5）。
    """
    import wave
    try:
        with wave.open(path, "rb") as w:
            n_ch = w.getnchannels()
            width = w.getsampwidth()
            sr = w.getframerate()
            raw = w.readframes(w.getnframes())
        if width == 2:
            y = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
        elif width == 4:
            y = np.frombuffer(raw, dtype="<i4").astype("float32") / 2147483648.0
        else:
            raise wave.Error(f"unsupported sample width: {width}")
        if n_ch > 1:
            y = y.reshape(-1, n_ch).mean(axis=1)
    # wave.Error に加えて EOFError も捕捉する: 空ファイル・ヘッダ途中で切れた
    # ファイルでは wave モジュールが EOFError を裸で投げ、従来はトレースバック
    # ごと落ちていた（2026-07-15 レビュー F5、プローブ probe_wav.py で確認）。
    except (wave.Error, EOFError):
        try:
            import torchaudio
            t, sr = torchaudio.load(path)
        except Exception as e:
            # torchaudio 未導入 / torchcodec 欠如 / 非対応・破損ファイルは
            # ユーザーが対処可能なメッセージで終了する（内部トレースバックを
            # 見せない。--wav はCLIの入り口なので案内が最重要）。
            raise SystemExit(
                f"--wav: {path} を読み込めませんでした（{type(e).__name__}: {e}）。\n"
                "  PCM WAV に変換してください（例: ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav）"
            ) from e
        y = t.mean(dim=0).numpy().astype("float32")
    if sr != SR:
        # 線形補間による簡易リサンプル（STT入力用途には十分）
        n_out = int(round(len(y) * SR / sr))
        y = np.interp(np.linspace(0, len(y) - 1, n_out),
                      np.arange(len(y)), y).astype("float32")
    return y


def _run_from_wav(state: SessionState, args):
    """WAVファイルを擬似ライブで送信する.

    Reactive WAV: agentが発話中はWAV再生・ASR送信を一時停止し、
    介入終了後に自動再開する。
    """
    agent = state.agent
    y = _load_wav_mono_16k(args.wav)
    step = int(SR * 0.12)
    out = mic = None
    if args.play or args.join:
        import sounddevice as sd
        out = sd.OutputStream(samplerate=SR, channels=1, dtype="float32")
        out.start()
    if args.join:
        import sounddevice as sd
        mic = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=step)
        mic.start()
    i = 0
    _wav_paused = False
    while not state.stop.is_set():
        if agent is not None and (agent.ai_speaking or agent._responding):
            if not _wav_paused:
                _wav_paused = True
                print("# WAV: AI介入中 — 再生を一時停止", flush=True)
            time.sleep(0.05)
            continue
        if _wav_paused:
            _wav_paused = False
            print("# WAV: 再生を再開", flush=True)
        chunk = np.clip(y[i:i + step], -1, 1).astype("float32") if i < len(y) else \
            np.zeros(0, dtype="float32")
        if len(chunk) < step:
            chunk = np.pad(chunk, (0, step - len(chunk)))
        i += step
        if i - step >= len(y) and mic is None:
            break
        if mic is not None:
            mdata, _ = mic.read(step)
            mix = np.clip(chunk + mdata[:, 0], -1, 1)
        else:
            mix = chunk
        state.audio_q.put((mix * 32767).astype("<i2").tobytes())
        if out is not None:
            out.write(chunk.reshape(-1, 1))
            if mic is None:
                continue
        if mic is None and out is None:
            time.sleep(0.12)
    for s in (out, mic):
        if s is not None:
            s.stop()
            s.close()
    state.audio_q.put(None)


def _run_sender(state: SessionState, backend: STTBackend):
    """audio_qからPCMを読みWebSocketに送信 + PCMバッファ/ファイル書き出し.

    送信先は state.stt_ws を毎回参照する。STT接続を作り直しても追従し、
    作り直し中(古いwsが閉じている瞬間)の送信エラーは無視する（音声を捨てる）。

    録音wavには**STTへ送れたチャンクだけ**を書く。発話の ms は送信済み音声の
    バイト位置（`asr_pcm_total_bytes // 32`）そのものなので、こうすると wav の
    位置と ms が 1:1 で対応し、後から wav を ms で切って採点・アノテーション
    できる。送れなかった分まで書くと wav だけが先へずれ、そのずれは二度と
    戻らない（実測: 4分の会議で +1.5秒→+2.5秒、短い発話のオラクル精度が
    偶然以下まで落ちた）。捨てた量は `pcm_total_bytes - asr_pcm_total_bytes`
    で分かり、`finalize_wav` が知らせる。
    """
    seq = 0
    while True:
        pcm = state.audio_q.get()
        ws = state.stt_ws
        if pcm is None:
            if ws is not None:
                with contextlib.suppress(Exception):
                    ws.send(backend.make_end_message(seq))
            break
        setup_capture_only = state.waiting_to_start and ws is None
        with state.buf_lock:
            state.pcm_buf.extend(pcm)
            if not setup_capture_only:
                state.pcm_total_bytes += len(pcm)
            if len(state.pcm_buf) > state._PCM_KEEP_BYTES + SR * 2 * 10:
                trim = len(state.pcm_buf) - state._PCM_KEEP_BYTES
                del state.pcm_buf[:trim]
                state.pcm_buf_offset += trim
        if ws is not None:
            try:
                ws.send(pcm)
            except Exception:
                pass
            else:
                if state.pcm_file is not None:
                    try:
                        state.pcm_file.write(pcm)
                        state.pcm_file.flush()
                    except OSError:
                        pass
                with state.buf_lock:
                    state.asr_pcm_buf.extend(pcm)
                    state.asr_pcm_total_bytes += len(pcm)
                    if len(state.asr_pcm_buf) > state._PCM_KEEP_BYTES + SR * 2 * 10:
                        trim = len(state.asr_pcm_buf) - state._PCM_KEEP_BYTES
                        del state.asr_pcm_buf[:trim]
                        state.asr_pcm_buf_offset += trim
                if state.diarization_provider is not None:
                    with contextlib.suppress(Exception):
                        state.diarization_provider.send_audio(pcm)
                    state.drain_diarization_provider()
                seq += 1


def _cleanup(state: SessionState, tracker, wav_path: str, out_path: str, html_path: str):
    """セッション終了時のリソース解放・ファイル保存."""
    from das.asr.live import _SYS_HOOK_REF
    _SYS_HOOK_REF[0] = None
    state.stop.set()
    if state.partner is not None:
        state.partner.close()
    if state.simulator is not None:
        state.simulator.shutdown()
    if state.agent is not None:
        state.agent.close()
    state.save(live=False)
    if tracker is not None:
        _print_line(f"# レイテンシ統計: {tracker.stats()}")
    saved = state.out_path if state._serve else f"{state.out_path} / {state.html_path}"
    _print_line(f"# 議事録を保存しました: {saved}")
    # WAVファイルのヘッダを確定して正規のWAVにする（state.wav_pathを使用）
    saved_wav = state.finalize_wav()
    if saved_wav:
        _print_line(f"# 録音を保存しました: {saved_wav}")

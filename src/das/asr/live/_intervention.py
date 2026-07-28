"""介入するかどうかの判断（候補づくり・抑止・割り込み/通常の決定）.

`_workers.py` から切り出した。ここにあるのは「いま口を挟むか、挟むなら
どの種類か」を決める層だけで、実際に喋らせるのも、話者を判定するのも別。

    各チェッカー（話題・脱線・ファクト・参加・構造化…）
        └─> _PendingInterventions に積む
              └─> _build_candidates が候補に変換
                    └─> _controller_barge_in_decision / _controller_normal_decision
                          └─> 決定（_workers 側が実行）

`_InterventionReviewRecorder` は「その判断が妥当だったか」を後から見るための
記録で、判断そのものには影響しない。
"""
from __future__ import annotations

import collections
import contextlib
import datetime
import queue
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._session_state import SessionState

from ._constants import (
    _AGENT_DEBATE_SILENCE,
    _BACKCHANNEL_RE,
    _DRIFT_PENDING_TTL,
    _FACTCHECK_COOLDOWN,
    _FACTCHECK_PENDING_TTL,
    _MANUAL_CALL_TTL,
    _PARTIAL_FLOOR_MAX_AGE,
    AGENT_SPEAKER,
)
from ._facilitation import (
    FacilitationController,
    FacilitationDecision,
    FacilitationInput,
    InterventionCandidate,
    InterventionLogEntry,
    confidence_score,
    fact_expires_at,
)
from ._speaker_policy import intervention_speaker_name


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

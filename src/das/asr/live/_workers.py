"""main()から抽出されたワーカー関数群.

ログ接頭辞の規約（Phase 3 R4）:
  # [state]   ... エージェントの状態遷移（RESPONDING/SPEAKING/INTERRUPTED/IDLE等）
  # [trigger] ... ファシリテーターのトリガー理由（drift/retry/count/silence/invite/skip）
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
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ._session_state import SessionState
    from .stt import STTBackend

from ._constants import (
    _AGENDA_MIN_UTTS,
    _AGENDA_RETRY_SEC,
    _AGENDA_WINDOW,
    _AGENT_CONV_SILENCE,
    _AGENT_DEBATE_SILENCE,
    _BACKCHANNEL_RE,
    _DRIFT_CHECK_INTERVAL,
    _DRIFT_CHECK_WINDOW,
    _DRIFT_WARMUP,
    _FACTCHECK_CHECK_SEC,
    _FACTCHECK_COOLDOWN,
    _FACTCHECK_MAX_RETRIES,
    _FACTCHECK_MIN_CHARS,
    _FACTCHECK_PENDING_TTL,
    _INTERRUPT_MIN_CHARS,
    _INTERVENTION_COOLDOWN,
    _INTERVENTION_PAUSE_COUNT,
    _INTERVENTION_PAUSE_DRIFT,
    _INTERVENTION_PAUSE_FACT,
    _INTERVENTION_PAUSE_MANUAL,
    _INTERVENTION_PAUSE_RETRY,
    _INVITE_CHECK_SEC,
    _INVITE_QUIET_RATIO,
    _INVITE_SILENCE,
    _INVITE_WARMUP,
    _MANUAL_CALL_TTL,
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
)
from ._ui import _print_line

_FACT_PHONE_NUMBER_RE = re.compile(r"\b0\d{1,3}-\d{2,4}-\d{3,4}\b")
_FACT_QUESTION_RE = re.compile(
    r"(ですか|でしょうか|ますか|かな|かね|なの|だっけ|でしたっけ|"
    r"何|どれ|どこ|誰|いつ|いくつ|いくら|\?|？)"
)
_FACT_UNCERTAIN_RE = re.compile(
    r"(たぶん|多分|おそらく|多分だけど|うろ覚え|曖昧|わからない|分からない|"
    r"知らない|覚えてない|忘れた|気がする|かもしれない|かも|らしい)"
)
_FACT_PREFERENCE_RE = re.compile(
    r"(好き|嫌い|好み|苦手|うれしい|嬉しい|楽しい|面白い|つまらない|"
    r"良い|いい|悪い|きれい|綺麗|かわいい|かっこいい|おいしい|美味しい|"
    # 評価・主観（「XはYです」を通す前に、評価文を確実に落とすため強化）
    r"良さそう|よさそう|良さげ|よさげ|悪そう|わるそう|うまそう|まずそう|"
    r"微妙|最悪|素晴らしい|すばらしい|失礼|最低ライン)"
)
_FACT_META_TALK_RE = re.compile(
    r"(話しましょう|確認しましょう|決めましょう|考えましょう|進めましょう|"
    r"について話|の話です|という話|話題|論点|議題|雑談)"
)
_FACT_CREATIVE_EXPRESSION_RE = re.compile(
    r"(奴|襲い|跳弾|跳ね返った弾丸|二丁拳銃|拳銃|銃|弾丸|弾切れ|"
    r"極小の銃|ビビ弾|凶悪だぜ|踊れ|見抜いていた|お釣りだ|"
    r"受け取っとけ|間合い)"
)
_FACT_STRONG_ANCHOR_RE = re.compile(
    r"([=＋+\-*/÷]|cm|m|km|kg|g|メートル|キロ|円|ドル|回|個|勝|日付|"
    r"計算式|数式|の式|式は|値|定義|単位|制度|上限|下限|分子|分母|2乗|二乗|"
    r"首都|所属|出身|作者|CEO|国|地域|地方|都道府県|東北|関東|中部|"
    r"山|湖|川|島|時代|順序|ランキング|順位|トップ|番目)"
)
# 含有・成分関係（食品成分など安定した一般事実。question/uncertain 等の
# negative filter を通過したものだけがここに来る）。
_FACT_CONTAINMENT_RE = re.compile(r"(含まれ|含む|含有|成分|主成分)")
# 「XはYです / XはYではありません / XはYに属します / XはYで発生しました」等の
# 断定文。負のフィルタ（好み・質問・曖昧・メタ・創作）を先に通したうえで拾う。
_FACT_ASSERTION_RE = re.compile(
    r".+は.+?(です|である|ではありません|ではない|じゃありません|"
    r"に属し|に分類され|で発生し)"
)
# 指示語主語（これ/それ/あれは…）は外部照合に向かない自己言及・曖昧断定なので、
# 断定ゲートからは除外する（強アンカー・含有ゲートには影響しない）。
_FACT_DEICTIC_SUBJECT_RE = re.compile(
    r"^[\s、。]*(これ|それ|あれ|こちら|そちら|あちら)(は|が|も)"
)


def _looks_like_fact_claim(text: str) -> bool:
    """LLMに渡す前の候補フィルタ（「明らかに判定不要なものを落とす」中心）。

    明確な誤りかどうかの最終判断は LLM（confidence==high のみ採用）に任せる。
    前段は会議を止めないため、相槌・質問・曖昧表現・好み/評価・創作表現・メタ発話・
    電話番号を確実に落とす。そのうえで、明確な事実断定らしいもの（強い事実アンカー・
    含有/成分・断定文）は広めに LLM へ通す。「強いアンカー必須」ではなく
    「明確な除外に該当しなければ、断定文なら通す」方針。
    """
    s = (text or "").strip()
    if len(s) < _FACTCHECK_MIN_CHARS:
        return False
    if _BACKCHANNEL_RE.match(s):
        return False
    uncertain_only = re.fullmatch(
        r"[\s、。,.!?！？]*(たぶん|多分|なんでしたっけ|何でしたっけ|"
        r"わからない|分からない|知らない|覚えてない|忘れた)[\s、。,.!?！？]*",
        s,
    )
    if uncertain_only:
        return False
    # --- 明確に判定不要なものを落とす（negative filters） ---
    if _FACT_PHONE_NUMBER_RE.search(s):
        return False
    if _FACT_QUESTION_RE.search(s):
        return False
    if _FACT_UNCERTAIN_RE.search(s):
        return False
    if _FACT_PREFERENCE_RE.search(s):
        return False
    if _FACT_CREATIVE_EXPRESSION_RE.search(s):
        return False
    if _FACT_META_TALK_RE.search(s):
        return False
    # --- 事実断定らしいものを通す（positive gates。最終判断はLLM） ---
    if _FACT_STRONG_ANCHOR_RE.search(s):
        return True
    if _FACT_CONTAINMENT_RE.search(s):
        return True
    return bool(_FACT_ASSERTION_RE.search(s)
                and not _FACT_DEICTIC_SUBJECT_RE.match(s))


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

    def drop_stale_manual(self, *, now: float) -> None:
        """会話タイミングを外した古い手動呼び出しを破棄する（TTL）."""
        if self.manual_call is None:
            return
        age = now - float(self.manual_call.get("created_at", now))
        if age > _MANUAL_CALL_TTL:
            print(f"# [trigger] skip: 古い手動呼び出しを破棄（{age:.0f}秒経過）",
                  flush=True)
            self.manual_call = None

    def clear_manual(self) -> None:
        self.manual_call = None

    def clear_all(self) -> None:
        """介入オフ/モードオフ時に、worker内で保持した候補も破棄する."""
        self.drift_reason = None
        self.drift_count = 0
        self.last_drift_request_at = 0.0
        self.last_drift_request_wall_at = None
        self.facts.clear()
        self.invite = None
        self.manual_call = None

    def drop_stale_facts(self, *, now: float) -> None:
        """会話タイミングを外した古い事実補正を破棄する."""
        while self.facts:
            age = now - float(self.facts[0].get("_queued_at", now))
            if age <= _FACTCHECK_PENDING_TTL:
                return
            stale = self.facts.popleft()
            print(f"# [trigger] skip: 古い事実補正を破棄 {stale.get('correction', '')}",
                  flush=True)

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


def _floor_available_for_intervention(
    *,
    silence_elapsed: float,
    partner_busy: bool,
    in_echo_window: bool,
    pause_required: float,
) -> bool:
    """参加者の会話を遮らず、介入が自然に入れる短い間があるか."""
    return (
        silence_elapsed >= pause_required
        and not partner_busy
        and not in_echo_window
    )


def _intervention_timing_metadata(
    *,
    kind: str,
    now: float,
    silence_elapsed: float,
    pause_required: float,
    queued_at: float | None = None,
    queued_wall_at: str | None = None,
    policy: str = "floor_pause",
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
    return timing


def _select_barge_in_decision(
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
    diag_tick: int,
) -> _BargeInDecision:
    """ガードを越えて差し込む介入を、優先順位順に1つだけ選ぶ."""
    pending.drop_stale_facts(now=now)
    while pending.facts:
        pending_fact = pending.facts[0]
        correction = str(pending_fact.get("correction") or "").strip()
        if not correction:
            pending.facts.popleft()
            continue
        fact_floor_available = _floor_available_for_intervention(
            silence_elapsed=silence_elapsed,
            partner_busy=partner_busy,
            in_echo_window=in_echo_window,
            pause_required=_INTERVENTION_PAUSE_FACT,
        )
        if not fact_floor_available:
            if diag_tick % 4 == 0:
                print("# [trigger] hold: 発話の切れ目待ちの事実補正", flush=True)
            return _BargeInDecision("hold")
        if now - last_fact_at < _FACTCHECK_COOLDOWN:
            if diag_tick % 4 == 0:
                print("# [trigger] hold: クールダウン中の事実補正", flush=True)
            return _BargeInDecision("hold")
        return _BargeInDecision("fact", fact=pending_fact)

    if pending.drift_reason is not None:
        required_drift_count = int(state.proactivity.get("drift_confirmations", 1))
        if pending.drift_count < required_drift_count:
            if diag_tick % 20 == 0:
                print(
                    "# [trigger] hold: 脱線判定の確認待ち "
                    f"{pending.drift_count}/{required_drift_count}",
                    flush=True,
                )
            return _BargeInDecision("hold")
        drift_floor_available = _floor_available_for_intervention(
            silence_elapsed=silence_elapsed,
            partner_busy=partner_busy,
            in_echo_window=in_echo_window,
            pause_required=_INTERVENTION_PAUSE_DRIFT,
        )
        if not drift_floor_available:
            if diag_tick % 4 == 0:
                print("# [trigger] hold: 発話の切れ目待ちの脱線介入", flush=True)
            return _BargeInDecision("hold")
        if now - last_intervention_at < cooldown:
            print("# [trigger] skip: クールダウン中の脱線介入", flush=True)
            pending.clear_drift()
        else:
            return _BargeInDecision("drift", drift_reason=pending.drift_reason)

    if agent._pending_intervention is not None:
        retry_floor_available = _floor_available_for_intervention(
            silence_elapsed=silence_elapsed,
            partner_busy=partner_busy,
            in_echo_window=in_echo_window,
            pause_required=_INTERVENTION_PAUSE_RETRY,
        )
        if not retry_floor_available:
            if diag_tick % 4 == 0:
                print("# [trigger] hold: 発話の切れ目待ちの再送", flush=True)
            return _BargeInDecision("hold")
        return _BargeInDecision("retry")
    return _BargeInDecision("none")


def _select_normal_trigger_decision(
    *,
    pending: _PendingInterventions,
    agent,
    silence_elapsed: float,
    silence_summarize: float | None,
    partner_present: bool,
    now: float,
    last_intervention_at: float,
    cooldown: float,
    last_invited: str | None,
) -> _NormalTriggerDecision:
    """通常の間で発火する介入を、優先順位順に1つだけ選ぶ（Controller例外時のfallback）.

    注: stall（介入不要後の一押し）は Phase3 で廃止したため、このfallbackでも
    扱わない。通常の採否は Controller（_controller_normal_decision）が担う。
    """
    if agent.mode == "conversation":
        if agent.pending_count > 0 and silence_elapsed > _AGENT_CONV_SILENCE:
            return _NormalTriggerDecision(
                "conversation", f"沈黙{silence_elapsed:.1f}秒")
        return _NormalTriggerDecision("none")

    silence_thresh = (_AGENT_DEBATE_SILENCE if partner_present
                      else silence_summarize)
    if agent.pending_count >= agent.trigger_n:
        if silence_elapsed < _INTERVENTION_PAUSE_COUNT:
            return _NormalTriggerDecision("none")
        return _NormalTriggerDecision(
            "count", f"{agent.pending_count}>={agent.trigger_n}発話")
    if (silence_thresh is not None
            and agent.pending_count > 0
            and silence_elapsed > silence_thresh):
        return _NormalTriggerDecision(
            "silence", f"{silence_elapsed:.1f}>{silence_thresh:.1f}秒")
    if (pending.invite is not None
            and silence_elapsed > _INVITE_SILENCE
            and now - last_intervention_at > cooldown):
        if pending.invite == last_invited:
            return _NormalTriggerDecision("skip_invite", invite_target=pending.invite)
        return _NormalTriggerDecision(
            "invite", f"{pending.invite}さんに声かけ", invite_target=pending.invite)
    return _NormalTriggerDecision("none")


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
        cands.append(InterventionCandidate(
            id="drift",
            kind="drift",
            brief=str(pending.drift_reason),
            created_at=float(pending.last_drift_request_at or now),
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

    if (mode != "conversation"
            and getattr(agent, "pending_count", 0) >= getattr(agent, "trigger_n", 0)
            and getattr(agent, "trigger_n", 0) > 0):
        cands.append(InterventionCandidate(
            id="count",
            kind="count",
            brief=f"{agent.pending_count}発話が蓄積",
            created_at=now,
            interrupt_policy="wait_for_pause",
        ))

    if mode == "conversation" and getattr(agent, "pending_count", 0) > 0:
        cands.append(InterventionCandidate(
            id="conversation",
            kind="conversation",
            brief=f"{agent.pending_count}発話が蓄積",
            created_at=now,
            interrupt_policy="wait_for_pause",
        ))

    silence_thresh = (_AGENT_DEBATE_SILENCE if partner_present
                      else silence_summarize)
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
    return {
        "id": c.id,
        "kind": c.kind,
        "brief": c.brief[:60],
        "confidence": round(c.confidence, 3),
        "target_speaker": c.target_speaker,
    }


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
# Phase3: stall は廃止（Speaker から「介入不要」判断を外したため）。
_NORMAL_KINDS = ("count", "silence", "invite", "conversation")


def _suppressed_for(
    decision: FacilitationDecision,
    *,
    candidate_id: str,
    reason_part: str,
) -> bool:
    """Controllerの抑制理由に特定候補/理由が含まれるかを調べる."""
    return any(
        s.get("candidate_id") == candidate_id
        and reason_part in str(s.get("reason", ""))
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
    pending.drop_stale_manual(now=now)  # 古い手動呼び出しは破棄（TTL）
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
    if _suppressed_for(
        decision,
        candidate_id="drift",
        reason_part="直前の介入から間隔不足",
    ):
        pending.clear_drift()
    if decision.candidate_id is None:
        # 候補はあるが今は採らない → 保持して次の機会を待つ（hold）。
        return _BargeInDecision("hold"), decision, cands, latency_ms
    chosen = next(c for c in cands if c.id == decision.candidate_id)
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
    if decision.candidate_id is None:
        # 連続声かけ抑制（同じ人を続けて誘わない）→ skip_invite で invite を消費。
        for s in decision.suppressed:
            if (s["candidate_id"].startswith("invite-")
                    and "直前と同じ" in s["reason"]):
                return (_NormalTriggerDecision(
                    "skip_invite", invite_target=pending.invite), decision, cands, latency_ms)
        return _NormalTriggerDecision("none"), decision, cands, latency_ms
    chosen = next(c for c in cands if c.id == decision.candidate_id)
    if chosen.kind == "conversation":
        return (_NormalTriggerDecision("conversation", f"沈黙{silence_elapsed:.1f}秒"),
                decision, cands, latency_ms)
    if chosen.kind == "count":
        return (_NormalTriggerDecision(
            "count", f"{agent.pending_count}>={agent.trigger_n}発話"), decision, cands, latency_ms)
    if chosen.kind == "silence":
        thresh = float(chosen.payload.get("pause_required", 0.0))
        return (_NormalTriggerDecision(
            "silence", f"{silence_elapsed:.1f}>{thresh:.1f}秒"), decision, cands, latency_ms)
    if chosen.kind == "invite":
        return (_NormalTriggerDecision(
            "invite", f"{chosen.target_speaker}さんに声かけ",
            invite_target=chosen.target_speaker), decision, cands, latency_ms)
    return _NormalTriggerDecision("none"), decision, cands, latency_ms


def _run_agenda_detector(state: SessionState, oai_key: str, oai_model: str):
    """会議冒頭の発話から議題を1回推定してシードする（S3, --topic未指定時）.

    既に論点があれば（明示シード or 抽出済み）何もしない。十分な発話が
    たまったらLLMで議題を推定し、成功したら seed_topic して終了する。
    判断できなければ一定間隔で再試行し、論点が現れたら（topic_worker等が
    先に論点を作ったら）役目を終えて停止する。
    """
    from das.asr.live._bootstrap import detect_agenda as _detect_agenda

    _last_attempt = 0.0
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
            talk_rs = intervention_records([
                r for r in state.records
                if "speaker" in r and r.get("text")
                and r.get("speaker") != AGENT_SPEAKER
            ])
        if len(talk_rs) < _AGENDA_MIN_UTTS:
            continue
        if time.monotonic() - _last_attempt < _AGENDA_RETRY_SEC:
            continue
        _last_attempt = time.monotonic()
        utts = [{"speaker": intervention_speaker_name(state, r), "text": r["text"]}
                for r in talk_rs[:_AGENDA_WINDOW]]
        agenda = _detect_agenda(utts, oai_key, oai_model)
        if agenda:
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
        state.topic_cursor = n


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
        state.drift_cursor = n
        print(f"# [drift] チェック実行: {len(utts)}発話, "
              f"cursor={n}, topics={len(topics)}件", flush=True)
        # 脱線判定
        result = _check_drift(utts, topics, oai_key, oai_model)
        if result.get("drift"):
            reason = result.get("reason", "")
            _print_line(f"# 🔀 脱線検出: {reason}")
            # R2: trigger()は呼ばず、要求をキューに積む。agent_workerが裁定する。
            state.drift_requests.put(reason)
            print("# [drift] → 介入要求をキューに投入", flush=True)


def _run_fact_checker(state: SessionState, oai_key: str, oai_model: str):
    """明確な事実誤りだけを短く補正する要求を積む.

    脱線や発話量とは別ルートにする。ローカルでは会議を妨げやすい
    低価値候補を保守的に落とし、明確な誤りかどうかはLLMに任せる。
    採用するのは high confidence の訂正だけ。
    """
    from das.asr.live._bootstrap import check_fact_correction as _check_fact

    _last_check = 0.0
    _recent_corrections: list[tuple[float, str]] = []
    _retry_counts: dict[int, int] = {}
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
            if _looks_like_fact_claim(str(r.get("text") or "")):
                candidate = r
                break
            next_idx += 1
        if candidate is None:
            state.fact_cursor = n
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
        state.fact_cursor = next_idx + 1
        _retry_counts.pop(next_idx, None)
        if result.get("should_correct"):
            correction = str(result.get("correction") or "").strip()
            if correction:
                norm = re.sub(r"[\s、。,.，．!！?？]+", "", correction).lower()
                _recent_corrections = [
                    (t, c) for t, c in _recent_corrections if now - t < 90.0
                ]
                if any(c == norm for _, c in _recent_corrections):
                    print("# [fact] skip: 重複する補正", flush=True)
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
    # 受信スレッドはイベントを積むだけ。副作用は専用ワーカーで処理（受信ブロック回避）。
    agent.on_ai_utterance = lambda text: state.fac_events.put(("utterance", text))
    agent.on_speech_start = lambda: state.fac_events.put(("speech_start", None))
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
    # 採否の経緯（採択/抑制/latency）を intervention_review.jsonl へ記録する。
    _review = _InterventionReviewRecorder()
    _recent_interventions: collections.deque[InterventionLogEntry] = (
        collections.deque(maxlen=10))

    def _note_intervention(at: float, kind: str, detail: str = "") -> None:
        """実際に発火した介入を cooldown 用の直近履歴に記録する."""
        _recent_interventions.append(
            InterventionLogEntry(at=at, kind=kind, brief=detail))

    while not state.stop.is_set():
        time.sleep(0.25)
        _diag_tick += 1
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
            _skip = {AGENT_SPEAKER, "パートナー"}
            talk_rs = [r for r in state.records
                       if "speaker" in r and r.get("text")
                       and r.get("speaker") not in _skip]
        n = len(talk_rs)
        if n > state.agent_cursor:
            new_records = intervention_records(talk_rs[state.agent_cursor:])
            new_texts = [r.get("text", "") for r in new_records]
            if new_records:
                _last_utt_time[0] = time.monotonic()
            if _enabled:
                for r in new_records:
                    agent.feed(intervention_speaker_name(state, r), r.get("text", ""))
            state.agent_cursor = n
            # --- 自動割り込み ---
            _human_spoke = any(len(t.strip()) > _INTERRUPT_MIN_CHARS
                               for t in new_texts)
            if _human_spoke and agent.ai_speaking:
                agent.interrupt()
            if partner is not None and (partner.ai_speaking or partner._responding):
                _real_utterances = [t.strip() for t in new_texts
                                    if t.strip()
                                    and not _BACKCHANNEL_RE.match(t.strip())]
                if _real_utterances:
                    partner.interrupt()
                    for i, utt in enumerate(_real_utterances):
                        is_last = (i == len(_real_utterances) - 1)
                        partner.inject_context(
                            "人間", utt,
                            request_response=is_last)
        if not _enabled:
            _pending.clear_all()
            for q in (state.drift_requests, state.invite_requests,
                      state.factcheck_requests, state.manual_call_requests):
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

        # --- 最優先のバージイン（ガードバイパス）:
        # ①事実補正 ②脱線介入 ③中断介入のリトライ ---
        # agentがfree(応答中でなく発話中でもない)になった瞬間に、エコーウィンドウ・
        # パートナー発話・沈黙閾値を無視してトリガーする。会話が活発でも取りこぼさない。
        # trigger()の呼び出しはこの _run_agent_worker に一元化されている（R2）。
        if not agent._responding and not agent.ai_speaking:
            _now = time.monotonic()
            _silence_elapsed = _now - _last_utt_time[0]
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
                # 採否Controllerの想定外失敗時は、実績のある従来選択にfallback。
                print(f"# [diag] controller barge-in fallback: {exc}", flush=True)
                decision = _select_barge_in_decision(
                    pending=_pending, agent=agent, state=state, now=_now,
                    last_fact_at=_last_fact_at,
                    last_intervention_at=_last_intervention_at,
                    silence_elapsed=_silence_elapsed, partner_busy=_partner_busy,
                    in_echo_window=bool(agent.in_echo_window), cooldown=_cooldown,
                    diag_tick=_diag_tick)
                _ctrl_barge = None
                _barge_cands = []
                _ctrl_latency_ms = 0.0
            # 古い判断の破棄（§8.5）: 裁定後に新しい発話で世代がずれたら採らない。
            if (decision.reason not in ("none", "hold")
                    and _ctrl_barge is not None
                    and _ctrl_barge.valid_for_epoch != state.agent_cursor):
                print("# [trigger] skip: stale decision (epoch changed)", flush=True)
                continue
            if decision.reason != "none":
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
                else:
                    _review.evaluate(
                        state,
                        pending=_pending,
                        agent=agent,
                        now=_now,
                        silence_elapsed=_silence_elapsed,
                        epoch=state.agent_cursor,
                        recent_interventions=list(_recent_interventions),
                        legacy=_legacy,
                        silence_summarize=_silence_summarize,
                        partner_present=partner is not None,
                        last_invited=_last_invited,
                    )
            if decision.reason == "hold":
                continue
            if decision.reason == "fact" and decision.fact is not None:
                correction = str(decision.fact.get("correction") or "").strip()
                timing = _intervention_timing_metadata(
                    kind="fact",
                    now=_now,
                    silence_elapsed=_silence_elapsed,
                    pause_required=_INTERVENTION_PAUSE_FACT,
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
                timing = _intervention_timing_metadata(
                    kind="manual",
                    now=_now,
                    silence_elapsed=_silence_elapsed,
                    pause_required=_INTERVENTION_PAUSE_MANUAL,
                    queued_at=float(_pending.manual_call.get("created_at", _now))
                    if _pending.manual_call else _now,
                    policy="manual_call_pause",
                )
                print(f"# [trigger] manual_call: {detail}", flush=True)
                _log_intervention_event(
                    state, "manual_call", detail,
                    timing={**timing, "source": manual.get("source", "ui"),
                            "request": request})
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
                    pause_required=_INTERVENTION_PAUSE_DRIFT,
                    queued_at=_pending.last_drift_request_at or None,
                    queued_wall_at=_pending.last_drift_request_wall_at,
                    policy="drift_confirmation_pause",
                )
                print(f"# [trigger] drift: 脱線介入「{decision.drift_reason}」",
                      flush=True)
                _log_intervention_event(
                    state, "drift", decision.drift_reason, timing=timing)
                agent.trigger(topics=_bargein_topics,
                              drift_reason=decision.drift_reason)
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
                    pause_required=_INTERVENTION_PAUSE_RETRY,
                    queued_at=float(pending_intervention.get("created_at", _now)),
                    policy="retry_extra_pause",
                )
                print("# [trigger] retry: 中断された介入を再送（ガードバイパス）",
                      flush=True)
                _log_intervention_event(
                    state, "retry", "中断された介入を再送", timing=timing)
                agent.trigger(topics=_bargein_topics)
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
        _silence_elapsed = time.monotonic() - _last_utt_time[0]
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
            print(f"# [diag] controller normal fallback: {exc}", flush=True)
            normal_decision = _select_normal_trigger_decision(
                pending=_pending, agent=agent, silence_elapsed=_silence_elapsed,
                silence_summarize=_silence_summarize,
                partner_present=partner is not None,
                now=time.monotonic(),
                last_intervention_at=_last_intervention_at, cooldown=_cooldown,
                last_invited=_last_invited)
            _ctrl_normal = None
            _normal_cands = []
            _ctrl_latency_ms = 0.0
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
        if normal_decision.reason == "conversation":
            _log_intervention_event(state, "conversation", normal_decision.detail)
            agent.trigger()
            _note_intervention(time.monotonic(), "conversation", normal_decision.detail)
        elif normal_decision.reason == "count":
            timing = _intervention_timing_metadata(
                kind="count",
                now=time.monotonic(),
                silence_elapsed=_silence_elapsed,
                pause_required=_INTERVENTION_PAUSE_COUNT,
                policy="turn_count_pause",
            )
            print(f"# [trigger] count: {normal_decision.detail}", flush=True)
            _log_intervention_event(state, "count", normal_decision.detail, timing=timing)
            agent.trigger(topics=_topics)
            _last_intervention_at = time.monotonic()
            _note_intervention(_last_intervention_at, "count", normal_decision.detail)
        elif normal_decision.reason == "silence":
            timing = _intervention_timing_metadata(
                kind="silence",
                now=time.monotonic(),
                silence_elapsed=_silence_elapsed,
                pause_required=float(_silence_summarize or 0.0),
                policy="silence_summary",
            )
            print(f"# [trigger] silence: {normal_decision.detail}", flush=True)
            _log_intervention_event(
                state, "silence", normal_decision.detail, timing=timing)
            agent.trigger(topics=_topics)
            _last_intervention_at = time.monotonic()
            _note_intervention(_last_intervention_at, "silence", normal_decision.detail)
        elif normal_decision.reason == "invite":
            timing = _intervention_timing_metadata(
                kind="invite",
                now=time.monotonic(),
                silence_elapsed=_silence_elapsed,
                pause_required=_INVITE_SILENCE,
                policy="participation_pause",
            )
            print(f"# [trigger] invite: {normal_decision.detail}", flush=True)
            _log_intervention_event(
                state, "invite", normal_decision.detail, timing=timing)
            agent.trigger(topics=_topics,
                          invite_target=normal_decision.invite_target)
            _last_intervention_at = time.monotonic()
            _last_invited = normal_decision.invite_target
            _pending.invite = None
            _note_intervention(_last_intervention_at, "invite", normal_decision.detail)
        elif normal_decision.reason == "skip_invite":
            _pending.invite = None  # 同じ人を連続では誘わない


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
                with state.state_lock:
                    state.names["#" + label] = name
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


def _run_from_wav(state: SessionState, args):
    """WAVファイルを擬似ライブで送信する.

    Reactive WAV: agentが発話中はWAV再生・ASR送信を一時停止し、
    介入終了後に自動再開する。
    """
    import librosa
    agent = state.agent
    y, _ = librosa.load(args.wav, sr=SR)
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
        if not setup_capture_only and state.pcm_file is not None:
            try:
                state.pcm_file.write(pcm)
                state.pcm_file.flush()
            except OSError:
                pass
        if ws is not None:
            try:
                ws.send(pcm)
            except Exception:
                pass
            else:
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

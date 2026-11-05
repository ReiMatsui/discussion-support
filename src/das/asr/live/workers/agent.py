"""常駐処理の責務: agent。会話の判定規則・時刻条件は元実装を維持する。"""
from __future__ import annotations

import collections
import contextlib
import queue
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._session_state import SessionState

from .._constants import (
    _INTERVENTION_COOLDOWN,
    AGENT_SPEAKER,
    WORKER_TICK_SEC,
)
from .._facilitation import (
    FacilitationController,
    InterventionLogEntry,
    policy_for,
)
from .._intervention import (
    _NORMAL_SPECS,
    _controller_barge_in_decision,
    _controller_normal_decision,
    _effective_silence,
    _intervention_enabled,
    _intervention_timing_metadata,
    _InterventionReviewRecorder,
    _legacy_decision_brief,
    _log_intervention_event,
    _NormalTriggerDecision,
    _PendingInterventions,
    _recent_agent_texts,
    _set_manual_status,
)
from .._speaker_policy import (
    intervention_records,
    intervention_speaker_name,
)
from .af import _af_gate_status, _AfDeliveryGate
from .triage import _answered_directly


class _AgentWorker:
    """AI応答のトリガー管理（ターンテイキング）を1tickずつ回す.

    自然な会話のフロア交代を模倣する:

      - 人間のターン: 発話を即座に feed し、沈黙で譲渡 → AI が trigger
      - AI のターン: 応答を再生。人間の実質的な発話で自動 interrupt
      - AI ターン終了: フロアを人間に返す（沈黙タイマーをリセット）

    **なぜクラスなのか**: 1tick の処理は7段あり、そのほとんどが「直近の介入
    時刻」「誰を最後に誘ったか」「生成先行で抱えている af」といった**tickを
    またいで持ち越す値**を読み書きする。関数に切り出すと引数が十数個並ぶので、
    持ち越す値は属性に置き、段をメソッドにする。tick 内で閉じる値（partner /
    cooldown など）は引数で渡す——どちらなのかが署名で分かるようにするため。

    段の順番（`run` の while が読める形にしてある）:

      1. エージェントの生存確認（切れていれば再接続を試みて次tickへ）
      2. 新しい発話の取り込みと自動割り込み
      3. 介入オフなら溜まった要求を捨てる
      4. af 生成先行ゲート（--af 有効時のみ）
      5. 割り込み介入（事実補正・脱線・手動・リトライ）
      6. エコー窓/パートナー発話中は黙る
      7. 通常介入（要約・沈黙・声かけ）
    """

    def __init__(self, state: SessionState) -> None:
        self.state = state
        self.agent = state.agent
        self.last_utt_time = state._last_utt_time
        self.was_in_echo = state._was_in_echo
        self.diag_tick = 0
        self.last_intervention_at = 0.0   # 直近の介入時刻（cooldown の時計）
        self.last_invited: str | None = None   # 直近に声をかけた相手（連続回避）
        self.last_agent_reconnect_at = 0.0
        self.last_partner_reconnect_at = 0.0
        self.pending = _PendingInterventions()
        # 採否Controller: 固定優先順位に代わり最終採否を担当する。物理タイミング
        # （floor/barge-in）と fact fast lane は維持しつつ、「どの候補を今採るか／
        # 黙るか」を一元裁定する。
        self.controller = FacilitationController()
        # af 生成先行・再生ゲート（フェーズ6, --af 有効時のみ作動。af 限定）。
        self.af_gate = _AfDeliveryGate()
        # 採否の経緯（採択/抑制/latency）を intervention_review.jsonl へ記録する。
        self.review = _InterventionReviewRecorder()
        # maxlen はクールダウン照会に加えて同一内容抑止（duplicate_content, 10分窓）
        # の照会範囲を兼ねる。介入は最短でも十数秒間隔なので 30 件で窓を十分覆う。
        self.recent_interventions: collections.deque[InterventionLogEntry] = (
            collections.deque(maxlen=30))

    # -- 記帳 ---------------------------------------------------------

    def note_intervention(self, at: float, kind: str, detail: str = "",
                          *, invite_target: str | None = None) -> None:
        """実際に発火した介入を記録する（発火の副作用はここに一本化する）.

        記録するのは3つ:
          - 直近履歴 `recent_interventions`（同一内容抑止 duplicate_content 用）
          - `last_intervention_at`（cooldown の時計）
          - `last_invited`（同じ人への連続声かけの抑止。指名した時だけ）

        **時計と声かけ相手をここで進めるのは、分岐ごとの更新漏れを構造的に防ぐ
        ため**。発火分岐は種別ごとの巨大な if 連鎖で、実際に漏れが2件起きた
        （2026-07-25 の監査）:
          - `conversation` だけ時計を更新しておらず、会話モードでAIが応答した
            直後に `invite` 候補が古い時刻を基準に cooldown を通過していた
          - `af_l1`/`af_l2` は人を指名するのに `last_invited` を更新せず、
            同じ人への連続指名の抑止が効いていなかった
        以後、新しい発火分岐を足すときはこれを呼ぶだけでよい。
        """
        self.recent_interventions.append(
            InterventionLogEntry(at=at, kind=kind, brief=detail))
        self.last_intervention_at = at
        if invite_target:
            self.last_invited = invite_target

    # -- 1. エージェントの生存確認 -------------------------------------

    def _agent_ready(self) -> bool:
        """使える状態か。切れていれば再接続を試み、False を返して次tickへ."""
        agent = self.agent
        if agent is not None and agent.enabled and getattr(agent, "ready", agent._connected):
            return True
        if agent is None or not agent.enabled:
            self.pending.clear_all()
        if agent is not None and agent.enabled and getattr(agent, "_reopening", False):
            return False   # 開き直し中（mode/voice 変更）。候補は保持し、介入は出さない
        if agent is not None and agent.enabled and not agent._connected:
            now = time.monotonic()
            if now - self.last_agent_reconnect_at >= 5.0:
                self.last_agent_reconnect_at = now
                print("# AI Agent: 再接続を試みます", flush=True)
                with contextlib.suppress(Exception):
                    agent.connect()
        if self.diag_tick % 20 == 0:
            print(f"# [diag] _agent_worker skip: agent={agent is not None}"
                  f" conn={agent._connected if agent else '?'}"
                  f" enabled={agent.enabled if agent else '?'}", flush=True)
        return False

    def _keep_partner_connected(self, partner) -> None:
        """相手役の接続が切れていたら 5 秒間隔で張り直す（誰も見ていないと沈黙したまま）."""
        if partner is None or not getattr(partner, "enabled", False):
            return
        if partner._connected or getattr(partner, "_reopening", False):
            return
        now = time.monotonic()
        if now - self.last_partner_reconnect_at < 5.0:
            return
        self.last_partner_reconnect_at = now
        print("# Partner: 再接続を試みます", flush=True)
        with contextlib.suppress(Exception):
            partner.connect()

    # -- 2. 新しい発話の取り込み ---------------------------------------

    def _ingest_utterances(self, *, partner, enabled: bool) -> tuple[bool, bool]:
        """新しい発話をエージェントへ流し、必要なら割り込む.

        戻り値: (このtickを続けてよいか, 新規発話が来たか)。会議がリセット
        された（epoch がずれた）ときは続行不可——古い会議の発話を新しい
        エージェントに流さないため、このtickは丸ごと捨てる（H2）。
        """
        s = self.state
        agent = self.agent
        with s.state_lock:
            meeting_epoch = s.meeting_epoch
            skip = {AGENT_SPEAKER, "パートナー"}
            talk_rs = [r for r in s.records
                       if "speaker" in r and r.get("text")
                       and r.get("speaker") not in skip]
        n = len(talk_rs)
        if n <= s.agent_cursor:
            return True, False
        # raw スライス（話者未確定も含む）: 割り込み判定はこちらを使う。
        # 「遮る場面ほど帰属が壊れ、止まらない」問題への対処（C1）。
        raw_new = talk_rs[s.agent_cursor:]
        new_records = intervention_records(raw_new)
        # 発話供給（副作用）の直前で epoch 確認（H2）。
        with s.state_lock:
            if s.meeting_epoch != meeting_epoch:
                return False, False
        af_new_utt = False
        if new_records:
            self.last_utt_time[0] = time.monotonic()
            af_new_utt = True   # フロア成立前なら af 生成先行を cancel する材料
        if enabled:
            # 音声呼びかけの検出は _run_triage_worker（LLM分類）が担う。
            # ここでは発話をエージェントに供給するだけ。
            for r in new_records:
                agent.feed(intervention_speaker_name(self.state, r),
                           r.get("text", ""))
        # cursor 書き戻しは epoch 再確認と同一 lock で atomic に行う（H2）。
        with s.state_lock:
            if s.meeting_epoch != meeting_epoch:
                return False, af_new_utt
            s.agent_cursor = n
        # ファシリテーターもパートナーも GPT-Live が室内の声を聞いて自分で
        # 止まり、応じる。こちらでは割り込みを検出しない（WP2 / WP7）。
        return True, af_new_utt

    # -- 3. 介入オフ ---------------------------------------------------

    def _discard_while_disabled(self) -> None:
        """介入オフの間は、溜まっている要求も入ってくる要求も全部捨てる."""
        s = self.state
        if (self.pending.manual_call is not None
                or not s.manual_call_requests.empty()):
            _set_manual_status(s, "cancelled", detail="介入オフのため破棄")
        self.pending.clear_all()
        for q in (s.drift_requests, s.invite_requests, s.factcheck_requests,
                  s.manual_call_requests, s.summarize_requests):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        if self.diag_tick % 20 == 0:
            print("# [diag] agent: intervention disabled", flush=True)

    # -- 4. af 生成先行ゲート ------------------------------------------

    def _run_af_gate(self, *, partner, cooldown, af_new_utt: bool) -> None:
        """af の配信ゲート（--af 有効時のみ・af 限定）.

        AF が無効（既定）ならこのブロックは丸ごとスキップされ、ルールベースの
        採否・trigger 経路は一切変わらない（モード方針）。summarize には関与
        しない。WHEN は Controller に委譲し、ゲートは配信だけを担う。
        """
        s = self.state
        agent = self.agent
        try:
            now = time.monotonic()
            silence = _effective_silence(s, now, self.last_utt_time)
            with s.topics_lock:   # topics 読み出しは lock 取得で統一 (T9-5)
                af_topics = list(s.topics) if s.topics else None
            partner_busy = bool(
                partner is not None
                and (partner.ai_speaking or partner._responding))
            status, payload = _af_gate_status(
                self.controller, self.pending, agent,
                now=now,
                silence_elapsed=silence,
                recent_interventions=list(self.recent_interventions),
                cooldown=cooldown,
                last_intervention_at=self.last_intervention_at,
                epoch=s.agent_cursor,
                partner_busy=partner_busy,
                in_echo_window=bool(getattr(agent, "in_echo_window", False)),
            )
            action = self.af_gate.tick(
                agent=agent,
                af=payload,
                status=status,
                silence=silence,
                agent_busy=bool(agent._responding or agent.ai_speaking),
                topics=af_topics,
            )
            if action == "deliver":
                held = payload or {}
                kind = str(held.get("kind") or "af_l1")
                text = str(held.get("af_text") or "")
                rt = s.af_runtime
                if rt is not None and text:
                    with contextlib.suppress(Exception):
                        rt.note_intervention(kind, text)
                self.pending.clear_af()   # 消費
                self.note_intervention(now, kind, "af 配信",
                                       invite_target=held.get("target_speaker"))
                _log_intervention_event(
                    s, kind, "af 配信",
                    timing=_intervention_timing_metadata(
                        kind=kind, now=now,
                        silence_elapsed=silence,
                        pause_required=policy_for(kind).pause,
                        policy="af_intervention"))
        except Exception as e:  # pragma: no cover - 防御的
            print(f"# [af] gate error: {e}", flush=True)

    # -- 5. 割り込み介入 -----------------------------------------------

    def _try_barge_in(self, *, partner, cooldown, silence_summarize) -> bool:
        """最優先のバージイン（①事実補正 ②手動呼び出し ③脱線 ④リトライ）.

        agent が free（応答中でも発話中でもない）になった瞬間に、エコー窓・
        パートナー発話・沈黙閾値を無視してトリガーする。会話が活発でも
        取りこぼさないため。`trigger()` の呼び出しはこのワーカーに一元化されて
        いる（R2）。

        戻り値: True なら このtickはここで終わり（発火した／見送った）。
        """
        s = self.state
        agent = self.agent
        now = time.monotonic()
        # F3: アクティブな partial 中はフロア占有として沈黙 0 に倒す。
        silence_elapsed = _effective_silence(s, now, self.last_utt_time)
        partner_busy = bool(partner is not None
                            and (partner.ai_speaking or partner._responding))
        topics = None
        if agent.mode != "conversation":
            with s.topics_lock:
                topics = list(s.topics) if s.topics else None
        try:
            decision, ctrl, cands, latency_ms = _controller_barge_in_decision(
                self.controller,
                pending=self.pending,
                agent=agent,
                state=s,
                now=now,
                last_intervention_at=self.last_intervention_at,
                silence_elapsed=silence_elapsed,
                partner_busy=partner_busy,
                in_echo_window=bool(agent.in_echo_window),
                cooldown=cooldown,
                recent_interventions=list(self.recent_interventions),
                silence_summarize=silence_summarize,
                last_invited=self.last_invited,
                epoch=s.agent_cursor,
            )
        except Exception as exc:
            # Controller は決定的（LLMなし）なので、ここに来るのはバグのみ。
            # 別実装へ fallback すると障害時に挙動が静かに変わるため（M1）、
            # このtickは何もせず、次のtickで再評価する。
            print(f"# [diag] controller barge-in error（このtickは見送り）: {exc}",
                  flush=True)
            return True
        # 古い判断の破棄（§8.5）: 裁定後に新しい発話で世代がずれたら採らない。
        if (decision.reason not in ("none", "hold")
                and ctrl is not None
                and ctrl.valid_for_epoch != s.agent_cursor):
            print("# [trigger] skip: stale decision (epoch changed)", flush=True)
            return True
        if decision.reason != "none":
            self._record_barge_in_review(
                decision, ctrl, cands, latency_ms, silence_elapsed)
        # hold は「barge-in レーンに今すぐ話せる候補が無い」だけを意味する。
        # ここで打ち切ると、確認待ちの drift 候補1件で他の介入が無期限に飢餓
        # する（C1）。none と同様に通常レーンへ落とし、そちらの pause/cooldown
        # で判断させる。
        return self._fire_barge_in(decision, topics=topics, now=now,
                                   silence_elapsed=silence_elapsed)

    def _record_barge_in_review(self, decision, ctrl, cands, latency_ms,
                                silence_elapsed: float) -> None:
        """採否の経緯を記録する（判断には影響しない）."""
        if decision.reason == "fact" and decision.fact is not None:
            legacy = _legacy_decision_brief(
                "fact", str(decision.fact.get("correction") or "").strip())
        elif decision.reason == "manual" and decision.manual is not None:
            legacy = _legacy_decision_brief(
                "manual_call",
                str(decision.manual.get("request") or "").strip()
                or "直近の議論整理")
        elif decision.reason == "drift" and decision.drift_reason is not None:
            legacy = _legacy_decision_brief("drift", decision.drift_reason)
        else:
            legacy = _legacy_decision_brief(decision.reason)
        if ctrl is not None:
            self.review.record(
                self.state, candidates=cands, decision=ctrl,
                silence_elapsed=silence_elapsed, epoch=self.state.agent_cursor,
                legacy=legacy, latency_ms=latency_ms)

    def _fire_barge_in(self, decision, *, topics, now: float,
                       silence_elapsed: float) -> bool:
        """割り込みを1件発火する（戻り: 発火したらTrue＝このtickは終わり）."""
        s = self.state
        agent = self.agent
        if decision.reason == "fact" and decision.fact is not None:
            correction = str(decision.fact.get("correction") or "").strip()
            timing = _intervention_timing_metadata(
                kind="fact", now=now, silence_elapsed=silence_elapsed,
                pause_required=policy_for("fact").pause,
                queued_at=float(decision.fact.get("_queued_at", now)),
                queued_wall_at=str(decision.fact.get("_queued_wall_at") or ""),
                policy="fact_freshness_pause")
            if agent.trigger(topics=topics, fact_correction=decision.fact) is False:
                return False             # 送れなかった。候補は残して次の tick で再判断
            print(f"# [trigger] fact: {correction}", flush=True)
            _log_intervention_event(s, "fact", correction, timing=timing)
            self.pending.facts.popleft()
            self.note_intervention(time.monotonic(), "fact", correction)
            return True
        if decision.reason == "manual" and decision.manual is not None:
            manual = decision.manual
            request = str(manual.get("request") or "").strip()
            detail = request or "直近の議論整理"
            queued_payload = self.pending.manual_call or {}
            queued_at = float(queued_payload.get("created_at", now))
            if manual.get("source") == "voice" and _answered_directly(agent, queued_at):
                # GPT-Live は室内の音声を聞いていて、呼びかけには自分で答える。
                # STT→triage 経由で同じ依頼が届いても二度目は話させない
                print("# [trigger] manual_call: 呼びかけには直接応答済み（再送しない）", flush=True)
                _log_intervention_event(
                    s, "manual_call", detail,
                    timing={"source": "voice", "request": request, "queued_at": queued_at,
                            "outcome": "answered_directly"})
                _set_manual_status(s, "delivered", detail=detail, wait_sec=now - queued_at)
                self.pending.clear_manual()
                return False
            timing = _intervention_timing_metadata(
                kind="manual", now=now, silence_elapsed=silence_elapsed,
                pause_required=policy_for("manual").pause,
                queued_at=queued_at,
                queued_wall_at=str(queued_payload.get("created_wall_at") or ""),
                policy="manual_call_pause")
            if agent.trigger(topics=topics, manual_request=manual,
                             recent_agent_texts=_recent_agent_texts(s)) is False:
                return False
            print(f"# [trigger] manual_call: {detail}", flush=True)
            _log_intervention_event(
                s, "manual_call", detail,
                timing={**timing, "source": manual.get("source", "ui"),
                        "request": request, "queued_at": queued_at,
                        "outcome": "selected"})
            _set_manual_status(s, "dispatched", detail=detail,
                               wait_sec=now - queued_at)
            self.pending.clear_manual()
            self.note_intervention(time.monotonic(), "manual", detail)
            return True
        if decision.reason == "drift" and decision.drift_reason is not None:
            timing = _intervention_timing_metadata(
                kind="drift", now=now, silence_elapsed=silence_elapsed,
                pause_required=policy_for("drift").pause,
                queued_at=self.pending.last_drift_request_at or None,
                queued_wall_at=self.pending.last_drift_request_wall_at,
                policy="drift_confirmation_pause")
            if agent.trigger(topics=topics, drift_reason=decision.drift_reason,
                             recent_agent_texts=_recent_agent_texts(s)) is False:
                return False
            print(f"# [trigger] drift: 脱線介入「{decision.drift_reason}」", flush=True)
            _log_intervention_event(s, "drift", decision.drift_reason, timing=timing)
            self.pending.clear_drift()
            self.note_intervention(time.monotonic(), "drift", decision.drift_reason)
            return True
        return False

    # -- 6. 黙る（エコー窓・パートナー発話中） --------------------------

    def _hold_floor(self, why: str, *, partner, silence_summarize) -> None:
        """今は喋らないと決めた理由を記録する（発話はしない）."""
        self.was_in_echo[0] = True
        now = time.monotonic()
        self.review.evaluate(
            self.state,
            pending=self.pending,
            agent=self.agent,
            now=now,
            silence_elapsed=now - self.last_utt_time[0],
            epoch=self.state.agent_cursor,
            recent_interventions=list(self.recent_interventions),
            legacy=_legacy_decision_brief("hold", why),
            silence_summarize=silence_summarize,
            partner_present=partner is not None,
            last_invited=self.last_invited,
        )

    # -- 7. 通常介入 ---------------------------------------------------

    def _try_normal(self, *, partner, cooldown, silence_summarize) -> None:
        """通常レーン（要約・沈黙・声かけ・af）の採否を決めて発火する."""
        s = self.state
        agent = self.agent
        if self.diag_tick % 20 == 0:
            elapsed = time.monotonic() - self.last_utt_time[0]
            print(f"# [diag] agent: mode={agent.mode} pending={agent.pending_count}"
                  f" trigger_n={agent.trigger_n} responding={agent._responding}"
                  f" silence={elapsed:.1f}s echo={agent.in_echo_window}"
                  f" partner_talk={partner.ai_speaking if partner else '?'}",
                  flush=True)
        # --- 論点一覧を取得（facilitatorモードのみ） ---
        topics = None
        if agent.mode != "conversation":
            with s.topics_lock:
                topics = list(s.topics) if s.topics else None
        # F3: アクティブな partial 中はフロア占有として沈黙 0 に倒す。
        silence_elapsed = _effective_silence(s, time.monotonic(), self.last_utt_time)
        try:
            decision, ctrl, cands, latency_ms = _controller_normal_decision(
                self.controller,
                pending=self.pending,
                agent=agent,
                now=time.monotonic(),
                silence_elapsed=silence_elapsed,
                silence_summarize=silence_summarize,
                partner_present=partner is not None,
                last_intervention_at=self.last_intervention_at,
                cooldown=cooldown,
                last_invited=self.last_invited,
                recent_interventions=list(self.recent_interventions),
                epoch=s.agent_cursor,
            )
        except Exception as exc:
            # barge-in 側と同じ方針: 別実装へ fallback せず、このtickは見送る（M1）。
            print(f"# [diag] controller normal error（このtickは見送り）: {exc}",
                  flush=True)
            return
        # 古い判断の破棄（§8.5）。
        if (decision.reason not in ("none", "skip_invite")
                and ctrl is not None
                and ctrl.valid_for_epoch != s.agent_cursor):
            print("# [trigger] skip: stale normal decision (epoch changed)", flush=True)
            return
        detail = decision.detail
        if decision.reason == "skip_invite":
            detail = f"{decision.invite_target}さんへの連続声かけを抑制"
        legacy = _legacy_decision_brief(decision.reason, detail)
        if ctrl is not None:
            self.review.record(
                s, candidates=cands, decision=ctrl,
                silence_elapsed=silence_elapsed, epoch=s.agent_cursor,
                legacy=legacy, latency_ms=latency_ms)
        else:
            self.review.evaluate(
                s, pending=self.pending, agent=agent, now=time.monotonic(),
                silence_elapsed=silence_elapsed, epoch=s.agent_cursor,
                recent_interventions=list(self.recent_interventions),
                legacy=legacy, silence_summarize=silence_summarize,
                partner_present=partner is not None,
                last_invited=self.last_invited)
        self._fire_normal(decision, silence_elapsed, silence_summarize, topics)

    def _fire_normal(self, decision: _NormalTriggerDecision,
                     silence_elapsed: float, silence_summarize, topics) -> None:
        """通常介入を1件発火する（種別によらず手順は同じ。差分は _NORMAL_SPECS）.

        手順: timing算出 → 表示 → イベント記録 → agent.trigger → 種別ごとの
        後始末（AF受容計測・pending消費）→ 記帳。記帳は note_intervention に
        一本化してあるので、cooldown の時計と「誰を誘ったか」は種別を問わず
        必ず更新される（handoff §22.1 の再発防止）。
        """
        kind = decision.reason
        if kind == "skip_invite":
            self.pending.invite = None   # 同じ人を連続では誘わない（発話はしない）
            return
        spec = _NORMAL_SPECS.get(kind)
        if spec is None:
            return                       # "none" 等、発火しない判断
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
        available = {
            "topics": topics,
            "summary_focus": decision.summary_focus,
            "invite_target": decision.invite_target,
            "af_presentation": decision.af_text,
            "recent_agent_texts": _recent_agent_texts(self.state),
            "silence_sec": silence_elapsed,
        }
        # 送れなかった（開き直し中・話し中・送信失敗）なら候補を消費せず、記帳もしない。
        # 次の tick で Controller が改めて判断する
        if self.agent.trigger(**{k: available[k] for k in spec.trigger}) is False:
            return
        if spec.policy is not None:
            print(f"# [trigger] {kind}: {shown}", flush=True)
        _log_intervention_event(self.state, kind, decision.detail, timing=timing)
        if "af_presentation" in spec.trigger and decision.af_text:
            # 受容計測 (フェーズ5): 配信した af 介入を AF ランタイムに記録する。
            rt = getattr(self.state, "af_runtime", None)
            if rt is not None:
                with contextlib.suppress(Exception):
                    rt.note_intervention(kind, decision.af_text)
        if spec.consume == "af":
            self.pending.clear_af()
        elif spec.consume == "summarize":
            self.pending.summarize = None   # 採択したら消費（drainで再取得しない）
        elif spec.consume == "invite":
            self.pending.invite = None
        self.note_intervention(
            time.monotonic(), kind, decision.detail,
            invite_target=(decision.invite_target
                           if "invite_target" in spec.trigger else None))

    # -- ループ本体 -----------------------------------------------------

    def run(self) -> None:
        """停止が要求されるまで、1tick=0.25秒で段を順に回す."""
        s = self.state
        while not s.stop.is_set():
            time.sleep(WORKER_TICK_SEC)
            self.diag_tick += 1
            partner = s.partner   # 動的参照: 実行中の接続/切断に追従（F3）
            self._keep_partner_connected(partner)
            if not self._agent_ready():
                continue
            # 積極性プロファイル（S5）: 介入クールダウンと沈黙要約の閾値
            cooldown = s.proactivity.get("cooldown", _INTERVENTION_COOLDOWN)
            silence_summarize = s.proactivity.get("silence_summarize")
            enabled = _intervention_enabled(s)

            ok, af_new_utt = self._ingest_utterances(
                partner=partner, enabled=enabled)
            if not ok:
                continue
            if not enabled:
                self._discard_while_disabled()
                continue
            # drift_checker/participation_checker からの要求を回収（R2/S4）。
            # busy でも取りこぼさないよう、キューは毎ループ必ず drain する。
            self.pending.drain(s, now=time.monotonic())
            if getattr(s, "af_runtime", None) is not None:
                self._run_af_gate(partner=partner, cooldown=cooldown,
                                  af_new_utt=af_new_utt)
            agent_free = (not self.agent._responding
                          and not self.agent.ai_speaking)
            if agent_free and self._try_barge_in(
                    partner=partner, cooldown=cooldown,
                    silence_summarize=silence_summarize):
                continue
            if self.agent.in_echo_window:          # エコー窓中は trigger しない
                self._hold_floor("echo_window", partner=partner,
                                 silence_summarize=silence_summarize)
                continue
            if partner is not None and (partner.ai_speaking or partner._responding):
                self._hold_floor("partner_busy", partner=partner,
                                 silence_summarize=silence_summarize)
                continue
            # --- フロア返却 ---
            if self.was_in_echo[0]:
                self.was_in_echo[0] = False
                self.last_utt_time[0] = time.monotonic()
            self._try_normal(partner=partner, cooldown=cooldown,
                             silence_summarize=silence_summarize)


def _run_agent_worker(state: SessionState):
    """バックグラウンドでAI応答のトリガーを管理する（本体は `_AgentWorker`）.

    このスレッド1本が介入の発火と GPT-Live の再接続を兼ねる。予期しない例外で
    死ぬと介入も再接続も静かに止まるので、他のワーカーと同じく理由を出して
    再入する。ワーカーの状態（候補・cooldown の時計）は同じインスタンスに残す。
    """
    import traceback
    worker = _AgentWorker(state)
    while not state.stop.is_set():
        try:
            worker.run()
            return
        except Exception:
            traceback.print_exc()
            print("# [_run_agent_worker] 予期しないエラー。5秒後に再開します", flush=True)
            if state.stop.wait(timeout=5.0):
                return


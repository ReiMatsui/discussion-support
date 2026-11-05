"""常駐処理の責務: checks。会話の判定規則・時刻条件は元実装を維持する。"""
from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._session_state import SessionState

from .._constants import (
    _AGENDA_MIN_UTTS,
    _AGENDA_RETRY_SEC,
    _AGENDA_WINDOW,
    _DRIFT_CHECK_INTERVAL,
    _DRIFT_CHECK_WINDOW,
    _DRIFT_RUN_SEC_DEFAULT,
    _DRIFT_WARMUP,
    _FACTCHECK_CHECK_SEC,
    _FACTCHECK_MAX_RETRIES,
    _INVITE_CHECK_SEC,
    _INVITE_QUIET_RATIO,
    _INVITE_WARMUP,
    _STRUCTURING_WINDOW,
    AGENT_SPEAKER,
    WORKER_TICK_SEC,
)
from .._drift import DriftRun
from .._intervention import (
    _as_bool,
    _intervention_enabled,
)
from .._participation import (
    apply_diarization_time,
    diarization_time_stats,
    participation_share_key,
    participation_share_label,
    participation_stats,
    quietest_participation_share,
)
from .._speaker_policy import (
    intervention_records,
    intervention_speaker_name,
    reliable_human_records,
)
from .._ui import _print_line
from .common import _resilient


@_resilient
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
                # 議題がある間は何もしない。ただしスレッドは終了しない——
                # 終了すると「新しい会議」で topics がクリアされた後、2会議目
                # 以降の議題自動検出が二度と動かない（T3 の epoch リセットが
                # 意図している「リセットをまたいで動く」が実現しない。
                # レビュー 2026-07-30）。
                continue
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
            continue  # topics ができたので、次周から上の continue 側で待機


@_resilient
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


@_resilient
def _run_drift_checker(state: SessionState, oai_key: str, oai_model: str):
    """脱線検出のバックグラウンドワーカー（並列監視）.

    2発話ごとに直近の発話へ on / aside / off を付け、DriftRun で「離れている時間」
    を測る。閾値（積極性プロファイルの drift_run_sec）を超えたら理由を
    state.drift_requests に積む。実際のトリガーは _run_agent_worker が裁定する
    （R2: このワーカーは trigger() を呼ばない）。

    人間・パートナー双方の発話をチェック対象に含める。
    パートナーが脱線に付き合っている状態も検出するため。
    """
    from das.asr.live._bootstrap import label_drift as _label_drift

    _diag_tick = 0
    run = DriftRun()
    epoch_seen = state.meeting_epoch
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
        if state.meeting_epoch != epoch_seen:
            epoch_seen = state.meeting_epoch
            run = DriftRun()

        with state.topics_lock:
            topics = list(state.topics)
        agenda = [str(t.get("topic") or "") for t in topics
                  if t.get("speaker") in ("議題", "議題(自動)")]
        flow = [str(t.get("topic") or "") for t in topics
                if t.get("speaker") not in ("議題", "議題(自動)")]
        if not agenda and not flow:
            if _diag_tick % 30 == 0:
                print("# [drift] 待機中: 論点未抽出", flush=True)
            continue
        run.update_anchors(flow)
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
        window = talk_rs[max(0, n - _DRIFT_CHECK_WINDOW):]
        utts = [{"speaker": intervention_speaker_name(state, r), "text": r["text"]}
                for r in window]
        # cursor 書き戻し（副作用）の直前で epoch 確認（H2）: リセット跨ぎを破棄。
        with state.state_lock:
            if state.meeting_epoch != epoch:
                continue
            state.drift_cursor = n
        result = _label_drift(utts, agenda, run.anchor_topics, oai_key, oai_model)
        labels = result.get("labels") or []
        if not labels:
            continue
        with state.state_lock:
            if state.meeting_epoch != epoch:
                continue
        run.observe([{"ms": r.get("ms"), "end_ms": r.get("end_ms")} for r in window],
                    labels, result.get("reason", ""))
        threshold = float(state.proactivity.get("drift_run_sec", _DRIFT_RUN_SEC_DEFAULT))
        print(f"# [drift] 判定: {''.join('○' if x == 'on' else '△' if x == 'aside' else '×' for x in labels)}"
              f" 離れている時間={run.run_sec():.0f}s/{threshold:.0f}s"
              f"{'（' + run.reason + '）' if run.reason else ''}", flush=True)
        reason = run.should_fire(threshold_sec=threshold)
        if reason:
            _print_line(f"# 🔀 脱線が{threshold:.0f}秒以上続いています: {reason}")
            state.drift_requests.put(reason)
            print("# [drift] → 介入要求をキューに投入", flush=True)


@_resilient
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
        time.sleep(WORKER_TICK_SEC)
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


@_resilient
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
            epoch = state.meeting_epoch
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
        # 分離(diarization)の実測時間があれば時間軸をそれに置き換える
        # （§49.17 案D）。未確定に落ちた発話の時間も本人のクラスタに乗る
        # ため、「実はよく喋っている人」を静かと誤認しない。records側の
        # 全話者を分離が捕捉しているときだけ適用（物差しの混在を避ける。
        # Soniox単独=分離なしでは何も変わらない）。
        with state.diarization_lock:
            _ev = list(state.diarization_events)
        if _ev:
            with state.state_lock:
                _keymap = dict(state.diarization_speaker_keys)
            _diar = diarization_time_stats(
                _ev, lambda e, _m=_keymap: _m.get(f"{e.source}:{e.speaker}"))
            stats = apply_diarization_time(stats, _diar)
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
            with state.state_lock:
                if state.meeting_epoch != epoch:
                    # LLM呼び出し中に「新しい会議」が起きた。旧会議の統計に
                    # 基づく声かけを新会議のキューへ入れない（他ワーカーの
                    # H2 ガードと同じ。ここだけ無かった。レビュー 2026-07-30）。
                    continue
            _print_line(f"# 🙋 声かけ候補: {target}（{result.get('reason', '')}）")
            state.invite_requests.put(target)
            print("# [invite] → 声かけ要求をキューに投入", flush=True)


@_resilient
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


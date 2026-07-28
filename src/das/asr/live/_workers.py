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
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ._session_state import SessionState

from ._constants import (
    _ACK_CHIME_ENABLED,
    _AGENDA_MIN_UTTS,
    _AGENDA_RETRY_SEC,
    _AGENDA_WINDOW,
    _DRIFT_CHECK_INTERVAL,
    _DRIFT_CHECK_WINDOW,
    _DRIFT_WARMUP,
    _FACTCHECK_CHECK_SEC,
    _FACTCHECK_MAX_RETRIES,
    _INTERRUPT_MIN_CHARS,
    _INTERVENTION_COOLDOWN,
    _INVITE_CHECK_SEC,
    _INVITE_QUIET_RATIO,
    _INVITE_WARMUP,
    _MANUAL_CALL_MAX_CHARS,
    _STRUCTURING_WINDOW,
    _TRIAGE_BACKLOG_MAX,
    _TRIAGE_CONTEXT_WINDOW,
    _TRIAGE_MAX_RETRIES,
    _TRIAGE_MIN_CHARS,
    AGENT_SPEAKER,
    SR,
    WORKER_TICK_SEC,
)
from ._facilitation import (
    FacilitationController,
    FacilitationInput,
    InterventionLogEntry,
    policy_for,
)
from ._intervention import (
    _NORMAL_SPECS,
    _as_bool,
    _build_candidates,
    _controller_barge_in_decision,
    _controller_normal_decision,
    _effective_silence,
    _intervention_enabled,
    _intervention_timing_metadata,
    _InterventionReviewRecorder,
    _is_backchannel,
    _legacy_decision_brief,
    _log_intervention_event,
    _log_voice_call_diag,
    _NormalTriggerDecision,
    _PendingInterventions,
    _recent_agent_texts,
    _set_manual_status,
    _suppressed_for,
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
        time.sleep(WORKER_TICK_SEC)
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
        self.pending = _PendingInterventions()
        # 採否Controller: 固定優先順位に代わり最終採否を担当する。物理タイミング
        # （floor/barge-in）と fact fast lane は維持しつつ、「どの候補を今採るか／
        # 黙るか」を一元裁定する。
        self.controller = FacilitationController()
        # af 生成先行・再生ゲート（フェーズ6, --af 有効時のみ作動。af 限定）。
        self.af_gate = _AfEarlyGenGate()
        self.af_held_text = ""
        self.af_held_kind = "af_l1"
        # 生成先行(hold)中の af が指名している相手。release 時に「誰を誘ったか」を
        # 記録するために保持する（保持しないと同じ人を連続で指名しうる）。
        self.af_held_target: str | None = None
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
        if agent is not None and agent._connected and agent.enabled:
            return True
        if agent is None or not agent.enabled:
            self.pending.clear_all()
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
        # --- 自動割り込み ---
        # 発話の存在は話者未確定でも確実なので raw スライスで判定する。
        # ファシリテーター/パートナーとも相槌は割り込みに使わない
        # (T7: 長めの相槌でファシリテーター発話がキャンセルされるのを防ぐ)。
        raw_texts = [str(r.get("text", "")) for r in raw_new]
        human_spoke = any(len(t.strip()) > _INTERRUPT_MIN_CHARS
                          and not _is_backchannel(t)
                          for t in raw_texts)
        if human_spoke and agent.ai_speaking:
            agent.interrupt()
        if partner is not None and (partner.ai_speaking or partner._responding):
            real_utterances = [t.strip() for t in raw_texts
                               if not _is_backchannel(t)]
            if real_utterances:
                partner.interrupt()
                for i, utt in enumerate(real_utterances):
                    is_last = (i == len(real_utterances) - 1)
                    partner.inject_context("人間", utt, request_response=is_last)
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
        """af の生成先行・再生ゲート（フェーズ6, --af 有効時のみ・af 限定）.

        AF が無効（既定）ならこのブロックは丸ごとスキップされ、ルールベースの
        採否・trigger 経路は一切変わらない（モード方針）。summarize には関与
        しない。WHEN は Controller に委譲し、ゲートは再生タイミングだけを担う。
        pending.af を消費するのは deliver/release/cancel の時点だけで、hold 中は
        保持し続ける（Controller の判定が pause 成立で "deliver" に変わるのを待つ）。
        """
        s = self.state
        agent = self.agent
        try:
            now = time.monotonic()
            # 同じ沈黙時間を4回計算していた（1tickに4回。判定は同じ）
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
                new_utterance=af_new_utt,
                agent_busy=bool(agent._responding or agent.ai_speaking),
                now=now,
                topics=af_topics,
            )
            if action == "trigger":     # 生成先行(hold)開始 — pending.af は保持
                held = payload or {}
                self.af_held_text = str(held.get("af_text") or "")
                self.af_held_kind = str(held.get("kind") or "af_l1")
                self.af_held_target = held.get("target_speaker") or None
                _log_intervention_event(s, self.af_held_kind, "af 生成先行(hold)")
            elif action == "deliver":   # 取り込み遅延で pause 通過後に来た → 即時配信
                held = payload or {}
                kind = str(held.get("kind") or "af_l1")
                text = str(held.get("af_text") or "")
                rt = s.af_runtime
                if rt is not None and text:
                    with contextlib.suppress(Exception):
                        rt.note_intervention(kind, text)
                self.pending.clear_af()   # 消費
                self.note_intervention(now, kind, "af 即時配信",
                                       invite_target=held.get("target_speaker"))
                _log_intervention_event(
                    s, kind, "af 即時配信",
                    timing=_intervention_timing_metadata(
                        kind=kind, now=now,
                        silence_elapsed=silence,
                        pause_required=policy_for(kind).pause,
                        policy="af_intervention"))
            elif action == "release":   # フロア成立 → 生成先行分を一斉再生
                rt = s.af_runtime
                if rt is not None and self.af_held_text:
                    with contextlib.suppress(Exception):
                        rt.note_intervention(self.af_held_kind, self.af_held_text)
                self.pending.clear_af()   # 消費
                self.note_intervention(now, self.af_held_kind, "af release",
                                       invite_target=self.af_held_target)
                _log_intervention_event(
                    s, self.af_held_kind, "af release",
                    timing=_intervention_timing_metadata(
                        kind=self.af_held_kind, now=now,
                        silence_elapsed=silence,
                        pause_required=policy_for(self.af_held_kind).pause,
                        policy="af_intervention",
                        hold_to_release_ms=self.af_gate.last_release_ms))
                self.af_held_text = ""
            elif action == "cancel":    # フロアが返らず/会話が動いた → 破棄
                self.pending.clear_af()   # 消費 (リトライにしない, B4)
                self.af_held_text = ""
                print("# [af] 生成先行を破棄 (フロア未成立/新規発話)", flush=True)
        except Exception as e:  # pragma: no cover - 防御的
            print(f"# [af] early-gen error: {e}", flush=True)

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
        elif decision.reason == "retry":
            legacy = _legacy_decision_brief("retry", "中断された介入を再送")
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
            print(f"# [trigger] fact: {correction}", flush=True)
            _log_intervention_event(s, "fact", correction, timing=timing)
            agent.trigger(topics=topics, fact_correction=decision.fact,
                          retry_intervention=False)
            self.pending.facts.popleft()
            self.note_intervention(time.monotonic(), "fact", correction)
            return True
        if decision.reason == "manual" and decision.manual is not None:
            manual = decision.manual
            request = str(manual.get("request") or "").strip()
            detail = request or "直近の議論整理"
            queued_payload = self.pending.manual_call or {}
            queued_at = float(queued_payload.get("created_at", now))
            timing = _intervention_timing_metadata(
                kind="manual", now=now, silence_elapsed=silence_elapsed,
                pause_required=policy_for("manual").pause,
                queued_at=queued_at,
                queued_wall_at=str(queued_payload.get("created_wall_at") or ""),
                policy="manual_call_pause")
            print(f"# [trigger] manual_call: {detail}", flush=True)
            _log_intervention_event(
                s, "manual_call", detail,
                timing={**timing, "source": manual.get("source", "ui"),
                        "request": request, "queued_at": queued_at,
                        "outcome": "selected"})
            _set_manual_status(s, "dispatched", detail=detail,
                               wait_sec=now - queued_at)
            agent.trigger(topics=topics, manual_request=manual)
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
            print(f"# [trigger] drift: 脱線介入「{decision.drift_reason}」", flush=True)
            _log_intervention_event(s, "drift", decision.drift_reason, timing=timing)
            agent.trigger(topics=topics, drift_reason=decision.drift_reason,
                          recent_agent_texts=_recent_agent_texts(s))
            self.pending.clear_drift()
            self.note_intervention(time.monotonic(), "drift", decision.drift_reason)
            return True
        if decision.reason == "retry":
            pending_intervention = agent._pending_intervention or {}
            timing = _intervention_timing_metadata(
                kind="retry", now=now, silence_elapsed=silence_elapsed,
                pause_required=policy_for("retry").pause,
                queued_at=float(pending_intervention.get("created_at", now)),
                policy="retry_extra_pause")
            print("# [trigger] retry: 中断された介入を再送（ガードバイパス）", flush=True)
            _log_intervention_event(s, "retry", "中断された介入を再送", timing=timing)
            agent.trigger(topics=topics, is_retry=True)
            self.note_intervention(time.monotonic(), "retry", "中断された介入を再送")
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
            print(f"# [trigger] {kind}: {shown}", flush=True)
        _log_intervention_event(self.state, kind, decision.detail, timing=timing)
        available = {
            "topics": topics,
            "summary_focus": decision.summary_focus,
            "invite_target": decision.invite_target,
            "af_presentation": decision.af_text,
            "recent_agent_texts": _recent_agent_texts(self.state),
        }
        self.agent.trigger(**{k: available[k] for k in spec.trigger})
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
            # --- ファシリテーター優先 ---
            if (partner is not None
                    and (partner.ai_speaking or partner._responding)
                    and self.agent.ai_speaking):
                partner.interrupt()
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
    """バックグラウンドでAI応答のトリガーを管理する（本体は `_AgentWorker`）."""
    _AgentWorker(state).run()


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

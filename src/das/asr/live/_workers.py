"""main()から抽出されたワーカー関数群.

ログ接頭辞の規約（Phase 3 R4）:
  # [state]   ... エージェントの状態遷移（RESPONDING/SPEAKING/INTERRUPTED/IDLE等）
  # [trigger] ... ファシリテーターのトリガー理由（drift/retry/count/silence/stall/skip）
  # [drift]   ... 並列ドリフト（脱線）検出の動作
  # [diag]    ... 定期的な状態ダンプ・スキップ理由などの診断
"""
from __future__ import annotations

import queue
import re
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ._session_state import SessionState
    from .stt import STTBackend

import contextlib

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
    _FACTCHECK_MIN_CHARS,
    _FACTCHECK_WINDOW,
    _INTERRUPT_MIN_CHARS,
    _INTERVENTION_COOLDOWN,
    _INVITE_CHECK_SEC,
    _INVITE_QUIET_RATIO,
    _INVITE_SILENCE,
    _INVITE_WARMUP,
    _STALL_COOLDOWN,
    _STALL_SILENCE,
    AGENT_SPEAKER,
    SR,
)
from ._participation import participation_stats
from ._speaker_policy import (
    intervention_records,
    intervention_speaker_name,
    reliable_human_records,
)
from ._ui import _print_line


_FACT_CANDIDATE_RE = re.compile(
    r"("
    r"\d|%|％|割|倍|cm|kg|m2|㎡"
    r"|BMI|bmi|式|計算|定義|単位|平均|中央値|割合|確率|速度|距離|面積|体積"
    r"|とは|っていうのは|というのは|イコール|割る|掛ける|足す|引く|二乗|2乗"
    r")"
)


def _looks_like_fact_claim(text: str) -> bool:
    """LLMに渡す前の軽い候補絞り。意見・相槌・短文を極力落とす."""
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
    return bool(_FACT_CANDIDATE_RE.search(s))


def _log_intervention_event(state: SessionState, reason: str, detail: str = "") -> None:
    add_event = getattr(state, "add_intervention_event", None)
    if callable(add_event):
        add_event(reason, detail)


def _intervention_enabled(state: SessionState) -> bool:
    return bool(getattr(state, "intervention_enabled", True))


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

    脱線や発話量とは別ルートにする。ローカルの候補絞りで「式・数値・定義っぽい」
    発話だけに限定し、LLM側でも high confidence の訂正だけを採用する。
    """
    from das.asr.live._bootstrap import check_fact_correction as _check_fact

    _last_check = 0.0
    while not state.stop.is_set():
        time.sleep(0.5)
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
        new_records = talk_rs[state.fact_cursor:n]
        if not any(_looks_like_fact_claim(str(r.get("text") or "")) for r in new_records):
            state.fact_cursor = n
            continue
        now = time.monotonic()
        if now - _last_check < _FACTCHECK_CHECK_SEC:
            continue
        _last_check = now
        state.fact_cursor = n
        window = talk_rs[max(0, n - _FACTCHECK_WINDOW):]
        utts = [{"speaker": intervention_speaker_name(state, r), "text": r["text"]}
                for r in window]
        result = _check_fact(utts, oai_key, oai_model)
        if result.get("should_correct"):
            correction = str(result.get("correction") or "").strip()
            if correction:
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
        if min(d["time_share"] for d in stats.values()) >= equal * _INVITE_QUIET_RATIO:
            continue
        _last_check = time.monotonic()
        now_ms = max((d["last_end_ms"] for d in stats.values()
                      if d["last_end_ms"] is not None), default=None)
        participation = []
        valid_invite_targets: set[str] = set()
        for sp, d in stats.items():
            silent = ((now_ms - d["last_end_ms"]) / 1000.0
                      if now_ms is not None and d["last_end_ms"] is not None else 0.0)
            speaker_name = state.disp_name(sp)
            valid_invite_targets.add(speaker_name)
            participation.append({"speaker": speaker_name,
                                  "time_share": d["time_share"],
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

        with state.state_lock:
            state.records.append({"ms": None, "end_ms": None,
                                  "speaker": AGENT_SPEAKER, "text": text.strip()})
            state.color_of(AGENT_SPEAKER)
        if ON_UTTERANCE is not None:
            with contextlib.suppress(Exception):
                ON_UTTERANCE("ファシリテーター", text.strip())
        _print_line(f"\x1b[96m[ファシリテーター]\x1b[0m: {text.strip()}")
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
    _last_stall_at = 0.0  # 沈黙ブレーカーの最終発火時刻（ループ防止、Fix 10）
    _last_intervention_at = 0.0  # 直近の介入時刻（脱線介入のクールダウン用）
    _pending_drift_reason: str | None = None  # drift_checkerからの未処理介入要求（R2）
    _pending_drift_count = 0
    _last_drift_request_at = 0.0
    _pending_fact: dict | None = None
    _last_fact_at = 0.0
    _pending_invite: str | None = None  # participation_checkerからの声かけ要求（S4）
    _last_invited: str | None = None    # 直近に声をかけた相手（連続回避）
    _last_agent_reconnect_at = 0.0
    while not state.stop.is_set():
        time.sleep(0.5)
        _diag_tick += 1
        partner = state.partner  # 動的参照: 実行中のパートナー接続/切断に追従（F3）
        if agent is None or not agent._connected or not agent.enabled:
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
                agent._last_noop_at = 0.0  # 新たな発話で会話が動いた → 沈黙ブレーカー解除
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
            for q in (state.drift_requests, state.invite_requests, state.factcheck_requests):
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
        while True:
            try:
                _pending_drift_reason = state.drift_requests.get_nowait()
                now = time.monotonic()
                if now - _last_drift_request_at > 20.0:
                    _pending_drift_count = 0
                _last_drift_request_at = now
                _pending_drift_count += 1
            except queue.Empty:
                break
        while True:
            try:
                _pending_invite = state.invite_requests.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                _pending_fact = state.factcheck_requests.get_nowait()
            except queue.Empty:
                break

        # --- 最優先のバージイン（ガードバイパス）:
        # ①事実補正 ②脱線介入 ③中断介入のリトライ ---
        # agentがfree(応答中でなく発話中でもない)になった瞬間に、エコーウィンドウ・
        # パートナー発話・沈黙閾値を無視してトリガーする。会話が活発でも取りこぼさない。
        # trigger()の呼び出しはこの _run_agent_worker に一元化されている（R2）。
        if not agent._responding and not agent.ai_speaking:
            _bargein_topics = None
            if agent.mode != "conversation":
                with state.topics_lock:
                    _bargein_topics = list(state.topics) if state.topics else None
            if _pending_fact is not None:
                correction = str(_pending_fact.get("correction") or "").strip()
                if not correction:
                    _pending_fact = None
                elif time.monotonic() - _last_fact_at < _FACTCHECK_COOLDOWN:
                    print("# [trigger] skip: クールダウン中の事実補正", flush=True)
                    _pending_fact = None
                else:
                    print(f"# [trigger] fact: {correction}", flush=True)
                    _log_intervention_event(state, "fact", correction)
                    agent.trigger(topics=_bargein_topics,
                                  fact_correction=_pending_fact)
                    _pending_fact = None
                    _last_fact_at = time.monotonic()
                    _last_intervention_at = _last_fact_at
                    continue
            if _pending_drift_reason is not None:
                _required_drift_count = int(state.proactivity.get("drift_confirmations", 1))
                if _pending_drift_count < _required_drift_count:
                    if _diag_tick % 20 == 0:
                        print(
                            "# [trigger] hold: 脱線判定の確認待ち "
                            f"{_pending_drift_count}/{_required_drift_count}",
                            flush=True,
                        )
                    continue
                # クールダウン中は連発を避けるため要求を破棄（再脱線なら再検出される）
                if time.monotonic() - _last_intervention_at < _cooldown:
                    print("# [trigger] skip: クールダウン中の脱線介入", flush=True)
                    _pending_drift_reason = None
                    _pending_drift_count = 0
                else:
                    print(f"# [trigger] drift: 脱線介入「{_pending_drift_reason}」",
                          flush=True)
                    _log_intervention_event(state, "drift", _pending_drift_reason)
                    agent.trigger(topics=_bargein_topics,
                                  drift_reason=_pending_drift_reason)
                    _pending_drift_reason = None
                    _pending_drift_count = 0
                    _last_intervention_at = time.monotonic()
                    continue
            if agent._pending_intervention is not None:
                print("# [trigger] retry: 中断された介入を再送（ガードバイパス）",
                      flush=True)
                _log_intervention_event(state, "retry", "中断された介入を再送")
                agent.trigger(topics=_bargein_topics)
                _last_intervention_at = time.monotonic()
                continue
        # エコーウィンドウ中はtriggerしない
        if agent is not None and agent.in_echo_window:
            _was_in_echo[0] = True
            continue
        # Partnerが発話中はtriggerしない
        if partner is not None and (partner.ai_speaking or partner._responding):
            _was_in_echo[0] = True
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
        if agent.mode == "conversation":
            if (agent.pending_count > 0
                    and _silence_elapsed > _AGENT_CONV_SILENCE):
                _log_intervention_event(state, "conversation", f"沈黙{_silence_elapsed:.1f}秒")
                agent.trigger()
        else:
            # 沈黙要約の閾値: debateは従来通り、人間モードは積極性プロファイルに従う
            # （None なら沈黙だけでは要約介入しない＝過剰介入の抑制, S5）
            _silence_thresh = (_AGENT_DEBATE_SILENCE if partner is not None
                               else _silence_summarize)
            if agent.pending_count >= agent.trigger_n:
                print(f"# [trigger] count: {agent.pending_count}>={agent.trigger_n}", flush=True)
                _log_intervention_event(
                    state, "count", f"{agent.pending_count}>={agent.trigger_n}発話")
                agent.trigger(topics=_topics)
                _last_intervention_at = time.monotonic()
            elif (_silence_thresh is not None
                  and agent.pending_count > 0
                  and _silence_elapsed > _silence_thresh):
                print(f"# [trigger] silence: {_silence_elapsed:.1f}s > {_silence_thresh}s", flush=True)
                _log_intervention_event(
                    state, "silence", f"{_silence_elapsed:.1f}>{_silence_thresh:.1f}秒")
                agent.trigger(topics=_topics)
                _last_intervention_at = time.monotonic()
            # --- 沈黙ブレーカー: 介入不要後にデッドエアになった場合の一押し（Fix 10） ---
            # 「介入不要」の判断自体は尊重する（一度黙る）が、その後に会話が止まって
            # しまったら、本題へ戻す一言を促す。クールダウンで繰り返しを防ぐ。
            elif (agent._last_noop_at > 0
                  and _silence_elapsed > _STALL_SILENCE
                  and time.monotonic() - _last_stall_at > _STALL_COOLDOWN):
                print(f"# [trigger] stall: 介入不要後の沈黙{_silence_elapsed:.1f}s"
                      f"を解消", flush=True)
                _log_intervention_event(state, "stall", f"介入不要後の沈黙{_silence_elapsed:.1f}秒")
                agent.trigger(
                    topics=_topics,
                    drift_reason="会話が止まっています。本題に戻す一言を簡潔に述べてください。")
                _last_stall_at = time.monotonic()
                _last_intervention_at = time.monotonic()
                agent._last_noop_at = 0.0
            # --- 声かけ（参加度）: 沈黙の“間”で、発言の少ない人を誘う（S4） ---
            # 脱線(バージイン)と違い、声かけは人間を割り込まないよう間を待つ。
            # クールダウン共有＋同じ人を連続では誘わない。
            elif (_pending_invite is not None
                  and _silence_elapsed > _INVITE_SILENCE
                  and time.monotonic() - _last_intervention_at > _cooldown):
                if _pending_invite == _last_invited:
                    _pending_invite = None  # 同じ人を連続では誘わない
                else:
                    print(f"# [trigger] invite: {_pending_invite}さんに声かけ", flush=True)
                    _log_intervention_event(state, "invite", f"{_pending_invite}さんに声かけ")
                    agent.trigger(topics=_topics, invite_target=_pending_invite)
                    _last_intervention_at = time.monotonic()
                    _last_invited = _pending_invite
                    _pending_invite = None


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
            _print_line("# 名前登録はブラウザUIを推奨。ターミナル操作: 「1=松井」/「fix 2=1」/ Ctrl+Cで終了")


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
        with state.buf_lock:
            state.pcm_buf.extend(pcm)
            state.pcm_total_bytes += len(pcm)
            if len(state.pcm_buf) > state._PCM_KEEP_BYTES + SR * 2 * 10:
                trim = len(state.pcm_buf) - state._PCM_KEEP_BYTES
                del state.pcm_buf[:trim]
                state.pcm_buf_offset += trim
        if state.pcm_file is not None:
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
    _print_line(f"# 議事録を保存しました: {state.out_path} / {state.html_path}")
    # WAVファイルのヘッダを確定して正規のWAVにする（state.wav_pathを使用）
    saved_wav = state.finalize_wav()
    if saved_wav:
        _print_line(f"# 録音を保存しました: {saved_wav}")

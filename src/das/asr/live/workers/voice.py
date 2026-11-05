"""常駐処理の責務: voice。会話の判定規則・時刻条件は元実装を維持する。"""
from __future__ import annotations

import contextlib
import queue
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._session_state import SessionState

from .._constants import (
    AGENT_SPEAKER,
)
from .._ui import _print_line
from .agent import _run_agent_worker


def _delivery_timing(state: SessionState) -> dict | None:
    """発話1件の観測値（§3.5）を介入ログ用にまとめる.

    LiveAgent.last_turn_stats（指示→最初の声の遅延、声の長さ、指示の有無、終端の
    理由）に、マイク座標系での再生区間 [capture_start_ms, capture_end_ms] を添える。
    replay はこの区間と参加者発話の ms を突き合わせて「人の発話が重なった時刻」と
    「モデルが黙るまでの時間」を出す。属性が無い agent（テストの偽物）でも安全。
    """
    agent = state.agent
    stats = getattr(agent, "last_turn_stats", None)
    if not isinstance(stats, dict):
        latency = getattr(agent, "_last_speak_latency_ms", None)
        return {"speak_start_latency_ms": latency} if latency is not None else None
    timing = dict(stats)
    opener = getattr(state, "ai_speech_open_start_ms", None)
    capture_now = getattr(state, "current_capture_ms", None)
    if callable(opener) and callable(capture_now):
        with contextlib.suppress(Exception):
            start_ms = opener("agent")
            if start_ms is None:
                # 終端マーカーが先に再生されて区間が閉じた後（stall 終端で起きやすい）。
                # 閉じた区間の最後の agent 分を使う
                start_ms = getattr(state, "last_ai_speech_start_ms", lambda _s: None)("agent")
            since = stats.get("since_last_voice_sec")
            end_ms = int(capture_now()) - int((since or 0.0) * 1000)
            if start_ms is not None:
                timing["capture_start_ms"] = int(start_ms)
                timing["capture_end_ms"] = max(int(start_ms), end_ms)
    return timing


def _on_agent_text_factory(state: SessionState):
    """ファシリテーター発言コールバックを生成."""
    def _on_agent_text(text: str, timing: dict | None = None):
        from das.asr.live import ON_UTTERANCE

        text = text.strip()
        # 声が鳴っていたマイク座標の区間を行の時刻にする（無ければ従来どおり --:--）。
        # 遡及訂正は speaker_source を持つ行しか触らず、席の回収・声かけの対象は
        # 話者名で除外しているので、AI 行に ms が付いても帰属処理には影響しない
        ms = end_ms = None
        if isinstance(timing, dict):
            ms = timing.get("capture_start_ms")
            end_ms = timing.get("capture_end_ms")
        with state.state_lock:
            state.records.append({"ms": ms, "end_ms": end_ms,
                                  "speaker": AGENT_SPEAKER, "text": text})
            state.color_of(AGENT_SPEAKER)
        if ON_UTTERANCE is not None:
            with contextlib.suppress(Exception):
                ON_UTTERANCE("ファシリテーター", text)
        state.add_facilitator_delivery_event(text, timing=timing)
        study = getattr(state, "study", None)
        if study is not None:
            study.log("voice_delivery", text=text, timing=timing)
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
            item = state.fac_events.get(timeout=0.5)
        except queue.Empty:
            continue
        kind, text = item[0], item[1]
        timing = item[2] if len(item) > 2 else None
        try:
            if kind == "utterance" and text is not None:
                on_text(text, timing)
                if "介入不要" not in text:
                    p = state.partner
                    if p is not None and p._connected:
                        # パートナーは進行役の声を室内の音声として聞いている。
                        # 文字は聞き取れなかったときの補いとして渡す（止めはしない）。
                        p.inject_context("ファシリテーター", text)
                    sim = state.simulator
                    if sim is not None:
                        sim.inject_facilitator(text)
            elif kind == "speech_start":
                pass   # パートナーは進行役の声を聞いて自分で譲る（全二重）
        except Exception as e:
            print(f"# ファシリテーターイベント処理エラー: {e}", flush=True)


def _connect_agent(state: SessionState, on_text):
    """ファシリテーターAgentのコールバック設定・接続・ワーカー起動."""
    agent = state.agent
    if agent is None:
        return
    # 受信スレッドはイベントを積むだけ。副作用は専用ワーカーで処理（受信ブロック回避）。
    # 観測値（§3.5）は受信スレッドの時点で写す。後から読むと次の発話に上書きされ、
    # 再生区間も終端マーカーで閉じられてしまう
    agent.on_ai_utterance = lambda text: state.fac_events.put(
        ("utterance", text, _delivery_timing(state)))

    def _agent_speech_start() -> None:
        # AI再生区間を開き（P2-1）、従来のイベント通知も行う。
        state.note_ai_speech_start("agent")
        state.fac_events.put(("speech_start", None))

    agent.on_speech_start = _agent_speech_start
    agent.on_speech_end = lambda: state.note_ai_speech_end("agent")
    threading.Thread(target=_run_facilitator_event_worker,
                     args=(state, on_text), daemon=True).start()

    agent.connect()
    if getattr(state, "study", None) is None:
        threading.Thread(target=_run_agent_worker, args=(state,), daemon=True).start()
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


def wire_partner_callbacks(state: SessionState, p) -> None:
    """相手役の発話・再生区間を議事録側へつなぐ（起動時と実行中の接続で共用）.

    再生区間（note_ai_speech_start/end）を記録しないと、Soniox 側のエコー除去が
    区間で判定できず、相手役の声が人の発言として残る／人の発言が消える。
    """
    p.on_ai_utterance = _on_partner_text_factory(state)
    p.on_speech_start = lambda: state.note_ai_speech_start("partner")
    p.on_speech_end = lambda: state.note_ai_speech_end("partner")


def _attach_partner(state: SessionState):
    """AIパートナーを生成・接続して state.partner にセットする（会話モード）."""
    if state.partner is not None:
        return  # 既に接続済み
    cfg = state._partner_cfg or {}
    if not cfg.get("api_key"):
        _print_line("# 会話モードにできません（OPENAI_API_KEYが未設定）")
        return
    from das.asr.live.agents._live import LivePartner
    p = LivePartner(api_key=cfg["api_key"],
                    voice=cfg.get("voice") or "cedar",
                    topic=cfg.get("topic") or "")
    if state.tracker is not None:
        p.set_tracker(state.tracker)
    wire_partner_callbacks(state, p)
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


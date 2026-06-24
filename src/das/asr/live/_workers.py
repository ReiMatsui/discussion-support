"""main()から抽出されたワーカー関数群."""
from __future__ import annotations

import os
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
    _AGENT_CONV_SILENCE,
    _AGENT_DEBATE_SILENCE,
    _AGENT_SILENCE,
    _BACKCHANNEL_RE,
    _DRIFT_CHECK_INTERVAL,
    _DRIFT_CHECK_WINDOW,
    _INTERRUPT_MIN_CHARS,
    AGENT_SPEAKER,
    SR,
)
from ._polish import polish
from ._ui import _print_line


def _run_topic_worker(state: SessionState, oai_key: str, oai_model: str):
    """論点抽出のバックグラウンドワーカー（モジュールレベル関数）."""
    from das.asr.live._bootstrap import extract_topics as _extract_topics

    while not state.stop.is_set():
        time.sleep(3)
        if not oai_key:
            continue
        with state.state_lock:
            talk_rs = [r for r in state.records if "speaker" in r and r.get("text")]
        n = len(talk_rs)
        if n - state.topic_cursor < state._TOPIC_TRIGGER:
            continue
        window = talk_rs[max(0, n - state._TOPIC_WINDOW):]
        utts = [{"speaker": state.disp_name(r["speaker"]), "text": r["text"]} for r in window]
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
    脱線検出時はファシリテーターに脱線理由を伝えて即時トリガーする。

    人間・パートナー双方の発話をチェック対象に含める。
    パートナーが脱線に付き合っている状態も検出するため。

    _pending_drift: 検出済みだがtriggerできなかった脱線理由。
    agentがbusy(応答中/発話中)の間は保持し、freeになった瞬間にtriggerする。
    これにより脱線検出結果が消失しない。
    """
    from das.asr.live._bootstrap import check_drift as _check_drift

    _diag_tick = 0
    _pending_drift: str | None = None  # 検出済み・未配信の脱線理由
    while not state.stop.is_set():
        time.sleep(1)  # 保留中のリトライを素早く行うため短めに
        _diag_tick += 1
        agent = state.agent
        if not oai_key or agent is None or not agent.enabled:
            continue
        if agent.mode == "conversation":
            continue

        # --- 保留中の脱線トリガーをリトライ ---
        if _pending_drift is not None:
            if not agent._responding and not agent.ai_speaking:
                with state.topics_lock:
                    _topics = list(state.topics) if state.topics else None
                print(f"# [drift] → 保留トリガーをリトライ: {_pending_drift}",
                      flush=True)
                agent.trigger(topics=_topics, drift_reason=_pending_drift)
                _pending_drift = None
            continue  # リトライ待ち中は新規チェックしない

        # （中断された介入のリトライは _run_agent_worker に集約。Bug 3）

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
            talk_rs = [r for r in state.records
                       if "speaker" in r and r.get("text")
                       and r.get("speaker") != AGENT_SPEAKER]
        n = len(talk_rs)
        if n - state.drift_cursor < _DRIFT_CHECK_INTERVAL:
            continue
        # 直近の発話を取得
        window = talk_rs[max(0, n - _DRIFT_CHECK_WINDOW):]
        utts = [{"speaker": state.disp_name(r["speaker"]), "text": r["text"]}
                for r in window]
        state.drift_cursor = n
        print(f"# [drift] チェック実行: {len(utts)}発話, "
              f"cursor={n}, topics={len(topics)}件", flush=True)
        # 脱線判定
        result = _check_drift(utts, topics, oai_key, oai_model)
        if result.get("drift"):
            reason = result.get("reason", "")
            _print_line(f"# 🔀 脱線検出: {reason}")
            if agent._responding or agent.ai_speaking:
                # agentがbusy → 保留して次のループで即リトライ
                _pending_drift = reason
                print("# [drift] トリガー保留（agentがbusy、1秒後リトライ）",
                      flush=True)
            else:
                with state.topics_lock:
                    _topics = list(state.topics) if state.topics else None
                print("# [drift] → ファシリテーターをトリガー", flush=True)
                agent.trigger(topics=_topics, drift_reason=reason)


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


def _connect_agent(state: SessionState, on_text):
    """ファシリテーターAgentのコールバック設定・接続・ワーカー起動."""
    agent = state.agent
    partner = state.partner
    simulator = state.simulator
    if simulator is not None:
        def _on_agent_with_sim(text: str):
            on_text(text)
            if "介入不要" not in text:
                simulator.inject_facilitator(text)
        agent.on_ai_utterance = _on_agent_with_sim
    elif partner is not None:
        def _on_agent_with_partner(text: str):
            on_text(text)
            if "介入不要" not in text and partner._connected:
                partner.interrupt()
                partner.inject_context("ファシリテーター", text)
        agent.on_ai_utterance = _on_agent_with_partner
        def _on_facilitator_speech_start():
            if partner._connected and (partner.ai_speaking or partner._responding):
                partner.interrupt()
        agent.on_speech_start = _on_facilitator_speech_start
    else:
        agent.on_ai_utterance = on_text
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


def _run_agent_worker(state: SessionState):
    """バックグラウンドでAI応答のトリガーを管理（ターンテイキング）.

    自然な会話のフロア交代を模倣:
      - 人間のターン: 発話を即座にfeed、沈黙で譲渡 → AIがtrigger
      - AIのターン: 応答を再生。人間の実質的な発話で自動interrupt
      - AIターン終了: フロアを人間に返す（沈黙タイマーをリセット）
    """
    agent = state.agent
    partner = state.partner
    _last_utt_time = state._last_utt_time
    _was_in_echo = state._was_in_echo
    _diag_tick = 0
    while not state.stop.is_set():
        time.sleep(0.5)
        _diag_tick += 1
        if agent is None or not agent._connected or not agent.enabled:
            if _diag_tick % 20 == 0:
                print(f"# [diag] _agent_worker skip: agent={agent is not None}"
                      f" conn={agent._connected if agent else '?'}"
                      f" enabled={agent.enabled if agent else '?'}", flush=True)
            continue
        with state.state_lock:
            _skip = {AGENT_SPEAKER, "パートナー"}
            talk_rs = [r for r in state.records
                       if "speaker" in r and r.get("text")
                       and r.get("speaker") not in _skip]
        n = len(talk_rs)
        if n > state.agent_cursor:
            _last_utt_time[0] = time.monotonic()
            new_texts = [r.get("text", "") for r in talk_rs[state.agent_cursor:]]
            for r in talk_rs[state.agent_cursor:]:
                agent.feed(state.disp_name(r.get("speaker", "")), r.get("text", ""))
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
        # --- ファシリテーター優先 ---
        if (partner is not None
                and (partner.ai_speaking or partner._responding)
                and agent is not None
                and agent.ai_speaking):
            partner.interrupt()
        # --- 中断された介入の最優先リトライ（ガードバイパス、Bug 3で集約） ---
        # agentがfree(応答中でなく発話中でもない)になった瞬間に、エコーウィンドウ・
        # パートナー発話・沈黙閾値を無視して再送する。会話が活発でも中断された介入を
        # 取りこぼさない。リトライ責務はこの1箇所に集約（drift_checker側は廃止）。
        if (agent._pending_intervention is not None
                and not agent._responding and not agent.ai_speaking):
            _retry_topics = None
            if agent.mode != "conversation":
                with state.topics_lock:
                    _retry_topics = list(state.topics) if state.topics else None
            print("# [diag] TRIGGER by retry: 中断された介入を再送（ガードバイパス）",
                  flush=True)
            agent.trigger(topics=_retry_topics)
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
                agent.trigger()
        else:
            _silence_thresh = (_AGENT_DEBATE_SILENCE if partner is not None
                               else _AGENT_SILENCE)
            if agent.pending_count >= agent.trigger_n:
                print(f"# [diag] TRIGGER by count: {agent.pending_count}>={agent.trigger_n}", flush=True)
                agent.trigger(topics=_topics)
            elif (agent.pending_count > 0
                  and _silence_elapsed > _silence_thresh):
                print(f"# [diag] TRIGGER by silence: {_silence_elapsed:.1f}s > {_silence_thresh}s", flush=True)
                agent.trigger(topics=_topics)


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
            _print_line("# コマンド: 「1=松井」(声を登録) / 「fix 2=1」「fix 人物2=人物1」(統合) / Ctrl+Cで終了")


def _run_from_mic(state: SessionState, device):
    """マイクからPCMを読み取り audio_q に送信."""
    import sounddevice as sd
    partner = state.partner
    agent = state.agent

    def cb(indata, frames, t, status):
        pcm = (np.clip(indata[:, 0], -1, 1) * 32767).astype("<i2").tobytes()
        state.audio_q.put(pcm)
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


def _run_sender(state: SessionState, ws, backend: STTBackend):
    """audio_qからPCMを読みWebSocketに送信 + PCMバッファ/ファイル書き出し."""
    seq = 0
    while True:
        pcm = state.audio_q.get()
        if pcm is None:
            end_msg = backend.make_end_message(seq)
            ws.send(end_msg)
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
        ws.send(pcm)
        seq += 1


def _cleanup(state: SessionState, args, api_key: str,
             tracker, wav_path: str, out_path: str, html_path: str):
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
    _print_line(f"# 議事録を保存しました: {out_path} / {html_path}")
    # WAVファイルのヘッダを更新して正規のWAVにする
    if state.pcm_file is not None:
        try:
            import struct as _struct
            state.pcm_file.flush()
            data_size = state.pcm_total_bytes
            state.pcm_file.seek(4)
            state.pcm_file.write(_struct.pack("<I", 36 + data_size))
            state.pcm_file.seek(40)
            state.pcm_file.write(_struct.pack("<I", data_size))
            state.pcm_file.close()
            if state.pcm_total_bytes > SR * 2 * 10:
                _print_line(f"# 録音を保存しました: {wav_path}")
            else:
                os.remove(wav_path)
        except OSError as e:
            _print_line(f"# 録音保存に失敗: {e}")
        state.pcm_file = None
    # 清書
    if args.polish and not api_key and state.pcm_total_bytes > SR * 2 * 10:
        _print_line("# 清書はスキップ（SONIOX_API_KEY未設定。清書はSoniox非同期APIを使用）")
    if args.polish and api_key and state.pcm_total_bytes > SR * 2 * 10:
        try:
            with open(wav_path, "rb") as f:
                wav_data = f.read()
            recs = polish(api_key, wav_data[44:], args.lang, tracker, log=_print_line)
            fmd = os.path.splitext(out_path)[0] + ".final.md"
            fht = os.path.splitext(out_path)[0] + ".final.html"
            state.write_md(recs, fmd)
            state.write_html(live=False, recs=recs, path=fht, status="清書（非同期再処理済み）")
            state.write_turns(recs, os.path.splitext(out_path)[0] + ".final.turns.jsonl")
            _print_line(f"# 清書版を保存しました: {fmd} / {fht}")
            if not args.no_open:
                import webbrowser
                webbrowser.open("file://" + os.path.abspath(fht))
        except KeyboardInterrupt:
            _print_line("# 清書をスキップしました")
        except Exception as e:
            _print_line(f"# 清書に失敗しました: {type(e).__name__}: {e}")

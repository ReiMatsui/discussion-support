"""常駐処理の責務: triage。会話の判定規則・時刻条件は元実装を維持する。"""
from __future__ import annotations

import contextlib
import datetime
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .._session_state import SessionState

from .._constants import (
    _ACK_CHIME_ENABLED,
    _MANUAL_CALL_MAX_CHARS,
    _TRIAGE_BACKLOG_MAX,
    _TRIAGE_CONTEXT_WINDOW,
    _TRIAGE_MAX_RETRIES,
    _TRIAGE_MIN_CHARS,
    AGENT_SPEAKER,
    WORKER_TICK_SEC,
)
from .._intervention import (
    _as_bool,
    _intervention_enabled,
    _log_voice_call_diag,
    _set_manual_status,
)
from .._speaker_policy import (
    intervention_speaker_name,
    triage_records,
)
from .common import _resilient


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
        # 共有のスピーカー出力（24kHz, 1本）に載せる。別にストリームを開くと
        # 進行役の声と同時に鳴ったときに音が粗くなる
        with contextlib.suppress(Exception):
            from .. import _audio_out
            from ..agents._jitter import JitterBuffer
            dur = 0.15
            rate = _audio_out.RATE
            t = np.linspace(0, dur, int(rate * dur), endpoint=False)
            envelope = np.exp(-t * 12.0)                 # なめらかな減衰
            wave = 0.2 * envelope * np.sin(2 * np.pi * 880.0 * t)
            src = JitterBuffer(target_ms=0, min_ms=0)
            src.push((wave * 32767).astype("<i2").tobytes())
            _audio_out.register(src)
            time.sleep(dur + 0.3)
            _audio_out.unregister(src)

    threading.Thread(target=_play, daemon=True).start()


def _triage_classify_one(state, talk_rs, idx, r, text, *, retry_counts,
                         classify_fn, oai_key, oai_model) -> dict | None:
    """発話1件の triage 注釈を作る。一時失敗で再試行すべきなら None を返す."""
    if len(text) < _TRIAGE_MIN_CHARS:
        # 相槌は triage_records が除外済み。残る機械的ゲートは
        # 「極端に短い発話」のみ（LLM を呼ぶ価値がない, コスト0）。
        return {"factual_claim": False, "facilitator_request": ""}
    context = talk_rs[max(0, idx - _TRIAGE_CONTEXT_WINDOW):idx]
    utts = [{"speaker": intervention_speaker_name(state, c),
             "text": c["text"]} for c in context]
    utts.append({"speaker": intervention_speaker_name(state, r),
                 "text": text})
    result = classify_fn(utts, oai_key, oai_model)
    if result.get("retryable_error"):
        tries = retry_counts.get(idx, 0) + 1
        retry_counts[idx] = tries
        if tries <= _TRIAGE_MAX_RETRIES:
            print(f"# [triage] retry: LLM分類の一時失敗 "
                  f"{tries}/{_TRIAGE_MAX_RETRIES}", flush=True)
            return None   # 同一発話を次tickで再試行（カーソルは進めない）
        print("# [triage] skip: LLM分類の失敗が続いたため対象発話を"
              "スキップ", flush=True)
        return {"factual_claim": False, "facilitator_request": ""}
    return {
        "factual_claim": _as_bool(result.get("factual_claim")),
        "facilitator_request": str(
            result.get("facilitator_request") or ""
        ).strip()[:_MANUAL_CALL_MAX_CHARS],
    }


def _answered_directly(agent, queued_at: float, *, window_sec: float = 8.0) -> bool:
    """呼びかけの前後 window_sec 秒以内に、指示なしで話し始めていたか.

    呼びかけの音声は STT 確定と triage を経て queued_at になるので、GPT-Live の
    直接応答は queued_at の数秒前に始まっていることが多い。
    """
    at = float(getattr(agent, "last_unrequested_speech_at", 0.0) or 0.0)
    return at > 0.0 and at >= queued_at - window_sec


def _dispatch_facilitator_voice_call(state, text: str, request: str) -> None:
    """検出した呼びかけを手動呼び出しキューへ積む（UIボタンと同じ経路）."""
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
    # GPT-Live がすでに声で応じていれば、音は重なるだけなので鳴らさない
    if not _answered_directly(getattr(state, "agent", None), time.monotonic()):
        _play_ack_chime()


@_resilient
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
            annotation = _triage_classify_one(
                state, talk_rs, idx, r, text, retry_counts=_retry_counts,
                classify_fn=_classify, oai_key=oai_key, oai_model=oai_model)
            if annotation is None:
                break                     # 一時失敗 → 次tickで同じ発話を再試行
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
                _dispatch_facilitator_voice_call(state, text, request)


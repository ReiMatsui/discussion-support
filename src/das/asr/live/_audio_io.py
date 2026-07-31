"""音声の取り込みと送出（マイク / WAV 擬似ライブ / STTへの送信）.

`_workers.py` から切り出した。ここにあるのは「音を運ぶ」責任だけで、
話者の判定にも介入の判断にも関わらない。切り離しておくと、帰属の調査で
`_workers.py` を読むときにこの層を読み飛ばせる。

流れ:

    _run_from_mic / _run_from_wav  ──(audio_q)──>  _run_sender  ──> STT
                                                       ├─> asr_pcm_buf（帰属が切る音）
                                                       └─> 録音wav
"""
from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ._session_state import SessionState
    from .stt import STTBackend

from ._constants import SR


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


def _load_wav_mono_16k(path: str) -> np.ndarray:
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
        n_out = round(len(y) * SR / sr)
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
        if i - step >= len(y):
            break
        state.audio_q.put((chunk * 32767).astype("<i2").tobytes())
        time.sleep(0.12)
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
        state.note_send_backlog()
        with state.buf_lock:
            state.pcm_buf.extend(pcm)
            if not setup_capture_only:
                state.pcm_total_bytes += len(pcm)
            if len(state.pcm_buf) > state._PCM_KEEP_BYTES + SR * 2 * 10:
                trim = len(state.pcm_buf) - state._PCM_KEEP_BYTES
                del state.pcm_buf[:trim]
        if ws is not None:
            try:
                ws.send(pcm)
            except Exception:
                pass
            else:
                # wav への書き込みは buf_lock の中で行う。「新しい会議」の
                # finalize_wav / open_wav（同じロックを取る）と競合すると、
                # 閉じたファイルへの write が ValueError で送信スレッドを
                # 殺し、以後の文字起こしが無言で全停止する（レビュー
                # 2026-07-30）。閉じたファイルは OSError ではなく ValueError
                # を投げるので、捕捉の型も広げる。
                with state.buf_lock:
                    if state.pcm_file is not None:
                        try:
                            state.pcm_file.write(pcm)
                            state.pcm_file.flush()
                        except (OSError, ValueError):
                            pass
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


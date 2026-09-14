"""スピーカー出力の一本化（進行役・相手役・シミュレータの声を 1 本のストリームで混ぜる）.

これまでは進行役（24kHz）とシミュレータの参加者（16kHz）が別々の OutputStream を
同じデバイスに開いていた。macOS の PortAudio はストリームを開くたびにデバイスの
サンプルレートを合わせようとするので、レートの違う 2 本が同時に鳴ると片方が
変換越しの粗い音（がびがび）になる。ここでは 24kHz のストリームを 1 本だけ開き、
音源ごとのバッファをコールバックで足し合わせる。

- 音源は JitterBuffer（pull(nbytes) で PCM16 を返すもの）。register/unregister で出入り。
- コールバックは Python で動くので、他スレッドが GIL を長く握ると遅れる。ブロックを
  100ms にし、PortAudio 側の余裕（latency="high"）を取って耐性を上げる。
- デバイスが開けない環境（音声なしのサーバ等）では、実時間で取り出して捨てる
  スレッドが代わりに回る。溜まったまま「話し中」が戻らない事故を防ぐ。
"""
from __future__ import annotations

import contextlib
import threading
import time

import numpy as np

RATE = 24000
_BLOCK = 2400            # 100ms
_lock = threading.Lock()
_sources: list = []
_started = False
_stream = None
_status_warned = False


def register(source) -> None:
    """音源を追加する。初回でストリームを開く."""
    global _started
    with _lock:
        if source not in _sources:
            _sources.append(source)
        if not _started:
            _started = True
            _open()


def unregister(source) -> None:
    with _lock, contextlib.suppress(ValueError):
        _sources.remove(source)


def _mix(nbytes: int) -> np.ndarray:
    with _lock:
        srcs = list(_sources)
    out = np.zeros(nbytes // 2, dtype=np.float32)
    for src in srcs:
        pcm = src.pull(nbytes)
        out += np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    return np.clip(out, -1.0, 1.0)


def _open() -> None:
    global _stream
    try:
        import sounddevice as sd

        def _cb(outdata, frames, _t, status):
            global _status_warned
            if status and not _status_warned:
                _status_warned = True
                print(f"# スピーカー出力: {status}", flush=True)
            outdata[:, 0] = _mix(frames * 2)

        _stream = sd.OutputStream(samplerate=RATE, channels=1, dtype="float32",
                                  blocksize=_BLOCK, latency="high", callback=_cb)
        _stream.start()
    except Exception as e:
        print(f"# スピーカー出力を開けません（音は出さずに進みます）: {e}", flush=True)
        _stream = None
        threading.Thread(target=_drain_without_device, daemon=True).start()


def _drain_without_device() -> None:
    """デバイスなし: 実時間で取り出して捨てる（再生の時間感覚だけ保つ）."""
    period = _BLOCK / RATE
    next_at = time.monotonic()
    while True:
        next_at += period
        _mix(_BLOCK * 2)
        time.sleep(max(0.0, next_at - time.monotonic()))


def close() -> None:
    global _stream, _started
    with _lock:
        st, _stream = _stream, None
        _started = False
    if st is not None:
        with contextlib.suppress(Exception):
            st.stop()
            st.close()

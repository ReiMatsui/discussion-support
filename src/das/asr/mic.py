"""sounddevice からマイク音声を取り込み、PCM s16le 16kHz mono を流す。

WhisperLiveKit の ``AudioProcessor`` は FFmpeg 経由でも PCM 直接でも受
けられるが、FFmpeg を介さない方がデプロイが楽 (ffmpeg バイナリへの依存
が消える)。本モジュールはマイクから取った PCM を生のまま投入する。

設計判断:
  - sounddevice のコールバックは別スレッドで呼ばれるので、
    ``loop.call_soon_threadsafe`` で asyncio Queue にバトンを渡す
  - ブロックサイズは 100ms (1600 サンプル) を既定。WhisperLiveKit の
    バッファとの相性を見て将来調整可能
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from das.logging import get_logger

_log = get_logger("das.asr.mic")


SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"


async def iter_mic_chunks(
    *,
    block_size: int = 1600,  # 100ms @ 16kHz
    sample_rate: int = SAMPLE_RATE,
    stop_event: asyncio.Event | None = None,
    queue_maxsize: int = 128,
) -> AsyncIterator[bytes]:
    """マイクから PCM s16le bytes を非同期 yield する。

    ``stop_event.set()`` で停止する (Ctrl-C ハンドラから呼ぶ想定)。
    """

    # [asr] extras 経由の遅延 import
    import sounddevice as sd

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_maxsize)
    stop = stop_event or asyncio.Event()

    def _callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status:
            _log.warning("mic.status", status=str(status))
        try:
            loop.call_soon_threadsafe(queue.put_nowait, bytes(indata))
        except asyncio.QueueFull:
            _log.warning("mic.queue_full", qsize=queue.qsize())

    stream = sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=block_size,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=_callback,
    )

    _log.info("mic.start", sample_rate=sample_rate, block_size=block_size)
    with stream:
        while not stop.is_set():
            try:
                chunk = await asyncio.wait_for(queue.get(), timeout=0.5)
            except TimeoutError:
                continue
            yield chunk
    _log.info("mic.stop")


__all__ = ["SAMPLE_RATE", "iter_mic_chunks"]

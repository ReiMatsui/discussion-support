import asyncio
from collections.abc import AsyncIterator


class MicrophoneCaptureError(RuntimeError):
    pass


async def microphone_pcm_frames(
    *,
    sample_rate: int,
    channels: int,
    frame_ms: int,
    device: int | str | None = None,
) -> AsyncIterator[bytes]:
    """Yield raw PCM S16LE frames from the local microphone."""

    try:
        import sounddevice as sd
    except ImportError as exc:
        raise MicrophoneCaptureError(
            "Microphone capture requires 'sounddevice'."
        ) from exc

    if frame_ms <= 0:
        raise MicrophoneCaptureError("frame_ms must be greater than zero")

    blocksize = int(sample_rate * frame_ms / 1000)
    if blocksize <= 0:
        raise MicrophoneCaptureError("audio frame size is too small")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=20)

    def put_frame(frame: bytes) -> None:
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(frame)

    def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            pass
        loop.call_soon_threadsafe(put_frame, bytes(indata))

    try:
        with sd.RawInputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="int16",
            blocksize=blocksize,
            device=device,
            callback=callback,
        ):
            while True:
                yield await queue.get()
    except Exception as exc:
        raise MicrophoneCaptureError("Microphone capture failed") from exc

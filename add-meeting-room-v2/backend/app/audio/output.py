from __future__ import annotations


class AudioOutputError(RuntimeError):
    pass


class Pcm16AudioPlayer:
    def __init__(self, *, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self._stream = None
        self._aborted = False

    def __enter__(self) -> Pcm16AudioPlayer:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioOutputError("Audio playback requires sounddevice.") from exc

        self._stream = sd.RawOutputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            blocksize=0,
        )
        self._stream.start()
        self._aborted = False
        return self

    def __exit__(self, *args: object) -> None:
        if self._stream is None:
            return
        if not self._aborted:
            self._stream.stop()
        self._stream.close()
        self._stream = None

    def write(self, pcm: bytes) -> None:
        if not pcm or self._stream is None:
            return
        self._stream.write(pcm)

    def abort(self) -> None:
        if self._stream is None:
            return
        self._aborted = True
        self._stream.abort()
        self._stream.close()
        self._stream = None

"""pyannoteAI streaming diarization provider."""
from __future__ import annotations

import contextlib
import json
import queue
import threading
import urllib.request
from typing import Any

import numpy as np

from ._constants import SR
from ._diarization import DiarizationEvent


class PyannoteStreamingDiarizationProvider:
    """pyannoteAI のリアルタイム話者分離 WebSocket provider.

    入力側の共通形式は既存のライブ処理に合わせて 16kHz PCM16 bytes とし、
    pyannoteAI が要求する 16kHz mono float32 little-endian に内部変換して送る。
    """

    _CREATE_URL = "https://api.pyannote.ai/v1/live"

    def __init__(self, api_key: str, *, create_url: str | None = None) -> None:
        self.api_key = api_key
        self.create_url = create_url or self._CREATE_URL
        self._ws: Any = None
        self._events: queue.Queue[DiarizationEvent] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()
        self._active_starts: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "pyannote"

    def start(self) -> None:
        from websockets.sync.client import connect

        self._stop.clear()
        self._active_starts.clear()
        req = urllib.request.Request(self.create_url, data=b"{}", method="POST")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read())
        url = payload["url"]
        self._ws = connect(url)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def send_audio(self, pcm16k: bytes) -> None:
        if self._ws is None:
            return
        payload = pcm16_to_pyannote_f32(pcm16k)
        if not payload:
            return
        self._ws.send(payload)

    def drain_events(self) -> list[DiarizationEvent]:
        events: list[DiarizationEvent] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def active_events(self) -> list[DiarizationEvent]:
        return [
            DiarizationEvent(start_ms, None, speaker, self.name)
            for speaker, start_ms in self._active_starts.items()
        ]

    def close(self) -> None:
        self._stop.set()
        if self._ws is not None:
            with contextlib.suppress(Exception):
                self._ws.send(json.dumps({"type": "end_of_stream"}))
            with contextlib.suppress(Exception):
                self._ws.close()
        if self._reader is not None:
            self._reader.join(timeout=1.0)

    def _read_loop(self) -> None:
        while not self._stop.is_set() and self._ws is not None:
            try:
                raw = self._ws.recv()
            except Exception:
                break
            event = self._parse_message(raw)
            if event is not None:
                self._events.put(event)

    def _parse_message(self, raw: str | bytes) -> DiarizationEvent | None:
        msg = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        typ = msg.get("type")
        if typ not in {"diarization_speaker_start", "diarization_speaker_end"}:
            return None
        data = msg.get("data") or {}
        speaker = data.get("speaker")
        timestamp = data.get("timestamp")
        if not isinstance(speaker, str) or not isinstance(timestamp, int | float):
            return None
        ms = int(float(timestamp) * 1000)
        if typ == "diarization_speaker_start":
            self._active_starts[speaker] = ms
            return None
        start_ms = self._active_starts.pop(speaker, ms)
        return DiarizationEvent(
            start_ms=start_ms,
            end_ms=ms,
            speaker=speaker,
            source=self.name,
        )


def pcm16_to_pyannote_f32(pcm16k: bytes) -> bytes:
    """テストしやすいPCM16→pyannote入力形式変換."""
    samples = np.frombuffer(pcm16k, dtype="<i2").astype(np.float32) / 32768.0
    if SR != 16000:
        raise ValueError("pyannote streaming provider expects 16kHz audio")
    return samples.astype("<f4").tobytes()

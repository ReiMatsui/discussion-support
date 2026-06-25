"""AssemblyAI streaming diarization provider."""
from __future__ import annotations

import contextlib
import json
import queue
import threading
from typing import Any
from urllib.parse import urlencode

from ._constants import SR
from ._diarization import DiarizationEvent


class AssemblyAIStreamingDiarizationProvider:
    """AssemblyAI Streaming API の話者分離だけを外部 provider として使う.

    Soniox の文字起こし品質を残しつつ、AssemblyAI の speaker_labels/max_speakers を
    resolver に渡すための薄い adapter。
    """

    _WS_URL = "wss://streaming.assemblyai.com/v3/ws"

    def __init__(
        self,
        api_key: str,
        *,
        max_speakers: int | None = None,
        speech_model: str = "universal-3-5-pro",
        ws_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.max_speakers = max_speakers
        self.speech_model = speech_model
        self.ws_url = ws_url or self._WS_URL
        self._ws: Any = None
        self._events: queue.Queue[DiarizationEvent] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()
        self._active_by_turn: dict[int, DiarizationEvent] = {}

    @property
    def name(self) -> str:
        return "assemblyai"

    def start(self) -> None:
        from websockets.sync.client import connect

        self._stop.clear()
        self._active_by_turn.clear()
        params: dict[str, str | int] = {
            "sample_rate": SR,
            "encoding": "pcm_s16le",
            "speech_model": self.speech_model,
            "speaker_labels": "true",
            "format_turns": "true",
        }
        if self.max_speakers is not None:
            params["max_speakers"] = self.max_speakers
        url = f"{self.ws_url}?{urlencode(params)}"
        self._ws = connect(url, additional_headers={"Authorization": self.api_key})
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def send_audio(self, pcm16k: bytes) -> None:
        if self._ws is not None and pcm16k:
            self._ws.send(pcm16k)

    def set_max_speakers(self, max_speakers: int | None) -> None:
        self.max_speakers = max_speakers

    def drain_events(self) -> list[DiarizationEvent]:
        events: list[DiarizationEvent] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def active_events(self) -> list[DiarizationEvent]:
        return list(self._active_by_turn.values())

    def close(self) -> None:
        self._stop.set()
        if self._ws is not None:
            with contextlib.suppress(Exception):
                self._ws.send(json.dumps({"type": "Terminate"}))
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
            for event in self._parse_message(raw):
                self._events.put(event)

    def _parse_message(self, raw: str | bytes) -> list[DiarizationEvent]:
        msg = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        typ = msg.get("type")
        if typ == "Turn":
            return self._parse_turn(msg)
        if typ == "SpeakerRevision":
            events: list[DiarizationEvent] = []
            for revision in msg.get("revisions") or []:
                events.extend(self._events_from_words(revision.get("words") or []))
            return events
        return []

    def _parse_turn(self, msg: dict[str, Any]) -> list[DiarizationEvent]:
        turn_order = msg.get("turn_order")
        speaker = _clean_speaker(msg.get("speaker_label"))
        words = msg.get("words") or []
        events = self._events_from_words(words)
        if not events and speaker is not None:
            events = self._event_from_turn_label(words, speaker)

        if isinstance(turn_order, int):
            if msg.get("end_of_turn"):
                self._active_by_turn.pop(turn_order, None)
            elif events:
                self._active_by_turn[turn_order] = events[-1]

        if not msg.get("end_of_turn"):
            return []
        return events

    def _events_from_words(self, words: list[dict[str, Any]]) -> list[DiarizationEvent]:
        events: list[DiarizationEvent] = []
        cur_speaker: str | None = None
        cur_start: int | None = None
        cur_end: int | None = None
        for word in words:
            if word.get("word_is_final") is False:
                continue
            speaker = _clean_speaker(word.get("speaker"))
            start = _as_int(word.get("start"))
            end = _as_int(word.get("end"))
            if speaker is None or start is None or end is None or end <= start:
                continue
            if speaker != cur_speaker and cur_speaker is not None and cur_start is not None:
                events.append(DiarizationEvent(cur_start, cur_end, cur_speaker, self.name))
                cur_start = start
            elif cur_start is None:
                cur_start = start
            cur_speaker = speaker
            cur_end = end
        if cur_speaker is not None and cur_start is not None and cur_end is not None:
            events.append(DiarizationEvent(cur_start, cur_end, cur_speaker, self.name))
        return events

    def _event_from_turn_label(
        self,
        words: list[dict[str, Any]],
        speaker: str,
    ) -> list[DiarizationEvent]:
        starts = [_as_int(w.get("start")) for w in words]
        ends = [_as_int(w.get("end")) for w in words]
        starts = [v for v in starts if v is not None]
        ends = [v for v in ends if v is not None]
        if not starts or not ends:
            return []
        start = min(starts)
        end = max(ends)
        if end <= start:
            return []
        return [DiarizationEvent(start, end, speaker, self.name)]


def _clean_speaker(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value == "UNKNOWN":
        return None
    return value


def _as_int(value: object) -> int | None:
    if isinstance(value, int | float):
        return int(value)
    return None

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from app.core.config import Settings


def current_timestamp_ms() -> int:
    return time.time_ns() // 1_000_000


@dataclass(frozen=True, slots=True)
class TranscriptEvent:
    type: Literal["transcript.delta", "transcript.final"]
    meeting_id: str
    segment_id: str
    speaker_label: str | None
    text: str
    is_final: bool
    endpoint_detected: bool
    start_ms: int | None
    end_ms: int | None
    server_timestamp_ms: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "meeting_id": self.meeting_id,
            "segment_id": self.segment_id,
            "speaker_label": self.speaker_label,
            "text": self.text,
            "is_final": self.is_final,
            "endpoint_detected": self.endpoint_detected,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "server_timestamp_ms": self.server_timestamp_ms,
        }


@dataclass(frozen=True, slots=True)
class TurnEndedEvent:
    type: Literal["turn.ended"]
    meeting_id: str
    endpoint_detected: bool
    server_timestamp_ms: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "meeting_id": self.meeting_id,
            "endpoint_detected": self.endpoint_detected,
            "server_timestamp_ms": self.server_timestamp_ms,
        }


SonioxEvent = TranscriptEvent | TurnEndedEvent


class SonioxClientError(RuntimeError):
    pass


class SonioxRealtimeClient:
    def __init__(self, *, settings: Settings) -> None:
        self._settings = settings
        self._segment_number = 0
        self._active_segment_id: str | None = None

    async def transcribe(
        self,
        audio_frames: AsyncIterator[bytes],
    ) -> AsyncIterator[SonioxEvent]:
        if not self._settings.soniox_api_key:
            raise SonioxClientError("SONIOX_API_KEY is not set")

        try:
            import websockets
        except ImportError as exc:
            raise SonioxClientError(
                "Soniox realtime transcription requires 'websockets'."
            ) from exc

        async with websockets.connect(self._settings.soniox_websocket_url) as websocket:
            await websocket.send(json.dumps(self._start_request()))

            send_task = asyncio.create_task(self._send_audio(websocket, audio_frames))
            try:
                async for raw_message in websocket:
                    response = self._parse_response(raw_message)
                    if response.get("finished") is True:
                        break
                    self._raise_for_error(response)
                    event = self._response_to_event(response)
                    if event is not None:
                        yield event
            finally:
                send_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass

    def _start_request(self) -> dict[str, Any]:
        return {
            "api_key": self._settings.soniox_api_key,
            "model": self._settings.soniox_model,
            "audio_format": self._settings.audio_format,
            "sample_rate": self._settings.audio_sample_rate,
            "num_channels": self._settings.audio_channels,
            "language_hints": self._settings.language_hints_list(),
            "enable_speaker_diarization": (
                self._settings.soniox_enable_speaker_diarization
            ),
            "enable_endpoint_detection": (
                self._settings.soniox_enable_endpoint_detection
            ),
            "max_endpoint_delay_ms": self._settings.soniox_max_endpoint_delay_ms,
            "client_reference_id": self._settings.meeting_id,
        }

    async def _send_audio(
        self,
        websocket: Any,
        audio_frames: AsyncIterator[bytes],
    ) -> None:
        async for frame in audio_frames:
            await websocket.send(frame)
        await websocket.send(b"")

    def _parse_response(self, raw_message: str | bytes) -> dict[str, Any]:
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8")
        try:
            response = json.loads(raw_message)
        except json.JSONDecodeError as exc:
            raise SonioxClientError("Soniox returned non-JSON response") from exc
        if not isinstance(response, dict):
            raise SonioxClientError("Soniox returned invalid response")
        return response

    def _raise_for_error(self, response: dict[str, Any]) -> None:
        if "error_code" not in response:
            return
        error_type = response.get("error_type") or "unknown_error"
        error_message = response.get("error_message") or "Soniox returned an error"
        raise SonioxClientError(f"{error_type}: {error_message}")

    def _response_to_event(self, response: dict[str, Any]) -> SonioxEvent | None:
        tokens = response.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            return None

        valid_tokens = [token for token in tokens if isinstance(token, dict)]
        if not valid_tokens:
            return None

        endpoint_detected = any(
            token.get("text") == "<end>" for token in valid_tokens
        )
        transcript_tokens = [
            token for token in valid_tokens if token.get("text") != "<end>"
        ]
        if not transcript_tokens and endpoint_detected:
            return TurnEndedEvent(
                type="turn.ended",
                meeting_id=self._settings.meeting_id,
                endpoint_detected=True,
                server_timestamp_ms=current_timestamp_ms(),
            )

        text = "".join(
            token.get("text", "")
            for token in transcript_tokens
            if isinstance(token.get("text"), str)
        )
        if not text:
            return None

        is_final = all(token.get("is_final") is True for token in valid_tokens)
        speaker_label = self._speaker_label(transcript_tokens)
        start_ms = self._min_ms(transcript_tokens, "start_ms")
        end_ms = self._max_ms(transcript_tokens, "end_ms")

        return TranscriptEvent(
            type="transcript.final" if is_final else "transcript.delta",
            meeting_id=self._settings.meeting_id,
            segment_id=self._segment_id(is_final=is_final),
            speaker_label=speaker_label,
            text=text,
            is_final=is_final,
            endpoint_detected=endpoint_detected,
            start_ms=start_ms,
            end_ms=end_ms,
            server_timestamp_ms=current_timestamp_ms(),
        )

    def _segment_id(self, *, is_final: bool) -> str:
        if self._active_segment_id is None:
            self._segment_number += 1
            self._active_segment_id = f"seg_{self._segment_number:06d}"
        segment_id = self._active_segment_id
        if is_final:
            self._active_segment_id = None
        return segment_id

    @staticmethod
    def _speaker_label(tokens: list[dict[str, Any]]) -> str | None:
        for token in tokens:
            speaker = token.get("speaker")
            if speaker is not None:
                return str(speaker)
        return None

    @staticmethod
    def _min_ms(tokens: list[dict[str, Any]], key: str) -> int | None:
        values = [token.get(key) for token in tokens if isinstance(token.get(key), int)]
        return min(values) if values else None

    @staticmethod
    def _max_ms(tokens: list[dict[str, Any]], key: str) -> int | None:
        values = [token.get(key) for token in tokens if isinstance(token.get(key), int)]
        return max(values) if values else None

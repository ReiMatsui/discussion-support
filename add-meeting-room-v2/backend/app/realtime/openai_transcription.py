import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import websockets
from websockets.asyncio.client import ClientConnection

from app.core.config import Settings, get_settings
from app.realtime.schemas import (
    AudioChunkMessage,
    DeviceJoinMessage,
    TranscriptDeltaEvent,
    TranscriptFinalEvent,
    TranscriptionConnectedEvent,
    TranscriptionDisconnectedEvent,
    TranscriptionErrorEvent,
)
from app.realtime.transcript_refiner import TranscriptRefiner

logger = logging.getLogger(__name__)

ViewerEventCallback = Callable[[object], Awaitable[None]]
RawOpenAIEventCallback = Callable[[dict[str, Any]], Awaitable[None]]
ErrorCallback = Callable[[str], Awaitable[None]]

OPENAI_TRANSCRIPTION_DELTA_EVENT = "conversation.item.input_audio_transcription.delta"
OPENAI_TRANSCRIPTION_COMPLETED_EVENT = (
    "conversation.item.input_audio_transcription.completed"
)
OPENAI_ERROR_EVENT = "error"


class TranscriptionError(RuntimeError):
    pass


class OpenAITranscriptionEventParseError(ValueError):
    pass


class TranscriptionClient(Protocol):
    model: str
    emits_lifecycle_events: bool

    async def connect(self) -> None:
        pass

    async def send_audio_chunk(self, message: AudioChunkMessage) -> None:
        pass

    async def clear_audio_buffer(self) -> None:
        pass

    async def close(self) -> None:
        pass


@dataclass(frozen=True, slots=True)
class TranscriptionSessionInfo:
    meeting_id: str
    device_id: str
    stream_id: str
    user_id: str
    display_name: str

    @classmethod
    def from_join_message(cls, message: DeviceJoinMessage) -> "TranscriptionSessionInfo":
        return cls(
            meeting_id=message.meeting_id,
            device_id=message.device_id,
            stream_id=message.stream_id,
            user_id=message.user_id,
            display_name=message.display_name,
        )


class OpenAITranscriptionEventConverter:
    def __init__(self, session: TranscriptionSessionInfo) -> None:
        self.session = session
        self._segment_ids_by_openai_key: dict[str, str] = {}
        self._next_segment_number = 1
        self._last_final_text: str | None = None

    def convert(
        self,
        event: dict[str, Any],
    ) -> TranscriptDeltaEvent | TranscriptFinalEvent | None:
        event_type = event.get("type")
        if event_type == OPENAI_TRANSCRIPTION_DELTA_EVENT:
            return self._build_delta_event(event)
        if event_type == OPENAI_TRANSCRIPTION_COMPLETED_EVENT:
            final_event = self._build_final_event(event)
            # 直前の final と同一テキストなら重複として抑制
            if final_event.text == self._last_final_text:
                logger.info(
                    "Suppressed duplicate transcript.final: %r",
                    final_event.text,
                )
                return None
            self._last_final_text = final_event.text
            return final_event
        return None

    def _build_delta_event(self, event: dict[str, Any]) -> TranscriptDeltaEvent:
        return TranscriptDeltaEvent(
            meeting_id=self.session.meeting_id,
            device_id=self.session.device_id,
            stream_id=self.session.stream_id,
            user_id=self.session.user_id,
            display_name=self.session.display_name,
            segment_id=self._segment_id(event),
            text=self._required_text(event, "delta"),
            start_ms=self._optional_ms(event, ("start_ms", "audio_start_ms")),
            end_ms=self._optional_ms(event, ("end_ms", "audio_end_ms")),
        )

    def _build_final_event(self, event: dict[str, Any]) -> TranscriptFinalEvent:
        return TranscriptFinalEvent(
            meeting_id=self.session.meeting_id,
            device_id=self.session.device_id,
            stream_id=self.session.stream_id,
            user_id=self.session.user_id,
            display_name=self.session.display_name,
            segment_id=self._segment_id(event),
            text=self._required_text(event, "transcript"),
            start_ms=self._optional_ms(event, ("start_ms", "audio_start_ms")),
            end_ms=self._optional_ms(event, ("end_ms", "audio_end_ms")),
        )

    def _segment_id(self, event: dict[str, Any]) -> str:
        key = self._openai_segment_key(event)
        segment_id = self._segment_ids_by_openai_key.get(key)
        if segment_id is not None:
            return segment_id

        segment_id = (
            f"{self.session.device_id}_seg_{self._next_segment_number:06d}"
        )
        self._segment_ids_by_openai_key[key] = segment_id
        self._next_segment_number += 1
        return segment_id

    def _openai_segment_key(self, event: dict[str, Any]) -> str:
        item_id = event.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise OpenAITranscriptionEventParseError(
                "OpenAI transcription event item_id is missing"
            )

        content_index = event.get("content_index", 0)
        if not isinstance(content_index, int):
            raise OpenAITranscriptionEventParseError(
                "OpenAI transcription event content_index is invalid"
            )
        return f"{item_id}:{content_index}"

    def _required_text(self, event: dict[str, Any], key: str) -> str:
        value = event.get(key)
        if not isinstance(value, str):
            raise OpenAITranscriptionEventParseError(
                f"OpenAI transcription event {key} is missing"
            )
        return value

    def _optional_ms(
        self,
        event: dict[str, Any],
        keys: tuple[str, ...],
    ) -> int | None:
        for key in keys:
            value = event.get(key)
            if isinstance(value, int):
                return value
        return None


class NullTranscriptionClient:
    def __init__(
        self,
        *,
        model: str,
        emits_lifecycle_events: bool = True,
    ) -> None:
        self.model = model
        self.emits_lifecycle_events = emits_lifecycle_events
        self.connected = False
        self.audio_chunks_received = 0

    async def connect(self) -> None:
        self.connected = True

    async def send_audio_chunk(self, message: AudioChunkMessage) -> None:
        if not self.connected:
            raise TranscriptionError("transcription client is not connected")
        self.audio_chunks_received += 1

    async def clear_audio_buffer(self) -> None:
        pass

    async def close(self) -> None:
        self.connected = False


class OpenAIRealtimeTranscriptionClient:
    emits_lifecycle_events = True

    def __init__(
        self,
        *,
        settings: Settings,
        on_raw_event: RawOpenAIEventCallback,
        on_error: ErrorCallback,
    ) -> None:
        self.settings = settings
        self.model = settings.openai_realtime_transcription_model
        self._on_raw_event = on_raw_event
        self._on_error = on_error
        self._websocket: ClientConnection | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._uncommitted_audio_ms = 0

        # ローカル VAD スマートコミット用ステート
        self._voice_was_active = False  # バッファ内で声が1度でも検出されたか
        self._silence_ms = 0  # 直近の連続無音時間 (ms)

    async def connect(self) -> None:
        api_key = self.settings.openai_api_key
        if not api_key:
            raise TranscriptionError("OPENAI_API_KEY is not set")

        url = self._websocket_url()
        try:
            self._websocket = await websockets.connect(
                url,
                additional_headers={"Authorization": f"Bearer {api_key}"},
            )
            await self._send_session_update()
            self._receive_task = asyncio.create_task(self._receive_events())
        except Exception as exc:
            await self.close()
            raise TranscriptionError("OpenAI transcription connection failed") from exc

    async def send_audio_chunk(self, message: AudioChunkMessage) -> None:
        if self._websocket is None:
            raise TranscriptionError("OpenAI transcription is not connected")

        try:
            await self._websocket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": message.audio.base64,
                    }
                )
            )
            await self._commit_audio_buffer_if_needed(message)
        except Exception as exc:
            raise TranscriptionError("OpenAI transcription audio send failed") from exc

    async def clear_audio_buffer(self) -> None:
        """Discard all uncommitted audio in the OpenAI buffer.

        Used when the active speaker changes to prevent stale cross-talk
        audio from being transcribed.
        """
        if self._websocket is None:
            return

        try:
            await self._websocket.send(
                json.dumps({"type": "input_audio_buffer.clear"})
            )
            self._uncommitted_audio_ms = 0
            self._voice_was_active = False
            self._silence_ms = 0
            logger.info("Cleared OpenAI audio buffer (speaker transition)")
        except Exception:
            logger.warning("Failed to clear OpenAI audio buffer", exc_info=True)

    async def close(self) -> None:
        if self._websocket is not None:
            try:
                await self._commit_audio_buffer()
            except TranscriptionError:
                logger.info("OpenAI transcription final audio commit failed")

        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._websocket is not None:
            await self._websocket.close()
            self._websocket = None

    def _websocket_url(self) -> str:
        parts = urlsplit(self.settings.openai_realtime_websocket_url)
        query_items = [
            (key, value)
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
            if key not in {"intent", "model"}
        ]
        query_items.append(("intent", "transcription"))
        return urlunsplit(
            (
                parts.scheme,
                parts.netloc,
                parts.path,
                urlencode(query_items),
                parts.fragment,
            )
        )

    async def _send_session_update(self) -> None:
        if self._websocket is None:
            raise TranscriptionError("OpenAI transcription is not connected")

        await self._websocket.send(json.dumps(self._session_update_payload()))

    def _session_update_payload(self) -> dict[str, Any]:
        audio_input: dict[str, Any] = {
            "format": {
                "type": "audio/pcm",
                "rate": self.settings.openai_realtime_transcription_sample_rate,
            },
            "transcription": {
                "model": self.model,
                "language": self.settings.openai_realtime_transcription_language,
            },
        }

        if self.settings.openai_realtime_turn_detection_enabled:
            audio_input["turn_detection"] = {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 500,
            }
        else:
            audio_input["turn_detection"] = None

        return {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": audio_input,
                },
            },
        }

    async def _commit_audio_buffer_if_needed(
        self,
        message: AudioChunkMessage,
    ) -> None:
        if self.settings.openai_realtime_turn_detection_enabled:
            return

        self._uncommitted_audio_ms += message.duration_ms

        # VAD スコアの取得と声の状態を更新
        if self.settings.openai_realtime_vad_commit_enabled:
            va_score = (
                message.audio_features.local_voice_activity_score
                if message.audio_features is not None
                and message.audio_features.local_voice_activity_score is not None
                else None
            )

            if va_score is not None:
                threshold = self.settings.openai_realtime_vad_voice_active_threshold
                if va_score >= threshold:
                    self._voice_was_active = True
                    self._silence_ms = 0
                else:
                    self._silence_ms += message.duration_ms

        # ---- コミット判定 ----

        # 安全弁: max_buffer_ms を超えたら無条件でコミット
        if (
            self.settings.openai_realtime_vad_commit_enabled
            and self._uncommitted_audio_ms
            >= self.settings.openai_realtime_vad_max_buffer_ms
        ):
            await self._commit_audio_buffer()
            self._voice_was_active = False
            self._silence_ms = 0
            return

        # 基本間隔に達していなければ何もしない
        if (
            self._uncommitted_audio_ms
            < self.settings.openai_realtime_manual_commit_interval_ms
        ):
            return

        # 基本間隔に達した。VAD 延長を試みる。
        if self.settings.openai_realtime_vad_commit_enabled and self._voice_was_active:
            # 声が最近あった: 300ms 無音が続くまでコミットを延長する
            if self._silence_ms < self.settings.openai_realtime_vad_silence_commit_ms:
                return  # まだ延長中

        # コミット実行: 基本間隔到達 + (延長不要 or 延長終了)
        await self._commit_audio_buffer()
        self._voice_was_active = False
        self._silence_ms = 0

    async def _commit_audio_buffer(self) -> None:
        if self._websocket is None or self._uncommitted_audio_ms <= 0:
            return

        try:
            await self._websocket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.commit",
                    }
                )
            )
            self._uncommitted_audio_ms = 0
        except Exception as exc:
            raise TranscriptionError("OpenAI transcription audio commit failed") from exc

    async def _receive_events(self) -> None:
        if self._websocket is None:
            return

        try:
            async for raw_message in self._websocket:
                try:
                    event = json.loads(raw_message)
                except json.JSONDecodeError:
                    logger.warning("OpenAI Realtime returned non-JSON event")
                    continue

                if not isinstance(event, dict):
                    continue

                await self._on_raw_event(event)
                if event.get("type") == OPENAI_ERROR_EVENT:
                    await self._on_error(self._error_message(event))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.info("OpenAI transcription receive loop stopped: %s", exc)
            await self._on_error("OpenAI transcription receive loop stopped")

    def _error_message(self, event: dict[str, Any]) -> str:
        error = event.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
        return "OpenAI transcription returned an error event"


class TranscriptionClientFactory(Protocol):
    def create(
        self,
        session: TranscriptionSessionInfo,
        on_raw_event: RawOpenAIEventCallback,
        on_error: ErrorCallback,
    ) -> TranscriptionClient:
        pass


class DefaultTranscriptionClientFactory:
    def create(
        self,
        session: TranscriptionSessionInfo,
        on_raw_event: RawOpenAIEventCallback,
        on_error: ErrorCallback,
    ) -> TranscriptionClient:
        settings = get_settings()
        if not settings.openai_api_key:
            return NullTranscriptionClient(
                model=settings.openai_realtime_transcription_model,
                emits_lifecycle_events=False,
            )

        return OpenAIRealtimeTranscriptionClient(
            settings=settings,
            on_raw_event=on_raw_event,
            on_error=on_error,
        )


class TranscriptionClientManager:
    def __init__(
        self,
        factory: TranscriptionClientFactory | None = None,
        refiner: TranscriptRefiner | None = None,
    ) -> None:
        self._factory = factory or DefaultTranscriptionClientFactory()
        self._clients: dict[tuple[str, str], TranscriptionClient] = {}
        self._sessions: dict[tuple[str, str], TranscriptionSessionInfo] = {}
        self._refiner = refiner or TranscriptRefiner()
        self._refine_tasks: set[asyncio.Task[None]] = set()

    async def start_device(
        self,
        message: DeviceJoinMessage,
        broadcast: ViewerEventCallback,
    ) -> None:
        session = TranscriptionSessionInfo.from_join_message(message)
        converter = OpenAITranscriptionEventConverter(session)

        async def on_raw_event(raw_event: dict[str, Any]) -> None:
            try:
                event = converter.convert(raw_event)
            except OpenAITranscriptionEventParseError:
                await broadcast(
                    TranscriptionErrorEvent(
                        meeting_id=session.meeting_id,
                        device_id=session.device_id,
                        stream_id=session.stream_id,
                        user_id=session.user_id,
                        display_name=session.display_name,
                        message="Failed to parse transcription event",
                    )
                )
                return

            if event is not None:
                await broadcast(event)
                # transcript.final の後に LLM で整形 (fire-and-forget)
                if isinstance(event, TranscriptFinalEvent) and self._refiner.enabled:
                    task = asyncio.create_task(
                        self._refiner.refine_and_broadcast(event, broadcast)
                    )
                    self._refine_tasks.add(task)
                    task.add_done_callback(self._refine_tasks.discard)

        async def on_error(error_message: str) -> None:
            await broadcast(
                TranscriptionErrorEvent(
                    meeting_id=session.meeting_id,
                    device_id=session.device_id,
                    stream_id=session.stream_id,
                    user_id=session.user_id,
                    display_name=session.display_name,
                    message=error_message,
                )
            )

        key = self._key(session.meeting_id, session.device_id)
        await self._close_existing_client_for_reconnect(key, broadcast)

        client = self._factory.create(session, on_raw_event, on_error)
        try:
            await client.connect()
        except TranscriptionError as exc:
            await on_error(str(exc))
            self._clients[key] = NullTranscriptionClient(
                model=client.model,
                emits_lifecycle_events=False,
            )
            self._sessions[key] = session
            await self._clients[key].connect()
            return

        self._clients[key] = client
        self._sessions[key] = session
        if client.emits_lifecycle_events:
            await broadcast(
                TranscriptionConnectedEvent(
                    meeting_id=session.meeting_id,
                    device_id=session.device_id,
                    stream_id=session.stream_id,
                    user_id=session.user_id,
                    display_name=session.display_name,
                    model=client.model,
                )
            )

    async def send_audio_chunk(
        self,
        message: AudioChunkMessage,
        broadcast: ViewerEventCallback,
    ) -> None:
        key = self._key(message.meeting_id, message.device_id)
        client = self._clients.get(key)
        session = self._sessions.get(key)
        if client is None:
            await broadcast(
                TranscriptionErrorEvent(
                    meeting_id=message.meeting_id,
                    device_id=message.device_id,
                    stream_id=message.stream_id,
                    user_id=session.user_id if session is not None else "",
                    display_name=session.display_name if session is not None else "",
                    message="transcription client is not started",
                )
            )
            return

        try:
            await client.send_audio_chunk(message)
        except TranscriptionError as exc:
            await broadcast(
                TranscriptionErrorEvent(
                    meeting_id=message.meeting_id,
                    device_id=message.device_id,
                    stream_id=message.stream_id,
                    user_id=session.user_id if session is not None else "",
                    display_name=session.display_name if session is not None else "",
                    message=str(exc),
                )
            )

    async def stop_device(
        self,
        *,
        meeting_id: str,
        device_id: str,
        stream_id: str,
        user_id: str,
        display_name: str,
        broadcast: ViewerEventCallback,
    ) -> None:
        key = self._key(meeting_id, device_id)
        session = self._sessions.get(key)
        if session is not None and session.stream_id != stream_id:
            return

        client = self._clients.pop(key, None)
        self._sessions.pop(key, None)
        if client is None:
            return

        await client.close()
        if client.emits_lifecycle_events:
            await broadcast(
                TranscriptionDisconnectedEvent(
                    meeting_id=meeting_id,
                    device_id=device_id,
                    stream_id=stream_id,
                    user_id=user_id,
                    display_name=display_name,
                )
            )

    async def clear_device_audio_buffer(
        self,
        meeting_id: str,
        device_id: str,
    ) -> None:
        """Clear the OpenAI audio buffer for a specific device.

        Called on active speaker transitions to discard stale cross-talk
        audio that accumulated while the device was not the active speaker.
        """
        key = self._key(meeting_id, device_id)
        client = self._clients.get(key)
        if client is not None:
            await client.clear_audio_buffer()

    def clear(self) -> None:
        self._clients.clear()
        self._sessions.clear()

    async def _close_existing_client_for_reconnect(
        self,
        key: tuple[str, str],
        broadcast: ViewerEventCallback,
    ) -> None:
        client = self._clients.pop(key, None)
        session = self._sessions.pop(key, None)
        if client is None or session is None:
            return

        await client.close()
        if client.emits_lifecycle_events:
            await broadcast(
                TranscriptionDisconnectedEvent(
                    meeting_id=session.meeting_id,
                    device_id=session.device_id,
                    stream_id=session.stream_id,
                    user_id=session.user_id,
                    display_name=session.display_name,
                )
            )

    @staticmethod
    def _key(meeting_id: str, device_id: str) -> tuple[str, str]:
        return (meeting_id, device_id)


transcription_client_manager = TranscriptionClientManager()

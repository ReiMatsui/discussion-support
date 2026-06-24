from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable
from typing import Any
from urllib.parse import urlencode

import websockets

from app.core.config import Settings
from app.voice_agent.context import VoiceAgentContext
from app.voice_agent.prompts import VOICE_AGENT_SHORT_RESPONSE_INSTRUCTIONS


class Realtime2VoiceAgentError(RuntimeError):
    pass


class Realtime2VoiceAgentClient:
    def __init__(self, *, settings: Settings) -> None:
        self._settings = settings

    async def speak(
        self,
        context: VoiceAgentContext,
        *,
        on_audio_delta: Callable[[bytes], None],
    ) -> str:
        return await self.speak_text(
            context.prompt_text(),
            on_audio_delta=on_audio_delta,
        )

    async def speak_text(
        self,
        prompt_text: str,
        *,
        on_audio_delta: Callable[[bytes], None],
    ) -> str:
        if not self._settings.openai_api_key:
            raise Realtime2VoiceAgentError("OPENAI_API_KEY is not set")

        timeout_sec = self._settings.voice_agent_realtime_timeout_ms / 1000
        try:
            async with asyncio.timeout(timeout_sec):
                return await self._speak_text(
                    prompt_text,
                    on_audio_delta=on_audio_delta,
                )
        except Realtime2VoiceAgentError:
            raise
        except TimeoutError as exc:
            raise Realtime2VoiceAgentError("Realtime2 voice agent timed out") from exc
        except Exception as exc:
            raise Realtime2VoiceAgentError("Realtime2 voice agent failed") from exc

    async def _speak_text(
        self,
        prompt_text: str,
        *,
        on_audio_delta: Callable[[bytes], None],
    ) -> str:
        transcript_chunks: list[str] = []
        async with websockets.connect(
            self._websocket_url(),
            additional_headers={
                "Authorization": f"Bearer {self._settings.openai_api_key}",
            },
            max_size=None,
        ) as websocket:
            await websocket.send(json.dumps(self._session_update_payload()))
            conversation_item = self._conversation_item_payload(prompt_text)
            await websocket.send(json.dumps(conversation_item))
            await websocket.send(json.dumps(self._response_create_payload()))

            try:
                async for raw_message in websocket:
                    event = self._parse_event(raw_message)
                    event_type = str(event.get("type", ""))

                    if event_type in {
                        "response.output_audio.delta",
                        "response.audio.delta",
                    }:
                        delta = event.get("delta")
                        if isinstance(delta, str):
                            on_audio_delta(base64.b64decode(delta))
                        continue

                    if event_type in {
                        "response.output_audio_transcript.delta",
                        "response.audio_transcript.delta",
                    }:
                        delta = event.get("delta")
                        if isinstance(delta, str):
                            transcript_chunks.append(delta)
                        continue

                    if event_type == "error":
                        raise Realtime2VoiceAgentError(self._error_message(event))

                    if event_type == "response.done":
                        return "".join(transcript_chunks)
            except asyncio.CancelledError:
                try:
                    await websocket.send(json.dumps({"type": "response.cancel"}))
                except Exception:
                    pass
                raise

        raise Realtime2VoiceAgentError("Realtime2 voice agent ended unexpectedly")

    def _websocket_url(self) -> str:
        query = urlencode({"model": self._settings.openai_realtime_model})
        return f"{self._settings.openai_realtime_websocket_url}?{query}"

    def _session_update_payload(self) -> dict[str, Any]:
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "model": self._settings.openai_realtime_model,
                "instructions": VOICE_AGENT_SHORT_RESPONSE_INSTRUCTIONS,
                "audio": {
                    "output": {
                        "voice": self._settings.openai_realtime_voice,
                        "format": self._output_audio_format(),
                    }
                },
                "reasoning": {
                    "effort": self._settings.openai_realtime_reasoning_effort,
                },
            },
        }

    def _conversation_item_payload(
        self,
        prompt_text: str,
    ) -> dict[str, Any]:
        return {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt_text,
                    }
                ],
            },
        }

    def _response_create_payload(self) -> dict[str, Any]:
        return {
            "type": "response.create",
            "response": {
                "audio": {
                    "output": {
                        "format": self._output_audio_format(),
                    }
                },
            },
        }

    def _output_audio_format(self) -> dict[str, Any]:
        return {
            "type": "audio/pcm",
            "rate": self._settings.openai_realtime_output_sample_rate,
        }

    @staticmethod
    def _parse_event(raw_message: str | bytes) -> dict[str, Any]:
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8")
        event = json.loads(raw_message)
        if not isinstance(event, dict):
            raise Realtime2VoiceAgentError("Realtime2 returned invalid event")
        return event

    @staticmethod
    def _error_message(event: dict[str, Any]) -> str:
        error = event.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return message
        return json.dumps(event, ensure_ascii=False)

from __future__ import annotations

import asyncio
import sys
import time
from collections import deque

from app.audio.output import Pcm16AudioPlayer
from app.core.config import Settings
from app.soniox.client import SonioxEvent, TranscriptEvent, TurnEndedEvent
from app.voice_agent.context import TranscriptTurn, VoiceAgentContext
from app.voice_agent.realtime2_client import Realtime2VoiceAgentClient


def current_timestamp_ms() -> int:
    return time.time_ns() // 1_000_000


class VoiceAgentOrchestrator:
    def __init__(self, *, settings: Settings) -> None:
        self._settings = settings
        self._client = Realtime2VoiceAgentClient(settings=settings)
        self._recent_turns: deque[TranscriptTurn] = deque(maxlen=12)
        self._pending_task: asyncio.Task[None] | None = None
        self._agent_speaking = False
        self._last_agent_started_ms = 0
        self._last_agent_finished_ms = 0
        self._current_player: Pcm16AudioPlayer | None = None
        self._interrupted = False

    @property
    def is_speaking(self) -> bool:
        return self._agent_speaking

    async def interrupt(self, *, reason: str) -> bool:
        if not self._agent_speaking:
            return False

        self._interrupted = True
        if self._current_player is not None:
            try:
                self._current_player.abort()
            except Exception as exc:
                print(
                    f"[voice_agent] audio abort failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        task = self._pending_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._agent_speaking = False
        self._last_agent_finished_ms = 0
        self._interrupted = False

        print(
            f"[voice_agent] interrupted reason={reason}",
            file=sys.stderr,
            flush=True,
        )
        return True

    async def handle_soniox_event(
        self,
        event: SonioxEvent,
        *,
        display_name: str | None = None,
    ) -> None:
        if not self._settings.voice_agent_enabled:
            return

        if self._is_muted_for_agent_audio():
            return

        if isinstance(event, TranscriptEvent):
            self._cancel_pending_trigger()
            if event.is_final and event.text.strip():
                self._recent_turns.append(
                    TranscriptTurn(
                        speaker_label=event.speaker_label,
                        display_name=display_name,
                        text=event.text.strip(),
                        server_timestamp_ms=event.server_timestamp_ms,
                    )
                )
            if event.endpoint_detected or event.is_final:
                self._schedule_trigger("endpoint_detected")
            return

        if isinstance(event, TurnEndedEvent):
            self._schedule_trigger("turn_ended")

    async def aclose(self) -> None:
        task = self._pending_task
        self._cancel_pending_trigger()
        if task is None:
            return
        try:
            await task
        except asyncio.CancelledError:
            pass

    def _schedule_trigger(self, reason: str) -> None:
        if self._agent_speaking:
            return
        self._cancel_pending_trigger()
        self._pending_task = asyncio.create_task(self._trigger_after_silence(reason))

    def _cancel_pending_trigger(self) -> None:
        if self._pending_task is None or self._pending_task.done():
            self._pending_task = None
            return
        self._pending_task.cancel()
        self._pending_task = None

    async def _trigger_after_silence(self, reason: str) -> None:
        try:
            await asyncio.sleep(self._settings.voice_agent_silence_ms / 1000)
            if not self._should_speak():
                return
            await self._speak(reason)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[voice_agent] error: {exc}", file=sys.stderr, flush=True)
        finally:
            if asyncio.current_task() is self._pending_task:
                self._pending_task = None

    def _should_speak(self) -> bool:
        if self._agent_speaking or self._is_muted_for_agent_audio():
            return False

        now_ms = current_timestamp_ms()
        cooldown_elapsed_ms = now_ms - self._last_agent_started_ms
        if cooldown_elapsed_ms < self._settings.voice_agent_cooldown_ms:
            return False

        latest_text = self._latest_text()
        return len(latest_text) >= self._settings.voice_agent_min_transcript_chars

    async def _speak(self, reason: str) -> None:
        context = VoiceAgentContext(
            meeting_id=self._settings.meeting_id,
            recent_turns=list(self._recent_turns),
        )
        self._agent_speaking = True
        self._last_agent_started_ms = current_timestamp_ms()
        print(
            f"[voice_agent] trigger={reason} latest={context.latest_text()!r}",
            file=sys.stderr,
            flush=True,
        )
        transcript = ""
        try:
            player = Pcm16AudioPlayer(
                sample_rate=self._settings.openai_realtime_output_sample_rate
            )
            self._current_player = player
            with player:
                transcript = await self._client.speak(
                    context,
                    on_audio_delta=player.write,
                )
        finally:
            self._current_player = None
            self._agent_speaking = False
            self._last_agent_finished_ms = (
                0 if self._interrupted else current_timestamp_ms()
            )
            self._interrupted = False

        if transcript:
            print(
                f"[voice_agent] said={transcript!r}",
                file=sys.stderr,
                flush=True,
            )

    def _latest_text(self) -> str:
        if not self._recent_turns:
            return ""
        return self._recent_turns[-1].text

    def _is_muted_for_agent_audio(self) -> bool:
        if self._agent_speaking:
            return True
        if self._last_agent_finished_ms == 0:
            return False
        elapsed_ms = current_timestamp_ms() - self._last_agent_finished_ms
        return elapsed_ms < self._settings.voice_agent_post_agent_mute_ms

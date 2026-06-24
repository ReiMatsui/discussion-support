import argparse
import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TextIO

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.audio.mic_capture import microphone_pcm_frames
from app.audio.ring_buffer import PcmRingBuffer, record_pcm_frames
from app.core.config import Settings, get_settings
from app.soniox.client import SonioxEvent, SonioxRealtimeClient, TranscriptEvent
from app.speakers.registry import SpeakerRegistry
from app.speakers.stats import SpeakerStats
from app.speakers.verification.service import SpeakerVerificationService
from app.viewer import viewer_hub
from app.voice_agent.orchestrator import VoiceAgentOrchestrator


def current_timestamp_ms() -> int:
    import time

    return time.time_ns() // 1_000_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(run_viewer_transcription())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


settings = get_settings()
speaker_registry = SpeakerRegistry(settings=settings)
speaker_stats = SpeakerStats(settings=settings)
speaker_audio_buffer = PcmRingBuffer(
    sample_rate=settings.audio_sample_rate,
    channels=settings.audio_channels,
    max_duration_ms=settings.speaker_verification_ring_buffer_ms,
)
speaker_verification = SpeakerVerificationService(
    settings=settings,
    ring_buffer=speaker_audio_buffer,
    event_sink=viewer_hub.broadcast,
)
app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/ws/viewer")
async def viewer_websocket(websocket: WebSocket) -> None:
    await viewer_hub.connect(websocket)
    for event in await speaker_registry.state_events():
        await websocket.send_json(event)
    await websocket.send_json(speaker_stats.as_event())
    try:
        while True:
            raw_message = await websocket.receive_text()
            await handle_viewer_message(raw_message)
    except WebSocketDisconnect:
        pass
    finally:
        await viewer_hub.disconnect(websocket)


async def handle_viewer_message(raw_message: str) -> None:
    try:
        payload = json.loads(raw_message)
    except json.JSONDecodeError:
        return
    if not isinstance(payload, dict):
        return
    for event in await speaker_registry.handle_command(payload):
        speaker_verification.handle_registry_event(event)
        await viewer_hub.broadcast(event)


async def run_viewer_transcription() -> None:
    settings = get_settings()
    await speaker_verification.start()
    await viewer_hub.broadcast(
        {
            "type": "session.started",
            "meeting_id": settings.meeting_id,
            "soniox_model": settings.soniox_model,
            "sample_rate": settings.audio_sample_rate,
            "frame_ms": settings.audio_frame_ms,
            "server_timestamp_ms": current_timestamp_ms(),
        }
    )
    for event in await speaker_registry.state_events():
        speaker_verification.handle_registry_event(event)
        await viewer_hub.broadcast(event)
    await viewer_hub.broadcast(speaker_stats.as_event())

    voice_agent = VoiceAgentOrchestrator(settings=settings)
    try:
        client = SonioxRealtimeClient(settings=settings)
        audio_frames = record_pcm_frames(
            microphone_pcm_frames(
                sample_rate=settings.audio_sample_rate,
                channels=settings.audio_channels,
                frame_ms=settings.audio_frame_ms,
                device=settings.audio_input_device,
            ),
            ring_buffer=speaker_audio_buffer,
        )
        async for event in client.transcribe(audio_frames):
            if isinstance(event, TranscriptEvent):
                enriched, registry_events = await speaker_registry.enrich_transcript(
                    event
                )
                for registry_event in registry_events:
                    speaker_verification.handle_registry_event(registry_event)
                    await viewer_hub.broadcast(registry_event)
                await viewer_hub.broadcast(enriched)
                speaker_verification.observe_transcript(enriched)
                await interrupt_voice_agent_if_needed(
                    settings=settings,
                    voice_agent=voice_agent,
                    event=event,
                )
                if await should_stats_handle(registry_events):
                    stats_event = speaker_stats.update_from_transcript(enriched)
                    if stats_event is not None:
                        await viewer_hub.broadcast(stats_event)
                if await should_voice_agent_handle(event, registry_events):
                    display_name = enriched.get("display_name")
                    await voice_agent.handle_soniox_event(
                        event,
                        display_name=display_name
                        if isinstance(display_name, str)
                        else None,
                    )
            elif not await speaker_registry.is_intro_active():
                await voice_agent.handle_soniox_event(event)
    except asyncio.CancelledError:
        await viewer_hub.broadcast(
            {
                "type": "session.ended",
                "meeting_id": settings.meeting_id,
                "reason": "shutdown",
                "server_timestamp_ms": current_timestamp_ms(),
            }
        )
        raise
    except Exception as exc:
        await viewer_hub.broadcast(
            {
                "type": "transcription.error",
                "meeting_id": settings.meeting_id,
                "message": str(exc),
                "server_timestamp_ms": current_timestamp_ms(),
            }
        )
    finally:
        await voice_agent.aclose()
        await speaker_verification.close()


async def should_voice_agent_handle(
    event: TranscriptEvent,
    registry_events: list[dict[str, object]],
) -> bool:
    ignored_event_types = {
        "speaker.intro.candidate_detected",
        "speaker.intro.completed",
        "speaker.intro.expired",
    }
    if any(item.get("type") in ignored_event_types for item in registry_events):
        return False
    if await speaker_registry.is_intro_active():
        return False
    if await speaker_registry.is_agent_speaker(event.speaker_label):
        return False
    return True


async def interrupt_voice_agent_if_needed(
    *,
    settings: Settings,
    voice_agent: VoiceAgentOrchestrator,
    event: TranscriptEvent,
) -> None:
    if not settings.voice_agent_barge_in_enabled:
        return
    if not voice_agent.is_speaking:
        return
    if len(event.text.strip()) < settings.voice_agent_barge_in_min_chars:
        return
    if not await speaker_registry.is_human_speaker(event.speaker_label):
        return
    await voice_agent.interrupt(
        reason=f"mapped_human_speaker:{event.speaker_label}",
    )


async def should_stats_handle(registry_events: list[dict[str, object]]) -> bool:
    ignored_event_types = {
        "speaker.intro.candidate_detected",
        "speaker.intro.completed",
        "speaker.intro.expired",
    }
    if any(item.get("type") in ignored_event_types for item in registry_events):
        return False
    return not await speaker_registry.is_intro_active()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream the local microphone to Soniox and print transcripts.",
    )
    parser.add_argument(
        "--format",
        choices=("pretty", "jsonl"),
        default="pretty",
        help="Console output format. 'pretty' prints Speaker N lines.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for normalized transcript JSONL logs.",
    )
    return parser.parse_args()


async def run_soniox_console(
    *,
    console_format: str,
    output_path: Path | None,
) -> None:
    settings = get_settings()
    print_startup_settings(settings=settings, output_path=output_path)

    client = SonioxRealtimeClient(settings=settings)
    audio_frames = microphone_pcm_frames(
        sample_rate=settings.audio_sample_rate,
        channels=settings.audio_channels,
        frame_ms=settings.audio_frame_ms,
        device=settings.audio_input_device,
    )

    output_file: TextIO | None = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_path.open("w", encoding="utf-8")

    try:
        async for event in client.transcribe(audio_frames):
            if not isinstance(event, TranscriptEvent):
                continue
            line = json.dumps(event.as_dict(), ensure_ascii=False)
            if console_format == "jsonl":
                print(line, flush=True)
            else:
                print(format_pretty_event(event), flush=True)
            if output_file is not None:
                output_file.write(line + "\n")
                output_file.flush()
    finally:
        if output_file is not None:
            output_file.close()


def print_startup_settings(
    *,
    settings: Settings,
    output_path: Path | None,
) -> None:
    input_device = (
        str(settings.audio_input_device)
        if settings.audio_input_device is not None
        else "default"
    )
    output = str(output_path) if output_path is not None else "none"
    print("Starting meeting-room-v2 microphone transcription", file=sys.stderr)
    print(f"  meeting_id: {settings.meeting_id}", file=sys.stderr)
    print(f"  soniox_model: {settings.soniox_model}", file=sys.stderr)
    print(f"  sample_rate: {settings.audio_sample_rate}", file=sys.stderr)
    print(f"  frame_ms: {settings.audio_frame_ms}", file=sys.stderr)
    print(f"  input_device: {input_device}", file=sys.stderr)
    print(f"  jsonl_output: {output}", file=sys.stderr)
    print(f"  voice_agent_enabled: {settings.voice_agent_enabled}", file=sys.stderr)
    print(
        f"  speaker_verification_backend: "
        f"{settings.speaker_verification_backend}",
        file=sys.stderr,
    )
    if settings.voice_agent_enabled:
        print(
            f"  voice_agent_silence_ms: {settings.voice_agent_silence_ms}",
            file=sys.stderr,
        )
        print(
            f"  voice_agent_cooldown_ms: {settings.voice_agent_cooldown_ms}",
            file=sys.stderr,
        )
    print("Press Ctrl+C to stop.", file=sys.stderr)


def format_pretty_event(event: SonioxEvent) -> str:
    if not isinstance(event, TranscriptEvent):
        return "[turn-end]"
    speaker = f"Speaker {event.speaker_label}" if event.speaker_label else "Speaker ?"
    status = "final" if event.is_final else "delta"
    return f"[{status}] {speaker}: {event.text}"


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(
            run_soniox_console(
                console_format=args.format,
                output_path=args.output,
            )
        )
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)


if __name__ == "__main__":
    main()

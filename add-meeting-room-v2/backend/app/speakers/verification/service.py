from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from app.audio.ring_buffer import PcmRingBuffer
from app.core.config import Settings
from app.speakers.verification.base import (
    BackendName,
    EmbeddingResult,
    average_embeddings,
    cosine_similarity,
)
from app.speakers.verification.workers import embed_in_worker

VerificationMode = Literal["off", "speechbrain", "wespeaker", "both"]
EventSink = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class AudioSample:
    segment_id: str
    speaker_label: str
    participant_id: str | None
    display_name: str | None
    pcm: bytes
    duration_ms: int
    server_timestamp_ms: int


@dataclass(frozen=True, slots=True)
class MappingUpdate:
    speaker_map: dict[str, dict[str, str]]


QueueItem = AudioSample | MappingUpdate


class SpeakerVerificationService:
    def __init__(
        self,
        *,
        settings: Settings,
        ring_buffer: PcmRingBuffer,
        event_sink: EventSink,
    ) -> None:
        self._settings = settings
        self._ring_buffer = ring_buffer
        self._event_sink = event_sink
        self._mode: VerificationMode = settings.speaker_verification_backend
        self._queue: asyncio.Queue[QueueItem] = asyncio.Queue(
            maxsize=settings.speaker_verification_queue_size
        )
        self._consumer_task: asyncio.Task[None] | None = None
        self._executors: dict[BackendName, ProcessPoolExecutor] = {}
        self._cluster_samples: dict[str, list[AudioSample]] = defaultdict(list)
        self._speaker_map: dict[str, dict[str, str]] = {}
        self._profiles: dict[BackendName, dict[str, list[float]]] = defaultdict(dict)
        self._profile_sources: set[tuple[BackendName, str, str]] = set()
        self._pending_evaluation: dict[str, list[AudioSample]] = defaultdict(list)
        self._output_path = Path(settings.speaker_verification_results_path)
        self._output_file = None

    @property
    def enabled(self) -> bool:
        return self._mode != "off"

    async def start(self) -> None:
        if not self.enabled or self._consumer_task is not None:
            return
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_file = self._output_path.open("a", encoding="utf-8")
        for backend in self._backend_names():
            self._executors[backend] = ProcessPoolExecutor(max_workers=1)
        self._consumer_task = asyncio.create_task(self._consume())
        print(
            f"[speaker_verification] mode={self._mode} "
            f"results={self._output_path}",
            file=sys.stderr,
            flush=True,
        )

    async def close(self) -> None:
        task = self._consumer_task
        self._consumer_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        for executor in self._executors.values():
            executor.shutdown(wait=False, cancel_futures=True)
        self._executors.clear()
        if self._output_file is not None:
            self._output_file.close()
            self._output_file = None

    def observe_transcript(self, transcript: dict[str, Any]) -> None:
        if not self.enabled:
            return
        if transcript.get("type") != "transcript.final":
            return
        speaker_label = _optional_string(transcript.get("speaker_label"))
        start_ms = transcript.get("start_ms")
        end_ms = transcript.get("end_ms")
        if (
            speaker_label is None
            or not isinstance(start_ms, int)
            or not isinstance(end_ms, int)
        ):
            return

        pcm = self._ring_buffer.slice_ms(start_ms=start_ms, end_ms=end_ms)
        if pcm is None:
            return
        sample = AudioSample(
            segment_id=str(transcript.get("segment_id") or ""),
            speaker_label=speaker_label,
            participant_id=_optional_string(transcript.get("participant_id")),
            display_name=_optional_string(transcript.get("display_name")),
            pcm=pcm,
            duration_ms=max(0, end_ms - start_ms),
            server_timestamp_ms=int(
                transcript.get("server_timestamp_ms") or time.time_ns() // 1_000_000
            ),
        )
        self._put_nowait(sample)

    def handle_registry_event(self, event: dict[str, Any]) -> None:
        if not self.enabled or event.get("type") != "speaker.map.updated":
            return
        raw_map = event.get("speaker_map")
        if not isinstance(raw_map, dict):
            return
        speaker_map = {
            str(label): {
                key: value
                for key, value in assignment.items()
                if isinstance(key, str) and isinstance(value, str)
            }
            for label, assignment in raw_map.items()
            if isinstance(assignment, dict)
        }
        self._put_nowait(MappingUpdate(speaker_map=speaker_map))

    def _put_nowait(self, item: QueueItem) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(item)

    async def _consume(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if isinstance(item, MappingUpdate):
                    self._speaker_map = item.speaker_map
                    await self._enroll_available_profiles()
                else:
                    await self._handle_sample(item)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(
                    f"[speaker_verification] error: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    async def _handle_sample(self, sample: AudioSample) -> None:
        self._cluster_samples[sample.speaker_label].append(sample)
        assignment = self._speaker_map.get(sample.speaker_label)
        participant_id = (
            assignment.get("participant_id") if assignment is not None else None
        )

        if participant_id is not None and not self._has_all_profiles(participant_id):
            await self._enroll_participant(
                participant_id=participant_id,
                speaker_label=sample.speaker_label,
            )
            return

        self._pending_evaluation[sample.speaker_label].append(sample)
        pending = self._pending_evaluation[sample.speaker_label]
        duration_ms = sum(item.duration_ms for item in pending)
        if duration_ms < self._settings.speaker_verification_min_evaluation_ms:
            return
        self._pending_evaluation[sample.speaker_label] = []
        await self._evaluate_samples(
            samples=pending,
            expected_participant_id=participant_id,
        )

    async def _enroll_available_profiles(self) -> None:
        for speaker_label, assignment in self._speaker_map.items():
            participant_id = assignment.get("participant_id")
            if participant_id is None:
                continue
            await self._enroll_participant(
                participant_id=participant_id,
                speaker_label=speaker_label,
            )

    async def _enroll_participant(
        self,
        *,
        participant_id: str,
        speaker_label: str,
    ) -> None:
        samples = self._cluster_samples.get(speaker_label, [])
        duration_ms = sum(item.duration_ms for item in samples)
        if duration_ms < self._settings.speaker_verification_min_enrollment_ms:
            return
        pcm = b"".join(item.pcm for item in samples)

        for backend in self._backend_names():
            source_key = (backend, participant_id, speaker_label)
            if source_key in self._profile_sources:
                continue
            try:
                result = await self._embed(backend, pcm)
            except Exception as exc:
                error_event = {
                    "type": "speaker.verification.profile.error",
                    "meeting_id": self._settings.meeting_id,
                    "backend": backend,
                    "participant_id": participant_id,
                    "speaker_label": speaker_label,
                    "message": str(exc),
                    "server_timestamp_ms": time.time_ns() // 1_000_000,
                }
                await self._emit(error_event)
                self._write_jsonl(error_event)
                continue
            current = self._profiles[backend].get(participant_id)
            embeddings = (
                [result.embedding]
                if current is None
                else [current, result.embedding]
            )
            self._profiles[backend][participant_id] = average_embeddings(embeddings)
            self._profile_sources.add(source_key)
            ready_event = {
                "type": "speaker.verification.profile.ready",
                "meeting_id": self._settings.meeting_id,
                "backend": backend,
                "participant_id": participant_id,
                "speaker_label": speaker_label,
                "audio_duration_ms": duration_ms,
                "load_ms": result.load_ms,
                "inference_ms": result.inference_ms,
                "server_timestamp_ms": time.time_ns() // 1_000_000,
            }
            await self._emit(ready_event)
            self._write_jsonl(ready_event)

    async def _evaluate_samples(
        self,
        *,
        samples: list[AudioSample],
        expected_participant_id: str | None,
    ) -> None:
        if not samples:
            return
        pcm = b"".join(item.pcm for item in samples)
        backend_results = await asyncio.gather(
            *[
                self._evaluate_backend(
                    backend=backend,
                    pcm=pcm,
                    expected_participant_id=expected_participant_id,
                )
                for backend in self._backend_names()
            ],
            return_exceptions=True,
        )
        results: dict[str, Any] = {}
        for backend, result in zip(
            self._backend_names(),
            backend_results,
            strict=True,
        ):
            if isinstance(result, BaseException):
                results[backend] = {"error": str(result)}
            else:
                results[backend] = result

        event = {
            "type": "speaker.verification.evaluated",
            "meeting_id": self._settings.meeting_id,
            "segment_ids": [item.segment_id for item in samples],
            "speaker_label": samples[0].speaker_label,
            "expected_participant_id": expected_participant_id,
            "audio_duration_ms": sum(item.duration_ms for item in samples),
            "backends": results,
            "server_timestamp_ms": time.time_ns() // 1_000_000,
        }
        await self._emit(event)
        self._write_jsonl(event)

    async def _evaluate_backend(
        self,
        *,
        backend: BackendName,
        pcm: bytes,
        expected_participant_id: str | None,
    ) -> dict[str, Any]:
        profiles = self._profiles[backend]
        if not profiles:
            return {"status": "no_profiles"}

        result = await self._embed(backend, pcm)
        scores = {
            participant_id: cosine_similarity(result.embedding, profile)
            for participant_id, profile in profiles.items()
        }
        ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top1_participant_id, top1_score = ranking[0]
        top2_score = ranking[1][1] if len(ranking) > 1 else None
        margin = top1_score - top2_score if top2_score is not None else None
        return {
            "status": "ok",
            "scores": scores,
            "top1_participant_id": top1_participant_id,
            "top1_score": top1_score,
            "margin": margin,
            "load_ms": result.load_ms,
            "inference_ms": result.inference_ms,
            "correct": (
                top1_participant_id == expected_participant_id
                if expected_participant_id is not None
                else None
            ),
        }

    async def _embed(self, backend: BackendName, pcm: bytes) -> EmbeddingResult:
        loop = asyncio.get_running_loop()
        executor = self._executors[backend]
        model_name = (
            self._settings.speaker_verification_speechbrain_model
            if backend == "speechbrain"
            else self._settings.speaker_verification_wespeaker_model
        )
        return await loop.run_in_executor(
            executor,
            embed_in_worker,
            backend,
            model_name,
            self._settings.speaker_verification_device,
            pcm,
            self._settings.audio_sample_rate,
        )

    async def _emit(self, event: dict[str, Any]) -> None:
        await self._event_sink(event)

    def _write_jsonl(self, event: dict[str, Any]) -> None:
        if self._output_file is None:
            return
        self._output_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._output_file.flush()

    def _has_all_profiles(self, participant_id: str) -> bool:
        return all(
            participant_id in self._profiles[backend]
            for backend in self._backend_names()
        )

    def _backend_names(self) -> list[BackendName]:
        if self._mode == "speechbrain":
            return ["speechbrain"]
        if self._mode == "wespeaker":
            return ["wespeaker"]
        if self._mode == "both":
            return ["speechbrain", "wespeaker"]
        return []


def _optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None

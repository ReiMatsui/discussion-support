from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioChunk:
    start_frame: int
    frame_count: int
    pcm: bytes

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.frame_count


class PcmRingBuffer:
    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int,
        max_duration_ms: int,
    ) -> None:
        if sample_rate <= 0 or channels <= 0 or max_duration_ms <= 0:
            raise ValueError("Invalid PCM ring buffer configuration")
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = 2
        self._bytes_per_frame = channels * self.sample_width
        self._max_frames = int(sample_rate * max_duration_ms / 1000)
        self._chunks: deque[AudioChunk] = deque()
        self._total_frames = 0

    @property
    def total_duration_ms(self) -> int:
        return int(self._total_frames * 1000 / self.sample_rate)

    def append(self, pcm: bytes) -> None:
        if not pcm:
            return
        if len(pcm) % self._bytes_per_frame != 0:
            raise ValueError("PCM byte length is not aligned to complete frames")

        frame_count = len(pcm) // self._bytes_per_frame
        self._chunks.append(
            AudioChunk(
                start_frame=self._total_frames,
                frame_count=frame_count,
                pcm=pcm,
            )
        )
        self._total_frames += frame_count
        self._trim()

    def slice_ms(self, *, start_ms: int, end_ms: int) -> bytes | None:
        if start_ms < 0 or end_ms <= start_ms:
            return None

        start_frame = int(start_ms * self.sample_rate / 1000)
        end_frame = int(end_ms * self.sample_rate / 1000)
        if not self._chunks:
            return None
        if start_frame < self._chunks[0].start_frame:
            return None
        if end_frame > self._total_frames:
            return None

        parts: list[bytes] = []
        for chunk in self._chunks:
            overlap_start = max(start_frame, chunk.start_frame)
            overlap_end = min(end_frame, chunk.end_frame)
            if overlap_end <= overlap_start:
                continue
            local_start = overlap_start - chunk.start_frame
            local_end = overlap_end - chunk.start_frame
            byte_start = local_start * self._bytes_per_frame
            byte_end = local_end * self._bytes_per_frame
            parts.append(chunk.pcm[byte_start:byte_end])
            if overlap_end >= end_frame:
                break

        if not parts:
            return None
        pcm = b"".join(parts)
        expected_bytes = (end_frame - start_frame) * self._bytes_per_frame
        return pcm if len(pcm) == expected_bytes else None

    def _trim(self) -> None:
        minimum_frame = max(0, self._total_frames - self._max_frames)
        while self._chunks and self._chunks[0].end_frame <= minimum_frame:
            self._chunks.popleft()


async def record_pcm_frames(
    frames: AsyncIterator[bytes],
    *,
    ring_buffer: PcmRingBuffer,
) -> AsyncIterator[bytes]:
    async for frame in frames:
        ring_buffer.append(frame)
        yield frame

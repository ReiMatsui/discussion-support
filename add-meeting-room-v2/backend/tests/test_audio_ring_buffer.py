import asyncio

from app.audio.ring_buffer import PcmRingBuffer, record_pcm_frames


def pcm_frames(values: list[int]) -> bytes:
    return b"".join(value.to_bytes(2, "little", signed=True) for value in values)


def test_ring_buffer_slices_audio_relative_milliseconds() -> None:
    ring = PcmRingBuffer(sample_rate=1000, channels=1, max_duration_ms=1000)
    ring.append(pcm_frames(list(range(100))))
    ring.append(pcm_frames(list(range(100, 200))))

    sliced = ring.slice_ms(start_ms=50, end_ms=150)

    assert sliced == pcm_frames(list(range(50, 150)))


def test_ring_buffer_returns_none_for_trimmed_audio() -> None:
    ring = PcmRingBuffer(sample_rate=1000, channels=1, max_duration_ms=100)
    ring.append(pcm_frames(list(range(100))))
    ring.append(pcm_frames(list(range(100, 200))))

    assert ring.slice_ms(start_ms=0, end_ms=50) is None
    assert ring.slice_ms(start_ms=100, end_ms=150) == pcm_frames(
        list(range(100, 150))
    )


def test_record_pcm_frames_copies_frames_before_yielding() -> None:
    async def source():
        yield pcm_frames([1, 2])
        yield pcm_frames([3, 4])

    async def scenario() -> None:
        ring = PcmRingBuffer(sample_rate=1000, channels=1, max_duration_ms=1000)
        received = [
            frame
            async for frame in record_pcm_frames(source(), ring_buffer=ring)
        ]

        assert received == [pcm_frames([1, 2]), pcm_frames([3, 4])]
        assert ring.slice_ms(start_ms=0, end_ms=4) == pcm_frames([1, 2, 3, 4])

    asyncio.run(scenario())

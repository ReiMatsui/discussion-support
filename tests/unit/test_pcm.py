import io
import wave

import numpy as np
import pytest

from das.audio.pcm import tts_pcm24_to_16, tts_wav_to_pcm16


@pytest.mark.parametrize("length", [0, 1, 2, 3, 24, 101, 2400])
def test_linear_conversion_preserves_previous_sample_values(length):
    pcm = np.random.default_rng(11).integers(-32768, 32768, length, dtype=np.int16).tobytes()
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    n_out = int(len(samples) * 16000 / 24000)
    expected = b""
    if n_out >= 2:
        indices = np.linspace(0, len(samples) - 1, n_out)
        low = indices.astype(int)
        high = np.minimum(low + 1, len(samples) - 1)
        fraction = indices - low
        expected = (
            np.clip(samples[low] * (1 - fraction) + samples[high] * fraction, -32768, 32767)
            .astype("<i2")
            .tobytes()
        )
    assert tts_pcm24_to_16(pcm) == expected


@pytest.mark.parametrize("rate,channels", [(16000, 1), (24000, 1), (44100, 2)])
def test_wav_conversion_preserves_previous_nearest_neighbor_result(rate, channels):
    samples = np.arange(200 * channels, dtype=np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(samples.tobytes())
    expected = samples.astype(np.float32)
    if channels > 1:
        expected = expected.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        idx = (np.arange(int(len(expected) * 16000 / rate)) * rate / 16000).astype(int)
        expected = expected[np.clip(idx, 0, len(expected) - 1)]
    assert tts_wav_to_pcm16(buffer.getvalue()) == expected.astype("<i2").tobytes()

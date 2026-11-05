"""TTS検査音声で共用する16bit PCM変換。既存の補間方式を維持する。"""
from __future__ import annotations

import io
import wave

import numpy as np


def tts_pcm24_to_16(pcm: bytes) -> bytes:
    """24kHz mono int16 PCMを、従来の線形補間で16kHzへ変換する。"""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    n_out = int(len(samples) * 16000 / 24000)
    if n_out < 2:
        return b""
    indices = np.linspace(0, len(samples) - 1, n_out)
    low = indices.astype(int)
    high = np.minimum(low + 1, len(samples) - 1)
    fraction = indices - low
    result = samples[low] * (1 - fraction) + samples[high] * fraction
    return np.clip(result, -32768, 32767).astype("<i2").tobytes()


def tts_wav_to_pcm16(content: bytes) -> bytes:
    """TTSの16bit WAVをモノラル16kHzへ変換。従来の最近傍方式を保つ。"""
    with wave.open(io.BytesIO(content)) as wav:
        rate, channels = wav.getframerate(), wav.getnchannels()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        indices = (np.arange(int(len(samples) * 16000 / rate)) * rate / 16000).astype(int)
        samples = samples[np.clip(indices, 0, len(samples) - 1)]
    return samples.astype("<i2").tobytes()

"""--wav 入力の読み込み (_load_wav_mono_16k) の単体テスト.

librosa（未宣言依存）で --wav が常に落ちていたバグの回帰テスト
（docs/design/handoff_2026-07-14_unregistered_speakers.md の実地検証で発覚）。
"""

from __future__ import annotations

import wave

import numpy as np

from das.asr.live._constants import SR
from das.asr.live._audio_io import _load_wav_mono_16k


def _write_wav(path, data_i16: np.ndarray, sr: int, n_ch: int = 1) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(n_ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data_i16.tobytes())


def test_load_pcm16_mono_16k_passthrough(tmp_path):
    sig = (np.sin(np.linspace(0, 100, SR)) * 0.5 * 32767).astype("<i2")
    p = tmp_path / "a.wav"
    _write_wav(p, sig, SR)
    y = _load_wav_mono_16k(str(p))
    assert y.dtype == np.float32
    assert len(y) == SR
    assert abs(float(abs(y).max()) - 0.5) < 0.01


def test_load_stereo_441k_is_downmixed_and_resampled(tmp_path):
    t = np.linspace(0, 1, 44100)
    sig = (np.sin(2 * np.pi * 440 * t) * 0.5 * 32767).astype("<i2")
    stereo = np.column_stack([sig, sig]).ravel()
    p = tmp_path / "b.wav"
    _write_wav(p, stereo, 44100, n_ch=2)
    y = _load_wav_mono_16k(str(p))
    assert len(y) == SR  # 1秒 → 16000サンプル
    assert abs(float(abs(y).max()) - 0.5) < 0.02


def test_load_does_not_require_librosa(tmp_path):
    """librosa が無い環境でも読めること（回帰の核心）."""
    import sys
    import unittest.mock as mock

    sig = np.zeros(SR, dtype="<i2")
    p = tmp_path / "c.wav"
    _write_wav(p, sig, SR)
    with mock.patch.dict(sys.modules, {"librosa": None}):
        y = _load_wav_mono_16k(str(p))
    assert len(y) == SR


def test_load_empty_file_exits_with_conversion_hint(tmp_path):
    """空ファイルはトレースバックではなく変換手順つきの SystemExit（F5）.

    wave モジュールは空・ヘッダ途中のファイルで EOFError を裸で投げる。また
    torchaudio フォールバックは環境によって不可（torchaudio 2.9+ のデコードは
    torchcodec 必須で ImportError）のため、ユーザーが対処できるメッセージで
    終了する（2026-07-15 レビュー、probe_wav.py で確認）。
    """
    import pytest

    p = tmp_path / "empty.wav"
    p.write_bytes(b"")
    with pytest.raises(SystemExit) as e:
        _load_wav_mono_16k(str(p))
    assert "ffmpeg" in str(e.value)          # 変換手順（PCM WAV へ）を案内する


def test_load_truncated_header_exits_with_conversion_hint(tmp_path):
    """RIFFヘッダ途中で切れた破損ファイルも同様に明確なエラーで終了する（F5）."""
    import pytest

    p = tmp_path / "trunc.wav"
    p.write_bytes(b"RIFF\x10\x00\x00\x00WAVE")
    with pytest.raises(SystemExit) as e:
        _load_wav_mono_16k(str(p))
    assert "PCM WAV" in str(e.value)

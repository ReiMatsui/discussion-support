"""共有スピーカー出力（_audio_out）: 複数の音源を 1 本に混ぜる."""
from __future__ import annotations

import numpy as np

from das.asr.live import _audio_out
from das.asr.live.agents._jitter import JitterBuffer

MS = 48


def test_mix_sums_registered_sources_and_clips(monkeypatch):
    monkeypatch.setattr(_audio_out, "_sources", [])
    monkeypatch.setattr(_audio_out, "_started", True)      # デバイスは開かない
    a = JitterBuffer(target_ms=0, min_ms=0)
    b = JitterBuffer(target_ms=0, min_ms=0)
    a.push((np.full(100 * MS // 2, 20000, dtype="<i2")).tobytes())
    b.push((np.full(100 * MS // 2, 20000, dtype="<i2")).tobytes())
    _audio_out.register(a)
    _audio_out.register(b)
    out = _audio_out._mix(50 * MS)
    assert out.shape == (50 * MS // 2,)
    assert np.all(out == 1.0)                               # 足して 1.0 を超える分は切る
    _audio_out.unregister(b)
    out = _audio_out._mix(50 * MS)
    assert abs(float(out[0]) - 20000 / 32768) < 1e-6
    _audio_out.unregister(a)
    assert np.all(_audio_out._mix(50 * MS) == 0.0)          # 音源なし → 無音


def test_register_is_idempotent(monkeypatch):
    monkeypatch.setattr(_audio_out, "_sources", [])
    monkeypatch.setattr(_audio_out, "_started", True)
    src = JitterBuffer()
    _audio_out.register(src)
    _audio_out.register(src)
    assert _audio_out._sources.count(src) == 1
    _audio_out.unregister(src)
    _audio_out.unregister(src)                              # 二重解除でも落ちない

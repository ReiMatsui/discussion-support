"""再生のジッタバッファ（_jitter.JitterBuffer）."""
from __future__ import annotations

from das.asr.live.agents._jitter import JitterBuffer

MS = 48   # 24kHz PCM16 = 48 バイト/ms


def test_buffers_before_playing_then_streams():
    jb = JitterBuffer(target_ms=400)
    jb.begin_turn()
    jb.push(b"\x01" * (100 * MS), now=0.0)
    assert jb.pull(50 * MS, now=0.05) == bytes(50 * MS)     # まだ溜めている → 無音
    for i in range(1, 4):
        jb.push(b"\x01" * (100 * MS), now=0.1 * i)
    assert not jb.buffering                                   # 400ms 溜まった
    assert jb.pull(50 * MS, now=0.35) == b"\x01" * (50 * MS)
    assert jb.buffered_ms == 350


def test_underrun_switches_to_rebuffering_and_widens_the_target():
    jb = JitterBuffer(target_ms=300, step_ms=200, max_ms=1500)
    jb.begin_turn()
    for i in range(3):
        jb.push(b"\x01" * (100 * MS), now=0.1 * i)
    assert not jb.buffering
    for _ in range(6):
        jb.pull(50 * MS, now=0.5)
    assert jb.buffered_ms == 0
    out = jb.pull(50 * MS, now=0.6)                           # 尽きた
    assert out == bytes(50 * MS)
    assert jb.buffering and jb.underruns == 0                 # まだ「途中で尽きた」とは分からない
    jb.push(b"\x01" * (100 * MS), now=0.7)                    # 続きが来た → 途中で尽きていた
    assert jb.underruns == 1 and jb.target_ms == 500
    assert jb.pull(50 * MS, now=0.75) == bytes(50 * MS)      # 500ms 溜まるまで無音
    stats = jb.end_turn()
    assert stats["underruns"] == 1 and stats["target_ms"] == 500


def test_draining_at_the_end_of_a_turn_is_not_an_underrun():
    jb = JitterBuffer(target_ms=100, min_ms=100, step_ms=200)
    jb.begin_turn()
    jb.push(b"\x01" * (100 * MS), now=0.0)
    jb.pull(100 * MS, now=0.1)
    assert jb.pull(50 * MS, now=0.15) == bytes(50 * MS)      # 鳴り終わり
    assert jb.end_turn()["underruns"] == 0
    jb.begin_turn()                                           # 次の発話
    jb.push(b"\x01" * (100 * MS), now=5.0)
    assert jb.underruns == 0 and jb.target_ms <= 100


def test_short_tail_is_played_after_the_wait_limit():
    """発話の終わりの残りが目標に届かなくても、最後の到着から wait_max_sec で鳴らす."""
    jb = JitterBuffer(target_ms=400, wait_max_sec=0.8)
    jb.begin_turn()
    jb.push(b"\x01" * (100 * MS), now=0.0)
    assert jb.pull(50 * MS, now=0.5) == bytes(50 * MS)
    assert jb.pull(50 * MS, now=0.9) == b"\x01" * (50 * MS)


def test_arrival_gaps_are_measured_per_turn_and_target_relaxes_when_smooth():
    jb = JitterBuffer(target_ms=600, min_ms=300, step_ms=200)
    jb.begin_turn()
    jb.push(b"\x01" * (100 * MS), now=0.0)
    jb.push(b"\x01" * (100 * MS), now=0.1)
    jb.push(b"\x01" * (100 * MS), now=1.3)                   # 1.2 秒の空白
    stats = jb.end_turn()
    assert stats["max_gap_ms"] == 1200 and stats["stall_ms"] == 1200
    assert stats["underruns"] == 0
    assert jb.target_ms == 550                                # 尽きなければ少し戻す
    jb.begin_turn()
    assert jb.max_gap_ms == 0 and jb.stall_ms == 0


def test_clear_drops_everything_and_rebuffers():
    jb = JitterBuffer(target_ms=100)
    jb.begin_turn()
    jb.push(b"\x01" * (200 * MS), now=0.0)
    assert not jb.buffering
    jb.clear()
    assert jb.buffered_ms == 0 and jb.buffering

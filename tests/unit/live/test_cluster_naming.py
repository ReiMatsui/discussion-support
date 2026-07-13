"""ClusterVoiceNamer（pyannoteハイブリッド構成のクラスタ単位声紋名前付け）の単体テスト.

設計: docs/design/pyannote_live1_trial_2026-07-09.md §8.4/§9 参照。
"""
from __future__ import annotations

import numpy as np

from das.asr.live._cluster_naming import ClusterVoiceNamer
from das.asr.live._constants import SR


class _FakeTracker:
    """VoiceProfiles.match_profile だけを模したスタブ（副作用なしの照合API）."""

    def __init__(self, results):
        # results: 呼び出しごとに順に返す (name, conf) | None のリスト
        self._results = list(results)
        self.calls: list[int] = []   # 呼び出しごとの音声サンプル数（照合が試みられた回数の記録）

    def match_profile(self, wav):
        self.calls.append(wav.size)
        if not self._results:
            return None
        return self._results.pop(0)


def _wav(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_observe_stays_unconfirmed_below_threshold():
    """累積が閾値未満の間は照合すら試みず未確定のまま."""
    tracker = _FakeTracker([("田中", 0.8)])
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)

    assert namer.observe("pyannote:SPEAKER_00", _wav(2.0)) is None
    assert tracker.calls == []
    assert namer.confirmed_name("pyannote:SPEAKER_00") is None


def test_observe_confirms_once_threshold_reached():
    """累積が閾値に達したら照合し、confidence十分なら名前を確定する."""
    tracker = _FakeTracker([("田中", 0.8)])
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)

    namer.observe("pyannote:SPEAKER_00", _wav(3.0))
    name = namer.observe("pyannote:SPEAKER_00", _wav(2.5))

    assert name == "田中"
    assert namer.confirmed_name("pyannote:SPEAKER_00") == "田中"
    assert len(tracker.calls) == 1


def test_observe_keeps_accumulating_when_confidence_insufficient():
    """confidence不足(match_profileがNone)なら未確定のまま蓄積を続け、再照合のたびに試みる."""
    tracker = _FakeTracker([None, ("鈴木", 0.9)])
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)

    assert namer.observe("pyannote:SPEAKER_00", _wav(3.0)) is None
    assert namer.observe("pyannote:SPEAKER_00", _wav(2.5)) is None   # 1回目照合、confidence不足
    assert namer.confirmed_name("pyannote:SPEAKER_00") is None
    name = namer.observe("pyannote:SPEAKER_00", _wav(3.0))            # 再蓄積後、2回目照合
    assert name == "鈴木"
    assert len(tracker.calls) == 2


def test_observe_confirmed_cluster_returns_name_without_reobserving():
    """一度確定したクラスタは以後照合し直さず、確定名をそのまま返す."""
    tracker = _FakeTracker([("田中", 0.8)])
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)
    namer.observe("pyannote:SPEAKER_00", _wav(5.0))

    name = namer.observe("pyannote:SPEAKER_00", _wav(1.0))

    assert name == "田中"
    assert len(tracker.calls) == 1   # 確定後は再照合しない


def test_observe_skips_overlapped_audio_and_does_not_confirm():
    """重複発話区間の音声は蓄積しない（安全側, 設計点5）."""
    tracker = _FakeTracker([("田中", 0.8)])
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)

    for _ in range(3):
        assert namer.observe("pyannote:SPEAKER_00", _wav(3.0), overlapped=True) is None

    assert tracker.calls == []
    assert namer.confirmed_name("pyannote:SPEAKER_00") is None


def test_observe_confirmed_cluster_still_returns_name_when_overlapped():
    """確定済みクラスタは重複発話フラグが立っていても確定名をそのまま返す."""
    tracker = _FakeTracker([("田中", 0.8)])
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)
    namer.observe("pyannote:SPEAKER_00", _wav(5.0))

    assert namer.observe("pyannote:SPEAKER_00", _wav(1.0), overlapped=True) == "田中"


def test_reset_clears_buffers_and_confirmations():
    """会議リセットで蓄積・確定状態をクリアする."""
    tracker = _FakeTracker([("田中", 0.8)])
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)
    namer.observe("pyannote:SPEAKER_00", _wav(5.0))
    assert namer.confirmed_name("pyannote:SPEAKER_00") == "田中"

    namer.reset()

    assert namer.confirmed_name("pyannote:SPEAKER_00") is None


def test_buffer_caps_at_max_buffer_seconds_without_crashing():
    """際限ない蓄積を防ぐため上限を超えたら古い分から捨てる（直近音声で照合する）."""
    tracker = _FakeTracker([("田中", 0.8)])
    namer = ClusterVoiceNamer(tracker, min_sec=100.0, max_buffer_sec=10.0)

    for _ in range(6):
        namer.observe("pyannote:SPEAKER_00", _wav(3.0))   # 計18秒分投入、上限10秒でトリム

    # min_secに達しないので照合は試みられない（クラッシュしないことの確認が主眼）
    assert namer.confirmed_name("pyannote:SPEAKER_00") is None
    assert tracker.calls == []


def test_independent_clusters_do_not_share_buffers_or_confirmations():
    """クラスタ分裂対策: 別クラスタIDは独立に蓄積・確定する（設計点6）."""
    tracker = _FakeTracker([("田中", 0.8), ("田中", 0.8)])
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)

    name_a = namer.observe("pyannote:SPEAKER_00", _wav(5.0))
    assert namer.observe("pyannote:SPEAKER_01", _wav(2.0)) is None   # 別クラスタは未確定のまま
    name_b = namer.observe("pyannote:SPEAKER_01", _wav(3.0))

    assert name_a == "田中"
    assert name_b == "田中"   # 分裂しても同じ人物へ照合されるだけで無害

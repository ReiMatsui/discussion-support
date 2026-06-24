"""参加度メトリクス（S2）のユニットテスト."""
from __future__ import annotations

from das.asr.live._participation import participation_stats


def test_empty_records():
    assert participation_stats([]) == {}


def test_basic_shares():
    records = [
        {"speaker": "話者1", "text": "a", "ms": 0, "end_ms": 3000},     # 3s
        {"speaker": "話者2", "text": "b", "ms": 3000, "end_ms": 4000},  # 1s
        {"speaker": "話者1", "text": "c", "ms": 4000, "end_ms": 6000},  # 2s
    ]
    s = participation_stats(records)
    assert s["話者1"]["talk_ms"] == 5000
    assert s["話者2"]["talk_ms"] == 1000
    assert s["話者1"]["turns"] == 2
    assert s["話者2"]["turns"] == 1
    # シェア: 時間 5/6 vs 1/6、回数 2/3 vs 1/3
    assert abs(s["話者1"]["time_share"] - 5 / 6) < 1e-9
    assert abs(s["話者2"]["turn_share"] - 1 / 3) < 1e-9
    assert s["話者1"]["last_end_ms"] == 6000
    assert s["話者2"]["last_end_ms"] == 4000


def test_excludes_facilitator():
    records = [
        {"speaker": "話者1", "text": "a", "ms": 0, "end_ms": 2000},
        {"speaker": "ファシリテーター", "text": "戻しましょう", "ms": None, "end_ms": None},
    ]
    s = participation_stats(records, exclude_speakers=("ファシリテーター",))
    assert set(s.keys()) == {"話者1"}


def test_window_filters_old_utterances():
    records = [
        {"speaker": "話者1", "text": "古い", "ms": 0, "end_ms": 1000},        # 0s付近
        {"speaker": "話者2", "text": "新しい", "ms": 600_000, "end_ms": 602_000},  # 10分後
    ]
    # 直近5分窓 → 古い発話(話者1)は窓外、話者2のみ
    s = participation_stats(records, window_ms=300_000)
    assert set(s.keys()) == {"話者2"}


def test_missing_timestamps_counts_turns_only():
    records = [
        {"speaker": "話者1", "text": "a", "ms": None, "end_ms": None},
        {"speaker": "話者1", "text": "b", "ms": None, "end_ms": None},
    ]
    s = participation_stats(records)
    assert s["話者1"]["turns"] == 2
    assert s["話者1"]["talk_ms"] == 0.0
    assert s["話者1"]["turn_share"] == 1.0
    assert s["話者1"]["time_share"] == 0.0  # 時間情報なし → 0

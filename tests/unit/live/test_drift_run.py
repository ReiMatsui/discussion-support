"""脱線の状態機械（DriftRun）: 離れている時間で戻すか決める."""
from __future__ import annotations

from das.asr.live import _constants
from das.asr.live._drift import DriftRun


def _win(*spans):
    """(ms, end_ms) の列から window を作る."""
    return [{"ms": a, "end_ms": b} for a, b in spans]


def test_single_off_remark_does_not_fire():
    run = DriftRun()
    run.observe(_win((0, 5000), (6000, 11000), (12000, 17000)), ["on", "off", "on"], "雑談")
    assert not run.active
    assert run.should_fire(threshold_sec=45) is None


def test_off_run_fires_only_after_the_threshold_and_ignores_asides():
    run = DriftRun()
    run.observe(_win((0, 5000), (6000, 11000), (12000, 17000)), ["on", "off", "aside"], "料理")
    assert run.active and run.run_sec() == 5.0          # aside は区間を切らないが延ばさない
    assert run.should_fire(threshold_sec=45) is None
    # 2発話ごとに判定が来て、窓が進む（古い発話は窓から消える）
    run.observe(_win((6000, 11000), (12000, 17000), (18000, 30000), (31000, 52000)),
                ["off", "aside", "off", "off"], "料理の話")
    assert run.run_start_ms == 6000 and run.run_end_ms == 52000
    assert run.run_sec() == 46.0
    assert run.should_fire(threshold_sec=45) == "料理の話"
    assert run.should_fire(threshold_sec=45) is None      # 続けざまには出さない


def test_returning_to_the_agenda_clears_the_run():
    run = DriftRun()
    run.observe(_win((0, 60000)), ["off"], "雑談")
    assert run.active
    run.observe(_win((0, 60000), (61000, 65000)), ["off", "on"], "")
    assert not run.active and run.should_fire(threshold_sec=10) is None


def test_repeat_after_the_gap_if_still_off():
    run = DriftRun()
    run.observe(_win((0, 50000)), ["off"], "雑談")
    assert run.should_fire(threshold_sec=45) == "雑談"
    run.observe(_win((0, 50000), (51000, 80000)), ["off", "off"], "雑談")
    assert run.should_fire(threshold_sec=45) is None      # 60秒あけるまで待つ
    run.observe(_win((51000, 80000), (81000, 115000)), ["off", "off"], "雑談")
    assert run.should_fire(threshold_sec=45) == "雑談（まだ続いています）"


def test_anchors_freeze_while_off():
    run = DriftRun()
    run.update_anchors(["導入コスト"])
    run.observe(_win((0, 50000)), ["off"], "料理")
    run.update_anchors(["導入コスト", "レシピアプリ"])       # 離れている間に拾った話題
    assert run.anchor_topics == ["導入コスト"]
    run.observe(_win((0, 50000), (51000, 55000)), ["off", "on"], "")
    run.update_anchors(["導入コスト", "効果測定"])
    assert run.anchor_topics == ["導入コスト", "効果測定"]


def test_profiles_carry_a_run_threshold_and_no_confirmation_gate():
    for name, prof in _constants._PROACTIVITY_PROFILES.items():
        assert prof["drift_confirmations"] == 1, name
        assert prof["drift_run_sec"] > 0, name
    assert (_constants._PROACTIVITY_PROFILES["active"]["drift_run_sec"]
            < _constants._PROACTIVITY_PROFILES["standard"]["drift_run_sec"]
            < _constants._PROACTIVITY_PROFILES["controlled"]["drift_run_sec"])

"""自動登録を厳しめにする条件（長さ下限・一貫性しきい値の上乗せ）のテスト.

VoiceProfiles.__init__ はMLモデルを読むため、__new__ で必要フィールドだけ用意し、
_embed をスタブして登録ゲートの挙動を検証する。
"""
from __future__ import annotations

import threading

import numpy as np

from das.asr.live._voice_profiles import SR, VoiceProfiles


def _unit(*v) -> np.ndarray:
    a = np.array(v, dtype=np.float64)
    return a / np.linalg.norm(a)


def _tracker() -> VoiceProfiles:
    vp = VoiceProfiles.__new__(VoiceProfiles)
    vp._lock = threading.RLock()
    vp.profiles = {}
    vp._active_keys = set()
    vp.sp_map = {}
    vp.pool = []
    vp.label_embs = {}
    vp.same_sims = []
    vp.diff_sims = []
    vp.own_sims = {}
    vp.counts = {}
    vp.last = None
    vp.n_anon = 0
    vp.min_sec = 1.0
    vp.short_floor = 0.45
    vp.short_bonus = 0.05
    vp.short_margin_mult = 2.0
    vp.enroll_min_sec = 1.5
    vp.enroll_consist_bonus = 0.08
    vp.margin = 0.05
    vp.thresh = 0.5
    vp.consist = 0.34   # redimnet既定
    vp.dedupe = 0.5
    vp.model = "redimnet"
    vp.auto = True
    return vp


_LONG = np.ones(int(SR * 1.6), dtype=np.float32)          # >= enroll_min_sec
_MID = np.ones(int(SR * 1.2), dtype=np.float32)           # min_sec <= x < enroll_min_sec


def test_clean_long_utterances_register():
    """十分長くクリーンな3発話がそろえば従来どおり人物登録される."""
    vp = _tracker()
    vp._embed = lambda wav: _unit(1, 0, 0)  # type: ignore[method-assign]
    assert vp.classify(_LONG, "1", count=True) == "#1"
    assert vp.classify(_LONG, "1", count=True) == "#1"
    assert vp.classify(_LONG, "1", count=True) == "人物1"
    assert "人物1" in vp.profiles


def test_mid_length_utterances_do_not_register():
    """min_sec以上でも enroll_min_sec 未満の発話は登録に使わない（蓄積もしない）."""
    vp = _tracker()
    vp._embed = lambda wav: _unit(1, 0, 0)  # type: ignore[method-assign]
    for _ in range(4):
        assert vp.classify(_MID, "1", count=True) == "#1"
    assert vp.profiles == {}
    assert vp.pool == []


def test_loose_consistency_blocked_by_bonus():
    """3発話の一貫性が緩い（cs=0.34は超えるがecs=0.42未満）なら登録しない."""
    vp = _tracker()
    vp._embed = lambda wav: _unit(1, 0, 0)  # type: ignore[method-assign]
    vp.classify(_LONG, "1", count=True)     # pool e1
    vp.classify(_LONG, "1", count=True)     # pool e2
    vp._embed = lambda wav: _unit(0.38, 0.925, 0)  # vecAと0.38しか似ていない
    assert vp.classify(_LONG, "1", count=True) == "#1"   # 登録されない
    assert vp.profiles == {}

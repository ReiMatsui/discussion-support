"""短いラリーの取り違え安定化（short_floor 厳格照合）のテスト.

VoiceProfiles.__init__ はMLモデルを読むため、__new__ で必要フィールドだけ用意し、
_embed をスタブして「短い発話でも既知2人を厳格に区別できるときだけ正す」分岐を検証する。
"""
from __future__ import annotations

import threading

import numpy as np

from das.asr.live._voice_profiles import SR, VoiceProfiles


def _unit(*v) -> np.ndarray:
    a = np.array(v, dtype=np.float64)
    return a / np.linalg.norm(a)


def _tracker(emb: np.ndarray) -> VoiceProfiles:
    """既知2人(A,B)を持ち、_embed が emb を返すトラッカー."""
    vp = VoiceProfiles.__new__(VoiceProfiles)
    vp._lock = threading.RLock()
    vp.sp_map = {}
    vp.profiles = {"A": _unit(1, 0, 0), "B": _unit(0, 1, 0)}
    vp._active_keys = {"A", "B"}
    vp.pool = []
    vp.label_embs = {}
    vp.own_sims = {}
    vp.counts = {}
    vp.last = None
    vp.n_anon = 0
    vp.min_sec = 1.0
    vp.short_floor = 0.45
    vp.short_bonus = 0.05
    vp.short_margin_mult = 2.0
    vp.margin = 0.05
    vp.thresh = 0.5
    vp.model = "resemblyzer"
    vp.auto = True
    vp._embed = lambda wav: emb  # type: ignore[method-assign]
    return vp


_SHORT = np.ones(int(SR * 0.6), dtype=np.float32)   # short_floor < 0.6s < min_sec


def test_short_turn_corrects_to_clear_speaker():
    """短い発話でも、はっきりAの声ならAに割り当てる（取り違え安定化）."""
    vp = _tracker(_unit(1, 0, 0))            # 明確にA
    assert vp.classify(_SHORT, "1", count=True) == "A"
    assert vp.profiles.keys() == {"A", "B"}  # 登録は増やさない
    assert vp.pool == []                     # 蓄積もしない


def test_short_turn_ambiguous_falls_back_to_prev():
    """A/Bの中間で判別できない短い発話は、直前の割り当てに追従して暴れない."""
    vp = _tracker(_unit(1, 1, 0))            # AとBの中間（sim差が小さい）
    vp.sp_map["1"] = "B"                     # 直前はB
    assert vp.classify(_SHORT, "1", count=True) == "B"  # 勝手にAへ振らない


def test_short_turn_skipped_for_backchannel():
    """相槌(count=False)は短い厳格照合も通さず、直前に追従（課題④と両立）."""
    vp = _tracker(_unit(1, 0, 0))            # 声はAだが…
    vp.sp_map["1"] = "B"
    assert vp.classify(_SHORT, "1", count=False) == "B"  # 相槌では動かさない


def test_short_turn_needs_two_known_speakers():
    """既知が1人しかいなければ短い厳格照合はしない（区別不要）."""
    vp = _tracker(_unit(1, 0, 0))
    vp.profiles = {"A": _unit(1, 0, 0)}
    vp._active_keys = {"A"}
    assert vp.classify(_SHORT, "1", count=True) == "#1"  # prevなし→素のラベル

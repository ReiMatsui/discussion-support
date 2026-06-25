"""課題④: 相槌を声紋の人物確定に使わない（count=False）テスト.

VoiceProfiles.__init__ はMLモデルを読み込むため、__new__ で必要フィールドだけ
用意して _classify の分岐（count=False は声紋ブロックをスキップ）を検証する。
"""
from __future__ import annotations

import threading

import numpy as np

from das.asr.live._constants import _BACKCHANNEL_RE
from das.asr.live._voice_profiles import SR, VoiceProfiles


def _bare_tracker() -> VoiceProfiles:
    vp = VoiceProfiles.__new__(VoiceProfiles)
    vp._lock = threading.RLock()
    vp.sp_map = {}
    vp.profiles = {}
    vp.pool = []
    vp.label_embs = {}
    vp.own_sims = {}
    vp.counts = {}
    vp.last = None
    vp.n_anon = 0
    vp.min_sec = 1.0
    vp.auto = True
    return vp


def test_backchannel_does_not_enroll():
    """count=False では十分な長さの音声でも声紋を蓄積・登録しない（課題④）."""
    vp = _bare_tracker()
    wav = np.ones(int(SR * 2), dtype=np.float32)  # min_sec を超える長さ
    key = vp.classify(wav, "1", overlapped=False, count=False)
    assert key == "#1"          # 既存割り当てに追従（未知なので素のラベル）
    assert vp.profiles == {}    # 人物登録されない
    assert vp.pool == []        # プールにも溜まらない
    assert vp.n_anon == 0


def test_backchannel_follows_previous_speaker():
    """相槌(count=False)は、そのラベルの直近割り当てに追従する（課題④）."""
    vp = _bare_tracker()
    vp.sp_map["1"] = "松井"     # ラベル1は直前まで松井
    wav = np.ones(int(SR * 2), dtype=np.float32)
    assert vp.classify(wav, "1", overlapped=False, count=False) == "松井"
    assert vp.profiles == {}    # 相槌では声紋を触らない


def test_backchannel_regex_matches_common_aizuchi():
    for t in ["はい", "うん", "なるほど", "ええ", "そうですね"]:
        assert _BACKCHANNEL_RE.match(t), t
    for t in ["それは違うと思います", "コストが高いです"]:
        assert not _BACKCHANNEL_RE.match(t), t

"""課題④: 相槌を声紋の人物確定に使わない（count=False）テスト.

前話者追従（「短い発話＝直前の話者と同じ」の推測）は 2026-07-14 に全モードで
廃止した: 相槌は聞き手が打つ＝直前話者とは別人のことが多く、実測でも正解率28%
(n=32, transcripts/2026-07-14_1729 GT評価) と3人会話ではランダム未満だった
（ユーザー判断）。count=False の発話は、直前が確定済み人物なら未確定を返す。

VoiceProfiles.__init__ はMLモデルを読み込むため、__new__ で必要フィールドだけ
用意して _classify の分岐（count=False は声紋ブロックをスキップ）を検証する。
"""
from __future__ import annotations

import threading

import numpy as np

from das.asr.live._constants import _BACKCHANNEL_RE, UNSURE_SPEAKER
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


def test_backchannel_does_not_follow_previous_speaker():
    """相槌(count=False)は、直前の確定人物へ追従せず未確定になる（追従廃止）.

    旧仕様は「そのラベルの直近割り当て（松井）に追従」だったが、実測正解率28%
    （3人会話・n=32）でランダム未満＝害だったため、ユーザー判断で全モード廃止。
    sp_map（ラベル連続性）は保持し、次の確信ある声紋一致で連続性を回復する。
    """
    vp = _bare_tracker()
    vp.sp_map["1"] = "松井"     # ラベル1は直前まで松井
    wav = np.ones(int(SR * 2), dtype=np.float32)
    assert vp.classify(wav, "1", overlapped=False, count=False) == UNSURE_SPEAKER
    assert vp.last["kind"] == "相槌未確定"   # diag には従来どおり判定種別を残す
    assert vp.sp_map["1"] == "松井"          # マッピングは保持（連続性の回復用）
    assert vp.profiles == {}    # 相槌では声紋を触らない


def test_backchannel_regex_matches_common_aizuchi():
    for t in ["はい", "うん", "なるほど", "ええ", "そうですね"]:
        assert _BACKCHANNEL_RE.match(t), t
    for t in ["それは違うと思います", "コストが高いです"]:
        assert not _BACKCHANNEL_RE.match(t), t

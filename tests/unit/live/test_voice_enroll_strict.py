"""累積文字数ベースの自動登録のテスト.

「発話数」ではなく「声ごとのクリーンな発声の累積文字数」で登録する。長い連続発話は
窓分割で複数サンプル化して即登録でき、短い発話は累積で登録される。混在した声は
束ねられず、十分量に達した声だけが確定する。
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
    vp.enroll_min_total_chars = 45
    vp.enroll_win_sec = 1.5
    vp.enroll_consist_bonus = 0.08
    vp._POOL_CAP = 24
    vp.margin = 0.05
    vp.thresh = 0.5
    vp.consist = 0.34
    vp.dedupe = 0.5
    vp.model = "redimnet"
    vp.auto = True
    return vp


def test_long_continuous_utterance_registers_at_once():
    """1回の長い連続発話（窓分割で複数サンプル化）だけで即登録される."""
    vp = _tracker()
    vp._embed = lambda wav: _unit(1, 0, 0)  # type: ignore[method-assign]
    long_wav = np.ones(int(SR * 6), dtype=np.float32)   # 6秒 → 4窓
    assert vp.classify(long_wav, "1", count=True, chars=60) == "人物1"
    assert "人物1" in vp.profiles


def test_short_utterances_accumulate_then_register():
    """短い発話でも累積文字数が閾値を超えれば登録される."""
    vp = _tracker()
    vp._embed = lambda wav: _unit(1, 0, 0)  # type: ignore[method-assign]
    mid = np.ones(int(SR * 1.2), dtype=np.float32)
    assert vp.classify(mid, "1", count=True, chars=12) == "#1"   # 12
    assert vp.classify(mid, "1", count=True, chars=12) == "#1"   # 24
    assert vp.classify(mid, "1", count=True, chars=12) == "#1"   # 36
    assert vp.classify(mid, "1", count=True, chars=12) == "人物1"  # 48 >= 45


def test_short_exchange_utterances_accumulate_and_register():
    """min_sec未満の短い発話の応酬でも、累積で登録される（ショートパス経由）."""
    vp = _tracker()
    vp._embed = lambda wav: _unit(1, 0, 0)  # type: ignore[method-assign]
    short = np.ones(int(SR * 0.6), dtype=np.float32)   # < min_sec → ショートパス
    last = None
    for _ in range(5):                                  # 5×10字 = 50 >= 45
        last = vp.classify(short, "1", count=True, chars=10)
    assert last == "人物1"
    assert "人物1" in vp.profiles


def test_low_total_chars_never_registers():
    """累積が閾値に届かない（中身の薄い発話ばかり）なら登録しない."""
    vp = _tracker()
    vp._embed = lambda wav: _unit(1, 0, 0)  # type: ignore[method-assign]
    mid = np.ones(int(SR * 1.2), dtype=np.float32)
    for _ in range(6):
        assert vp.classify(mid, "1", count=True, chars=5) == "#1"
    assert vp.profiles == {}


def test_mixed_voices_do_not_pool_together():
    """別々の声は束ねられない（混ざって早期登録されない）."""
    vp = _tracker()
    mid = np.ones(int(SR * 1.2), dtype=np.float32)
    voice_a, voice_b = _unit(1, 0, 0), _unit(0, 1, 0)
    for v in (voice_a, voice_b, voice_a, voice_b):   # A,B各40字ぶん貯まるが閾値未満
        vp._embed = lambda wav, _v=v: _v
        assert vp.classify(mid, "1", count=True, chars=20) == "#1"
    assert vp.profiles == {}                    # どちらもまだ登録されない
    vp._embed = lambda wav: voice_a              # Aがもう1回 → Aだけ60字で登録
    assert vp.classify(mid, "1", count=True, chars=20) == "人物1"


def test_commit_merges_into_existing_person():
    """累積した声が既存人物と一致すれば、新規ではなく合流（重複登録を防ぐ）."""
    vp = _tracker()
    vp.profiles = {"山下くん": _unit(1, 0, 0)}
    vp._active_keys = {"山下くん"}
    assert vp._commit_profile(_unit(1, 0, 0), "2", None, 50) == "山下くん"
    assert vp.last["kind"] == "合流"
    assert not any(k.startswith("人物") for k in vp.profiles)

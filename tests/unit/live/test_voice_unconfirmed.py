"""新規話者の登録直前の発話を「未確定」として扱うテスト.

Sonioxが新しい声を既存ラベル（登録済みの人）に混ぜて出すと、声紋が一致しない
のに直前の人へ追従してしまう。声紋が prev と一致しないときは未確定(#ラベル)に
落とし、登録時に遡及リネームでまとめて確定できることを検証する。
"""
from __future__ import annotations

import threading

import numpy as np

from das.asr.live._constants import UNSURE_SPEAKER
from das.asr.live._voice_profiles import SR, VoiceProfiles


def _unit(*v) -> np.ndarray:
    a = np.array(v, dtype=np.float64)
    return a / np.linalg.norm(a)


def _tracker(emb: np.ndarray) -> VoiceProfiles:
    """登録済み「松井」を持ち、_embed が emb を返すトラッカー（ラベル2=松井に固定）."""
    vp = VoiceProfiles.__new__(VoiceProfiles)
    vp._lock = threading.RLock()
    vp.profiles = {"松井": _unit(1, 0, 0)}
    vp._active_keys = {"松井"}
    vp.sp_map = {"2": "松井"}          # Sonioxラベル2は直前まで松井
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
    vp.max_human_speakers = None
    vp.margin = 0.05
    vp.thresh = 0.5
    vp.consist = 0.6
    vp.dedupe = 0.7
    vp.model = "resemblyzer"
    vp.auto = True
    vp._embed = lambda wav: emb  # type: ignore[method-assign]
    return vp


_LONG = np.ones(int(SR * 1.6), dtype=np.float32)


def test_new_voice_not_shown_as_registered_person():
    """登録済みラベルに混ざった別人の声は、松井ではなく未確定(#2)になる."""
    vp = _tracker(_unit(0, 1, 0))      # 松井と全く違う声
    assert vp.classify(_LONG, "2", count=True) == "#2"
    assert vp.sp_map["2"] == "#2"      # マッピングも未確定に更新


def test_registration_retroactively_renames_unconfirmed():
    """別人の声が累積して人物登録されると、#2→人物Nの遡及リネームが出る."""
    vp = _tracker(_unit(0, 1, 0))
    assert vp.classify(_LONG, "2", count=True, chars=20) == "#2"   # 20
    assert vp.classify(_LONG, "2", count=True, chars=20) == "#2"   # 40
    assert vp.classify(_LONG, "2", count=True, chars=20) == "人物1"  # 60 >= 45
    assert vp.last["kind"] == "自動登録"
    assert vp.last["rename"] == ("#2", "人物1")          # 過去分を遡及で確定


def test_matching_voice_still_follows_registered_person():
    """声紋が松井に一致する発話は、従来どおり松井のまま（誤って未確定にしない）."""
    vp = _tracker(_unit(1, 0, 0))      # まさに松井の声
    assert vp.classify(_LONG, "2", count=True) == "松井"


def test_anonymous_person_keeps_same_label_on_moderate_match():
    """自動登録済み人物は、同じSTTラベルの低信頼発話を近ければ継続表示する."""
    vp = _tracker(_unit(0.90, 0.43, 0))
    vp.profiles = {"人物1": _unit(0, 1, 0)}
    vp._active_keys = {"人物1"}
    vp.sp_map = {"2": "人物1"}

    assert vp.classify(_LONG, "2", count=True, chars=20) == "人物1"
    assert vp.last["kind"] == "低信頼追従"


def test_max_speakers_turns_extra_new_voice_unsure():
    """参加人数上限に達した後の新しい声は、新参加者ではなく未確定にする."""
    vp = _tracker(_unit(0, 1, 0))
    vp.profiles = {"人物1": _unit(1, 0, 0)}
    vp._active_keys = {"人物1"}
    vp.sp_map = {}
    vp.max_human_speakers = 1

    assert vp.classify(_LONG, "3", count=True, chars=60) == UNSURE_SPEAKER
    assert vp.sp_map["3"] == UNSURE_SPEAKER
    assert "人物2" not in vp.profiles
    assert vp.last["kind"] == "話者数上限"

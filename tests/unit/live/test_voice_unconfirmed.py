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


def _closed_roster_tracker(emb: np.ndarray) -> VoiceProfiles:
    """登録済み A/B/C を持つ名簿確定(auto=False)トラッカー（4次元・直交声紋）."""
    vp = _tracker(emb)
    vp.profiles = {
        "A": _unit(1, 0, 0, 0),
        "B": _unit(0, 1, 0, 0),
        "C": _unit(0, 0, 1, 0),
    }
    vp._active_keys = {"A", "B", "C"}
    vp.sp_map = {}
    vp.auto = False   # 名簿を確定
    return vp


def test_closed_roster_long_unknown_marks_unsure():
    """閉じた名簿で、登録者の誰とも一致しない長い声は未確定（新規匿名を作らない）."""
    vp = _closed_roster_tracker(_unit(0, 0, 0, 1))   # A/B/C いずれとも直交=無一致
    assert vp.classify(_LONG, "7", count=True, chars=60) == UNSURE_SPEAKER
    assert vp.sp_map["7"] == UNSURE_SPEAKER
    assert vp.profiles.keys() == {"A", "B", "C"}     # 人物Nを増やさない
    assert vp.n_anon == 0


def test_closed_roster_confident_match_keeps_registered_name():
    """閉じた名簿でも、はっきり登録者の声なら従来どおりその名前に割り当てる."""
    vp = _closed_roster_tracker(_unit(1, 0, 0, 0))   # まさにAの声
    assert vp.classify(_LONG, "7", count=True, chars=60) == "A"


def test_closed_roster_does_not_autoenroll_unknown():
    """閉じた名簿では、未一致の声を何度受け取っても人物Nに自動登録しない."""
    vp = _closed_roster_tracker(_unit(0, 0, 0, 1))
    for _ in range(4):
        assert vp.classify(_LONG, "7", count=True, chars=60) == UNSURE_SPEAKER
    assert vp.profiles.keys() == {"A", "B", "C"}
    assert vp.n_anon == 0
    assert vp.pool == []


def test_closed_roster_unknown_does_not_inherit_registered_label():
    """STTが登録者と同じ生ラベルを別人に再利用しても、登録者を継がず未確定にする."""
    vp = _closed_roster_tracker(_unit(0, 0, 0, 1))   # Aと全く違う声
    vp.sp_map = {"2": "A"}                            # ラベル2は直前までA
    assert vp.classify(_LONG, "2", count=True, chars=60) == UNSURE_SPEAKER


def test_closed_roster_overlapped_speech_does_not_inherit_registered_label():
    """重なり発話は声紋を信用できないため、閉じた名簿では直前登録者を継がず未確定."""
    vp = _closed_roster_tracker(_unit(1, 0, 0, 0))
    vp.sp_map = {"2": "A"}

    assert vp.classify(_LONG, "2", overlapped=True, count=True, chars=60) == UNSURE_SPEAKER


def test_enroll_false_still_matches_but_does_not_accumulate():
    """enroll=False（エコー窓中）でも声紋一致で実名を返すが、蓄積はしない（P2-2）."""
    vp = _tracker(_unit(1, 0, 0))      # まさに松井の声
    assert vp.classify(_LONG, "2", count=True, enroll=False, chars=60) == "松井"
    # 蓄積用バッファ（label_embs）にはこの発話を溜めない
    assert vp.label_embs.get("2", []) == []


def test_enroll_false_does_not_autoenroll_new_person():
    """enroll=False の未知の声は、count=True で照合はするが人物Nを新規登録しない（P2-2）."""
    vp = _tracker(_unit(0, 1, 0))      # 松井と全く違う未知の声
    vp.n_anon = 0
    for _ in range(4):
        got = vp.classify(_LONG, "2", count=True, enroll=False, chars=60)
        assert got != "人物1"          # 蓄積が進まないので自動登録されない
    assert "人物1" not in vp.profiles
    assert vp.n_anon == 0


def test_enroll_true_still_accumulates_and_registers():
    """enroll=True（通常）は従来どおり蓄積して人物Nを自動登録する（回帰）."""
    vp = _tracker(_unit(0, 1, 0))
    vp.n_anon = 0
    assert vp.classify(_LONG, "2", count=True, enroll=True, chars=20) == "#2"
    assert vp.classify(_LONG, "2", count=True, enroll=True, chars=20) == "#2"
    assert vp.classify(_LONG, "2", count=True, enroll=True, chars=20) == "人物1"


def test_reset_keeps_activated_named_profiles():
    """リセット後も、有効化済みの実名プロファイルは照合対象に残る（課題C2）.

    匿名「人物N」はセッション限りなので落とし、AI声紋は維持する。
    """
    vp = _tracker(_unit(1, 0, 0))
    vp.profiles = {"松井": _unit(1, 0, 0), "人物1": _unit(0, 1, 0)}
    vp._active_keys = {"松井", "人物1", "__AI__"}
    vp.n_anon = 1

    vp.reset_session()

    assert "松井" in vp._active_human()      # 実名は次の会議へ引き継ぐ
    assert "人物1" not in vp._active_keys     # 匿名 人物N は非活性化
    assert "__AI__" in vp._active_keys        # AI声紋はエコー除去用に維持


def test_reset_keeps_named_profile_matchable_closed_roster():
    """リセット後、その実名話者の声は closed roster(auto=False) でも実名に一致する."""
    vp = _closed_roster_tracker(_unit(1, 0, 0, 0))   # まさにAの声

    vp.reset_session()

    assert vp.classify(_LONG, "7", count=True, chars=60) == "A"


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

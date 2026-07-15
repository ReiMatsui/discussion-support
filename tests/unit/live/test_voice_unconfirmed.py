"""未確定ラベルの蓄積・自動登録・ラベル継続のテスト.

新しいラベルの発話は #ラベルに蓄積され、登録時に遡及リネームでまとめて確定する。
一度声紋照合の成功で人物に結び付いたラベルは、照合に失敗した発話でも対応を
維持する（ラベル継続。2026-07-14, eval/replay_attribution.py での再設計:
不一致で対応を破棄する旧仕様は同一人物を #ラベルと人物Nに分裂させ
1:1帰属精度44%、継続化で54%→他の変更と合わせ79%）。
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
    vp.own_embs = {}
    vp._own_updates = {}
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


def test_mismatching_voice_keeps_label_person():
    """人物に結び付いたラベルは、照合に失敗した発話でも対応を維持（ラベル継続）.

    旧仕様は #2 に落としていたが、実会話では中尺発話の声紋が本人でもしきい値に
    届かず（eval/replay_attribution.py: 本人一致 0.17〜0.45）、一度の不一致で
    同一人物が #ラベルと人物に分裂して 1:1帰属精度44% の主因になっていた。
    """
    vp = _tracker(_unit(0, 1, 0))      # 松井と一致しない声
    assert vp.classify(_LONG, "2", count=True) == "松井"   # ラベル継続
    assert vp.sp_map["2"] == "松井"    # マッピングは維持


def test_registration_retroactively_renames_unconfirmed():
    """未確定ラベルの声が累積して人物登録されると、#2→人物Nの遡及リネームが出る."""
    vp = _tracker(_unit(0, 1, 0))
    vp.sp_map = {}                     # ラベル2はまだ誰にも結び付いていない
    assert vp.classify(_LONG, "2", count=True, chars=20) == "#2"   # 20
    assert vp.classify(_LONG, "2", count=True, chars=20) == "#2"   # 40
    assert vp.classify(_LONG, "2", count=True, chars=20) == "人物1"  # 60 >= 45
    assert vp.last["kind"] == "自動登録"
    assert vp.last["rename"] == ("#2", "人物1")          # 過去分を遡及で確定


def test_matching_voice_still_follows_registered_person():
    """声紋が松井に一致する発話は、従来どおり松井のまま（誤って未確定にしない）."""
    vp = _tracker(_unit(1, 0, 0))      # まさに松井の声
    assert vp.classify(_LONG, "2", count=True) == "松井"


def test_anonymous_person_weak_match_keeps_label_continuation():
    """匿名人物(人物1)に結び付いたラベルも、弱い一致の発話で対応を壊さない.

    しきい値を満たさない発話は帰属根拠を「そのラベルの現在の対応」に置く
    （ラベル継続）。対応先は声紋照合の成功でしか書き換わらないため、
    照合失敗のたびに #ラベルへ落とす旧仕様（同一人物の分裂＝帰属精度44%の主因）
    より一貫性が高い。蓄積(pool)は並行して進み、別人なら自動登録で分離される。
    """
    vp = _tracker(_unit(0.90, 0.43, 0))
    vp.profiles = {"人物1": _unit(0, 1, 0)}
    vp._active_keys = {"人物1"}
    vp.sp_map = {"2": "人物1"}

    assert vp.classify(_LONG, "2", count=True, chars=20) == "人物1"
    assert vp.last["kind"] == "蓄積中"   # 蓄積は並行して進む（ラベル継続で返答）


def test_medium_turn_requires_strict_threshold():
    """min_sec〜strict_sec の中尺発話は、短発話と同じ厳格しきい値でのみ即時判定.

    実測（eval/replay_attribution.py, 2026-07-14_142016）: 5秒超の照合は6/6正解
    に対し1〜2.5秒は誤一致が集中（1.1s sim=0.43, 2.0s sim=0.49 が別人に一致し、
    ラベルの人物対応を破壊）。埋め込みの信頼性は発話長に依存するため、
    基準しきい値(thresh)で信じるのは strict_sec 以上の発話のみ。
    """
    emb = _unit(0.55, float(np.sqrt(1 - 0.55 ** 2)), 0)   # 松井とのsim=0.55
    vp = _tracker(emb)                 # thresh=0.5, short_bonus=0.05
    vp.short_bonus = 0.08
    vp.sp_map = {}
    # 1.6s（< strict_sec=3.0）: 0.55 < 0.5+0.08 → 即時判定せず蓄積へ
    assert vp.classify(_LONG, "2", count=True, chars=20) == "#2"
    assert vp.last["kind"] == "蓄積中"
    # 3.2s（>= strict_sec）: 0.55 >= 0.5 → 基準しきい値で即時判定
    long_wav = np.ones(int(SR * 3.2), dtype=np.float32)
    assert vp.classify(long_wav, "2", count=True, chars=20) == "松井"
    assert vp.last["kind"] == "声紋一致"


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


def test_closed_roster_label_continuation_keeps_registered_person():
    """閉じた名簿でも、登録者Aに結び付いたラベルは照合失敗の発話でAを維持する.

    閉じた名簿ポリシーは「登録済みのアクティブな名前付きプロファイルへの継続
    だけを許す」。Aは声紋照合の成功でラベルに結び付いた登録者なので継続対象
    （未知・匿名への継続は従来どおり未確定に落ちる）。
    """
    vp = _closed_roster_tracker(_unit(0, 0, 0, 1))   # Aと一致しない声
    vp.sp_map = {"2": "A"}                            # ラベル2は直前までA
    assert vp.classify(_LONG, "2", count=True, chars=60) == "A"


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
    vp.sp_map = {}                     # ラベル2はまだ誰にも結び付いていない
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

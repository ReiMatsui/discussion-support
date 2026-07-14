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
    vp.max_human_speakers = None
    vp._embed = lambda wav: emb  # type: ignore[method-assign]
    return vp


_SHORT = np.ones(int(SR * 0.6), dtype=np.float32)   # short_floor < 0.6s < min_sec


def test_short_turn_corrects_to_clear_speaker():
    """短い発話でも、はっきりAの声ならAに割り当てる（取り違え安定化）."""
    vp = _tracker(_unit(1, 0, 0))            # 明確にA
    assert vp.classify(_SHORT, "1", count=True) == "A"
    assert vp.profiles.keys() == {"A", "B"}  # 登録は増やさない
    assert vp.pool == []                     # 蓄積もしない


def test_short_turn_ambiguous_marks_unsure():
    """A/Bの中間で判別できない短い発話は、確定済みの人へ追従せず未確定にする."""
    vp = _tracker(_unit(1, 1, 0))            # AとBの中間（sim差が小さい）
    vp.sp_map["1"] = "B"                     # 直前はB（確定済み）
    assert vp.classify(_SHORT, "1", count=True) == "?"   # Bと言い切らず未確定
    assert vp.sp_map["1"] == "B"             # マッピングは保持（次の確信発話の連続性）


def test_short_turn_closed_roster_marks_unsure():
    """閉じた名簿(auto=False)でも、判別できない短い曖昧発話は未確定にする.

    名簿確定モードは「登録済みの人 or 未確定」。証拠の無い曖昧な短発話を
    直近の確定者(B)に決めつけるのは、誤った確信付与になるので避ける。
    """
    vp = _tracker(_unit(1, 1, 0))      # A/Bの中間で判別不能
    vp.auto = False                    # 名簿確定モード
    vp.sp_map["1"] = "B"               # 直前はB
    assert vp.classify(_SHORT, "1", count=True) == "?"   # Bと言い切らず未確定
    assert vp.sp_map["1"] == "B"       # マッピングは保持（次の確信発話の連続性）


def test_short_turn_skipped_for_backchannel_marks_unsure():
    """相槌(count=False)は短い厳格照合を通さず、直前の確定人物にも追従しない.

    旧仕様は直前のBへ追従だったが、前話者追従は実測正解率28%（3人会話・n=32,
    transcripts/2026-07-14_1729 GT評価）でランダム未満だったため、ユーザー判断で
    全モード廃止（2026-07-14）。未確定を返し、sp_map は保持する。
    """
    vp = _tracker(_unit(1, 0, 0))            # 声はAだが相槌なので照合しない
    vp.sp_map["1"] = "B"
    assert vp.classify(_SHORT, "1", count=False) == "?"
    assert vp.last["kind"] == "相槌未確定"
    assert vp.sp_map["1"] == "B"             # マッピングは保持（連続性の回復用）  # 相槌では動かさない


def test_short_turn_needs_two_known_speakers():
    """既知が1人しかいなければ短い厳格照合はしない（区別不要）."""
    vp = _tracker(_unit(1, 0, 0))
    vp.profiles = {"A": _unit(1, 0, 0)}
    vp._active_keys = {"A"}
    assert vp.classify(_SHORT, "1", count=True) == "#1"  # prevなし→素のラベル


def test_short_turn_hybrid_matches_single_known_speaker():
    """ハイブリッド時は既知1人でも短発話を声紋照合し、当たれば声紋一致にする.

    実測（transcripts/2026-07-14_1729 GT評価）で声紋一致92% vs 前話者追従28%。
    蓄積期（登録1人）の短発話を追従に落とさず、当たる機構＝声紋照合に回す。
    """
    vp = _tracker(_unit(1, 0, 0))            # 明確にA
    vp.profiles = {"A": _unit(1, 0, 0)}
    vp._active_keys = {"A"}
    vp.hybrid = True
    assert vp.classify(_SHORT, "1", count=True) == "A"
    assert vp.last["kind"] == "声紋一致"
    assert vp.pool == []                     # 登録・蓄積はしない（既存どおり）


def test_short_turn_hybrid_keeps_strict_threshold():
    """ハイブリッドでも照合しきい値は既存の厳格運用のまま（弱い一致は拾わない）."""
    vp = _tracker(_unit(1, 2, 0))            # A(1,0,0)とはsim≈0.45 < 厳格しきい値
    vp.profiles = {"A": _unit(1, 0, 0)}
    vp._active_keys = {"A"}
    vp.hybrid = True
    # 当たらないだけで誤りは増やさない: prev が無いので #ラベルのプレースホルダに
    # 落ちる（確定人物への追従は全モード廃止済みだが、#ラベル継続はSTTラベル
    # ベースの機構＝遡及リネームの土台なので維持）。
    assert vp.classify(_SHORT, "1", count=True) == "#1"
    assert vp.last["kind"] == "相槌追従"

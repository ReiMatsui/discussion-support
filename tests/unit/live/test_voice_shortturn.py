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
    vp.own_embs = {}
    vp._own_updates = {}
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


def test_short_turn_ambiguous_keeps_label_continuation():
    """A/Bの中間で判別できない短い発話は、ラベルの現在の対応(B)を維持する.

    ラベル継続（2026-07-14, eval/replay_attribution.py での再設計）: 対応先Bは
    過去の声紋照合の成功でしか書き換わらないため、照合に失敗した発話で対応を
    壊さない。旧仕様（未確定に落とす）は実会話で同一人物の発話が未確定と人物Nに
    分裂し、1:1帰属精度を大きく下げていた（44%→継続化で54%→全体79%）。
    """
    vp = _tracker(_unit(1, 1, 0))            # AとBの中間（sim差が小さい）
    vp.sp_map["1"] = "B"                     # 直前はB（声紋照合で確定済み）
    assert vp.classify(_SHORT, "1", count=True) == "B"   # ラベル継続
    assert vp.last["kind"] == "ラベル継続"
    assert vp.sp_map["1"] == "B"             # マッピングは保持


def test_short_turn_closed_roster_keeps_label_continuation():
    """閉じた名簿(auto=False)でも、判別できない短い発話は登録者Bへのラベル継続.

    名簿確定モードは「登録済みの人 or 未確定」。Bは登録済みかつ声紋照合の成功で
    ラベルに結び付いた対応先なので、継続はポリシー内（未知/匿名への継続は別途
    未確定に落ちる）。
    """
    vp = _tracker(_unit(1, 1, 0))      # A/Bの中間で判別不能
    vp.auto = False                    # 名簿確定モード
    vp.sp_map["1"] = "B"               # 直前はB
    assert vp.classify(_SHORT, "1", count=True) == "B"   # ラベル継続
    assert vp.sp_map["1"] == "B"       # マッピングは保持


def test_short_turn_skipped_for_backchannel_keeps_label_continuation():
    """相槌(count=False)は短い厳格照合を通さず、ラベルの現在の対応を返す.

    相槌レコードの最終表示を未確定に落とす規則は RecvLoop.flush 側にある。
    ここでは声紋を触らず、ラベル状態（B）を維持することだけを保証する。
    """
    vp = _tracker(_unit(1, 0, 0))            # 声はAだが相槌なので照合しない
    vp.sp_map["1"] = "B"
    assert vp.classify(_SHORT, "1", count=False) == "B"
    assert vp.last["kind"] == "ラベル継続"
    assert vp.sp_map["1"] == "B"             # 相槌では動かさない


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


def test_short_turn_after_ai_echo_is_unsure_not_ai_key():
    """AI声紋一致直後の同ラベル短発話は __AI__ を継続せず未確定に落とす（F1）.

    sp_map に残った "__AI__" をラベル継続で返すと、_recv_loop の
    startswith("__") エコー破棄が発動して人間の短発話が本文ごと消える実バグ
    （2026-07-15 レビューで確定）。継続可否ガード（_continuation_target）は
    アクティブな人間プロファイルだけを継続対象とする。
    """
    from das.asr.live._constants import UNSURE_SPEAKER
    vp = _tracker(_unit(1, 1, 0))            # 判別できない短い声
    vp.profiles["__AI__"] = _unit(0, 0, 1)
    vp._active_keys.add("__AI__")
    vp.sp_map["1"] = "__AI__"                # 直前にAI声紋一致
    assert vp.classify(_SHORT, "1", count=True) == UNSURE_SPEAKER
    assert vp.last["kind"] == "継続不可"


def test_short_turn_after_deactivate_breaks_continuation():
    """deactivate 済み人物へのラベル継続は短発話でも切れる（F1）.

    メインパス（中尺）には従来からこのガードがあったが、短発話・相槌の
    継続経路には無く、無効化済み人物へ発話が帰属し続けていた。
    """
    from das.asr.live._constants import UNSURE_SPEAKER
    vp = _tracker(_unit(1, 1, 0))            # 判別できない短い声
    vp.sp_map["1"] = "A"
    vp._active_keys = {"B"}                  # A は deactivate 済み
    assert vp.classify(_SHORT, "1", count=True) == UNSURE_SPEAKER
    assert vp.last["kind"] == "継続不可"


def test_short_turn_overlapped_is_not_matched_or_corrected():
    """overlapped=True の短発話は声紋照合・補正（sp_map 書き換え）をしない（F2）.

    重なり音声の埋め込みはデタラメ（classify docstring）。中尺は「重なり
    スキップ」なのに短発話だけ照合まで走り、誤補正の穴だった（2026-07-15
    レビューで確定）。重なりは発話長によらずラベル継続へ落とす。
    """
    vp = _tracker(_unit(1, 0, 0))            # 声としては明確にA（=補正され得る声）
    vp.sp_map["1"] = "B"                     # 直前は声紋照合でB
    assert vp.classify(_SHORT, "1", count=True, overlapped=True) == "B"
    assert vp.last["kind"] == "ラベル継続"    # 「補正」「声紋一致」にならない
    assert vp.sp_map["1"] == "B"             # 書き換えない


def test_set_hybrid_sets_instance_attribute_only():
    """set_hybrid はインスタンス属性のみ設定し、クラス属性を汚染しない（F8）."""
    vp = _tracker(_unit(1, 0, 0))
    vp.set_hybrid(True)
    assert vp.__dict__.get("hybrid") is True
    assert VoiceProfiles.hybrid is False     # 既定値（クラス属性）は不変

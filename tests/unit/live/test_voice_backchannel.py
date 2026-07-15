"""課題④: 相槌を声紋の人物確定に使わない（count=False）テスト.

count=False の発話は声紋の照合・蓄積を一切行わず「ラベル継続」（そのSTTラベルの
現在の対応先＝声紋照合の成功で確定した人物 or #ラベル）を返す（2026-07-14,
eval/replay_attribution.py での再設計: 照合失敗でラベルの人物対応を破棄する旧仕様
は同一人物を #ラベルと人物Nに分裂させ 1:1帰属精度44%、継続化で54%→全体79%）。
相槌レコードの最終表示を未確定に落とす規則は RecvLoop.flush 側にある
（相槌は聞き手が打つ＝直前話者とは別人のことが多い）。

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
    vp._active_keys = set()
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


def test_backchannel_keeps_label_continuation():
    """相槌(count=False)は、ラベルの現在の人物対応をそのまま返す（ラベル継続）.

    対応先（松井）は過去の声紋照合の成功でしか書き換わらないため、根拠なしに
    直前話者へ寄せる旧「前話者追従」（実測28%で廃止）とは異なり、声の証拠に
    基づく最後の対応の維持。相槌レコードを未確定表示に落とす最終規則は
    RecvLoop.flush 側が持つ（本メソッドはラベル状態の管理に徹する）。
    """
    vp = _bare_tracker()
    # 「声紋照合で確定済み」＝アクティブな人間プロファイルが実在する状態を再現する
    # （継続可否ガードの統一（2026-07-15 F1）により、対応先がアクティブでなければ
    # 継続しない。旧テストのプロファイル無し状態は実運用で起こらない）。
    prof = {"松井": np.array([1.0, 0.0])}
    vp.profiles = dict(prof)
    vp._active_keys = {"松井"}
    vp.sp_map["1"] = "松井"     # ラベル1は直前まで松井（声紋照合で確定済み）
    wav = np.ones(int(SR * 2), dtype=np.float32)
    assert vp.classify(wav, "1", overlapped=False, count=False) == "松井"
    assert vp.last["kind"] == "ラベル継続"   # diag には判定種別を残す
    assert vp.sp_map["1"] == "松井"          # マッピングは保持
    assert vp.profiles.keys() == prof.keys()  # 相槌では声紋を触らない


def test_backchannel_regex_matches_common_aizuchi():
    for t in ["はい", "うん", "なるほど", "ええ", "そうですね"]:
        assert _BACKCHANNEL_RE.match(t), t
    for t in ["それは違うと思います", "コストが高いです"]:
        assert not _BACKCHANNEL_RE.match(t), t


def test_backchannel_after_ai_echo_is_unsure_not_ai_key():
    """AI声紋一致直後の同ラベル相槌は __AI__ を継続せず未確定に落とす（F1）.

    継続可否ガードの統一（2026-07-15）: "__AI__" を返すと _recv_loop の
    startswith("__") エコー破棄で人間の相槌が本文ごと消える。相槌経路にも
    メインパスと同じ _continuation_target 判定を適用する。
    """
    from das.asr.live._constants import UNSURE_SPEAKER
    vp = _bare_tracker()
    vp.profiles = {"__AI__": np.array([1.0, 0.0]), "松井": np.array([0.0, 1.0])}
    vp._active_keys = {"__AI__", "松井"}
    vp.sp_map["1"] = "__AI__"                # 直前にAI声紋一致
    wav = np.ones(int(SR * 2), dtype=np.float32)
    assert vp.classify(wav, "1", overlapped=False, count=False) == UNSURE_SPEAKER
    assert vp.last["kind"] == "継続不可"

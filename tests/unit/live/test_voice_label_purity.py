"""ラベル健全性によるラベル継続の門番（handoff §15.7）のユニットテスト.

背景: 高重なり会話ではSonioxが複数話者を同一STTラベルに混ぜることがある
（Chiba 0532 実測: 自動登録3人が全て同一ラベル発、ラベル継続の正解率22%が
誤帰属29%の主因）。直近の照合成功が複数人物に割れているラベルは「不純」と
みなし、そのラベルに基づく帰属（prev継続・#プレースホルダ）を未確定に落とす。
replay実測（2026-07-16_1723 移植GT）: 誤帰属 28%→8%（window=4）。
ラベルが健全なセッション（142016）は出力不変（79%維持）。
"""
from __future__ import annotations

import threading

import numpy as np

from das.asr.live._constants import SR, UNSURE_SPEAKER
from das.asr.live._voice_profiles import VoiceProfiles


def _unit(x, y, z):
    v = np.array([x, y, z], dtype=np.float32)
    return v / np.linalg.norm(v)


def _tracker(embs):
    """呼び出しごとに embs のベクトルを順に返すトラッカー（既知A/B）."""
    vp = VoiceProfiles.__new__(VoiceProfiles)
    vp._lock = threading.RLock()
    vp.sp_map = {}
    vp.label_hist = {}
    vp.profiles = {"A": _unit(1, 0, 0), "B": _unit(0, 1, 0)}
    vp._active_keys = {"A", "B"}
    vp.pool = []
    vp.label_embs = {}
    vp.own_sims = {}
    vp.same_sims, vp.diff_sims = [], []
    vp.embed_ms = []
    vp.counts = {}
    vp.last = None
    vp.n_anon = 0
    vp.min_sec = 1.0
    vp.short_floor = 0.45
    vp.short_bonus = 0.05
    vp.margin = 0.05
    vp.thresh = 0.5
    vp.consist = 0.62
    vp.model = "redimnet"
    vp.auto = False   # 自動登録経路を切り、照合とラベル継続だけを見る
    vp.max_human_speakers = None
    it = iter(embs)
    vp._embed = lambda wav: next(it)  # type: ignore[method-assign]
    return vp


_LONG = np.ones(int(SR * 4.0), dtype=np.float32)    # strict_sec 超の長尺
_AMBIG = _unit(1, 1, 0)                              # A/B中間（照合不成立）


def test_pure_label_keeps_continuation():
    """照合成功が単一人物に収束しているラベルは、従来どおり継続する."""
    vp = _tracker([_unit(1, 0, 0)] * 4 + [_AMBIG])
    for _ in range(4):
        assert vp.classify(_LONG, "1", count=True) == "A"
    # 5発話目は判別できない → ラベル1は健全（A×4）なので A を継続
    assert vp.classify(_LONG, "1", count=True) == "A"
    assert vp.last["kind"] == "未確定" or vp.last["kind"] == "ラベル継続"


def test_impure_label_falls_to_unsure():
    """照合成功がA/Bに割れているラベルは不純 → 継続せず未確定."""
    vp = _tracker([_unit(1, 0, 0), _unit(0, 1, 0),
                   _unit(1, 0, 0), _unit(0, 1, 0), _AMBIG])
    assert vp.classify(_LONG, "1", count=True) == "A"
    assert vp.classify(_LONG, "1", count=True) == "B"   # 同一ラベルで別人が成功
    assert vp.classify(_LONG, "1", count=True) == "A"
    assert vp.classify(_LONG, "1", count=True) == "B"
    # 判別できない発話: 旧仕様なら B を継続していたが、不純なので未確定
    assert vp.classify(_LONG, "1", count=True) == UNSURE_SPEAKER
    assert vp.last["kind"] == "ラベル不純"


def test_impure_label_does_not_poison_sp_map():
    """不純落ちは sp_map を書き換えない（収束すれば継続が自然に復活する）."""
    vp = _tracker([_unit(1, 0, 0), _unit(0, 1, 0),
                   _unit(1, 0, 0), _unit(0, 1, 0), _AMBIG])
    for _ in range(4):
        vp.classify(_LONG, "1", count=True)
    prev = vp.sp_map["1"]
    assert vp.classify(_LONG, "1", count=True) == UNSURE_SPEAKER
    assert vp.sp_map["1"] == prev   # 対応は維持（"?" で汚さない）


def test_window_zero_disables_gate():
    """label_purity_window=0 で旧挙動（無条件の継続）に戻せる."""
    vp = _tracker([_unit(1, 0, 0), _unit(0, 1, 0),
                   _unit(1, 0, 0), _unit(0, 1, 0), _AMBIG])
    vp.label_purity_window = 0
    for _ in range(4):
        vp.classify(_LONG, "1", count=True)
    assert vp.classify(_LONG, "1", count=True) == "B"   # 旧仕様: 直近対応Bを継続


def test_short_utterance_continuation_also_gated():
    """短発話経路のラベル継続にも同じ門番が効く."""
    short = np.ones(int(SR * 0.6), dtype=np.float32)
    vp = _tracker([_unit(1, 0, 0), _unit(0, 1, 0),
                   _unit(1, 0, 0), _unit(0, 1, 0), _AMBIG])
    for _ in range(4):
        vp.classify(_LONG, "1", count=True)
    assert vp.classify(short, "1", count=True) == UNSURE_SPEAKER
    assert vp.last["kind"] == "ラベル不純"


def test_impure_label_does_not_feed_enrollment():
    """不純ラベルの音声は登録プールに入らない（プロファイル汚染の防止, §15.8）.

    Chiba 0532 実測: 混載ラベル由来のプロファイルが人物1へ11回・人物2へ4回と
    交互に合流し、プロファイル自体が2人分の声で汚染された。不純ラベルは
    帰属だけでなく蓄積・登録からも外す。
    """
    vp = _tracker([_unit(1, 0, 0), _unit(0, 1, 0),
                   _unit(1, 0, 0), _unit(0, 1, 0), _AMBIG, _AMBIG])
    vp.auto = True   # 自動登録経路を有効化
    vp.enroll_min_total_chars = 45
    vp.enroll_win_sec = 1.5
    vp.enroll_consist_bonus = 0.08
    for _ in range(4):
        vp.classify(_LONG, "1", count=True, chars=30)
    assert not vp._label_pure("1")
    pool_before = len(vp.pool)
    vp.classify(_LONG, "1", count=True, chars=30)     # 不純ラベルの照合失敗発話
    assert vp.last["kind"] == "ラベル不純"            # 蓄積中ではなく不純落ち
    assert len(vp.pool) == pool_before                # プールに追加されない

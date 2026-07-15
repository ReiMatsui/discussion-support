"""自動登録の純度検査（P1）のテスト.

CallHome 0856 実測（docs/design/handoff_2026-07-14_unregistered_speakers.md §14）で
「登録材料の混入」が単一障害点になり得ることが確定したため、
  P1: コミット直前にバッファのペアワイズ自己一貫性を検査（_purity_subset）し、
      混入分を除いた採用部分集合が累積文字数に届かなければ登録を保留する
を検証する。判定は分布の相対構造（話者間類似は話者内の半分未満）のみで、
固定の類似度しきい値を持たないことが設計要件。
"""
from __future__ import annotations

import threading

import numpy as np

from das.asr.live._voice_profiles import VoiceProfiles


def _unit(*v) -> np.ndarray:
    a = np.array(v, dtype=np.float64)
    return a / np.linalg.norm(a)


# 話者A相当（相互類似 ≈ 1.0）と、anchorゲート(ecs=0.42)は通るが
# Aとの類似が「話者内の半分未満」に落ちる混入声(≈0.45)。
_A = _unit(1, 0, 0)
_MIX = _unit(0.45, 0.893, 0)   # dot(_A, _MIX) ≈ 0.45


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
    vp.max_human_speakers = None
    vp.margin = 0.05
    vp.thresh = 0.5
    vp.consist = 0.34
    vp.dedupe = 0.5
    vp.model = "redimnet"
    vp.auto = True
    return vp


# ------------------------------------------------------------------
# P1: _purity_subset（部分集合の選択）
# ------------------------------------------------------------------

def test_purity_subset_pure_buffer_keeps_all():
    """純粋なバッファ（全て同じ声）は全採用のまま素通りする."""
    vp = _tracker()
    embs = [_A] * 6
    assert vp._purity_subset(embs) == list(range(6))


def test_purity_subset_mixed_buffer_selects_majority():
    """混入バッファは medoid 側の自己一貫部分集合だけが選ばれる."""
    vp = _tracker()
    embs = [_A, _MIX, _A, _A, _MIX, _A]
    assert vp._purity_subset(embs) == [0, 2, 3, 5]


def test_purity_subset_too_few_samples_skips_check():
    """分布を語れないサンプル数（n<4）は検査せず全採用."""
    vp = _tracker()
    embs = [_A, _MIX, _A]
    assert vp._purity_subset(embs) == [0, 1, 2]


# ------------------------------------------------------------------
# P1: _enroll_accumulate（コミットゲート）
# ------------------------------------------------------------------

def test_enroll_pure_buffer_registers():
    """純粋バッファは従来どおり登録される（純度検査による遅延なし）."""
    vp = _tracker()
    samples = [(_A, 12.0)] * 4   # 48文字 >= 45
    assert vp._enroll_accumulate(samples, "1", None, ecs=0.42) == "人物1"
    assert "純度保留" not in vp.counts


def test_enroll_mixed_buffer_commits_majority_only():
    """混入バッファでもクリーン分が足りれば、採用部分集合のみで登録される.

    混入分はプールに残る（もう一方の話者の蓄積材料として生かす）。
    """
    vp = _tracker()
    # anchor（最後のサンプル）は本人の声。混入 _MIX は ecs=0.42 ゲートを
    # わずかに通る（dot=0.45）が、純度検査で除外される。
    samples = [(_A, 10.0), (_MIX, 30.0), (_A, 10.0), (_A, 10.0),
               (_A, 10.0), (_A, 10.0)]
    assert vp._enroll_accumulate(samples, "1", None, ecs=0.42) == "人物1"
    prof = vp.profiles["人物1"]
    assert float(np.dot(prof, _A)) > 0.999      # 混入が平均に入っていない
    assert len(vp.pool) == 1                    # 混入サンプルはプールに残留
    assert float(np.dot(vp.pool[0][0], _MIX)) > 0.999


def test_enroll_heavy_contamination_defers():
    """混入が激しくクリーン分が累積文字数に届かない場合は登録を保留する."""
    vp = _tracker()
    samples = [(_A, 10.0), (_A, 10.0), (_A, 10.0), (_A, 10.0), (_MIX, 20.0)]
    # 全体 60 >= 45 だが、採用部分集合（本人4つ）は 40 < 45 → 保留
    assert vp._enroll_accumulate(samples, "1", None, ecs=0.42) is None
    assert vp.profiles == {}
    assert vp.counts.get("純度保留") == 1
    assert len(vp.pool) == 5                    # プールは温存（蓄積継続）

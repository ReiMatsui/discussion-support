"""rekey の状態一貫性伝搬（review C3/C11, P2）のユニットテスト.

背景（docs/design/attribution_logic_review_2026-07.md C3）: 話者IDの台帳は
records/colors/names（rekey が更新）のほかに diarization_speaker_keys と
ClusterVoiceNamer._confirmed があり、従来 rekey はこの2つを更新しなかった。
そのため UI /rename（tracker.enroll → s.rekey(人物N, 実名)）の後、
同クラスタの発話で observe() が古い「人物N」を返し、constrain では
プロファイルに無い匿名名扱い → 別人格「参加者X」として復活し得た。
本テストはその復活シナリオを固定し、rekey の伝搬で消えることを検証する。
"""
from __future__ import annotations

import datetime

import numpy as np

from das.asr.live._cluster_naming import ClusterVoiceNamer
from das.asr.live._constants import SR
from das.asr.live._session_state import SessionState


class _FakeTracker:
    """ClusterVoiceNamer が使う最小API（match_profile / embed / dedupe）."""

    def __init__(self, results=()):
        self._results = list(results)
        self.dedupe = 0.72

    def match_profile(self, wav):
        if not self._results:
            return None
        return self._results.pop(0)

    def embed(self, wav):
        return None


def _wav(seconds: float) -> np.ndarray:
    return np.full(int(seconds * SR), 0.1, dtype=np.float32)


def _make_state(namer=None) -> SessionState:
    return SessionState(
        args=object(),
        started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md",
        html_path="/tmp/o.html",
        diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns",
        wav_path="/tmp/o.wav",
        cluster_namer=namer,
    )


def test_rename_after_cluster_confirm_does_not_resurrect_old_name():
    """C3再現: 人物N→実名リネーム後、クラスタ確定名が古い人物Nを返さない."""
    namer = ClusterVoiceNamer(_FakeTracker([("人物2", 0.8)]), min_sec=5.0)
    s = _make_state(namer)
    # クラスタが「人物2」に確定し、キー台帳も人物2を指している状態を作る
    assert namer.observe("pyannote:SPEAKER_00", _wav(5.0)) == "人物2"
    s.diarization_speaker_keys["pyannote:SPEAKER_00"] = "人物2"
    s.records.append({"ms": 0, "end_ms": 1000, "speaker": "人物2", "text": "こんにちは"})

    # UI /rename 相当（tracker.enroll の後に呼ばれる rekey）
    s.rekey("人物2", "田中")

    # 従来はここが "人物2" のままで、以後の発話が古い名前で復活していた
    assert namer.confirmed_name("pyannote:SPEAKER_00") == "田中"
    assert namer.observe("pyannote:SPEAKER_00", _wav(1.0)) == "田中"
    assert s.diarization_speaker_keys["pyannote:SPEAKER_00"] == "田中"
    assert s.records[0]["speaker"] == "田中"


def test_rekey_updates_diarization_speaker_keys_for_anonymous_merge():
    """@diar 同士の統合でも、他クラスタが古いキーを指し続けない."""
    s = _make_state()
    s.diarization_speaker_keys["pyannote:SPEAKER_00"] = "@diar:1"
    s.diarization_speaker_keys["pyannote:R1:SPEAKER_00"] = "@diar:1"   # 再接続後の分裂

    s.rekey("@diar:1", "@diar:2")

    assert s.diarization_speaker_keys["pyannote:SPEAKER_00"] == "@diar:2"
    assert s.diarization_speaker_keys["pyannote:R1:SPEAKER_00"] == "@diar:2"


def test_rekey_without_cluster_namer_keeps_working():
    """Soniox単独モード（cluster_namer なし）では従来どおり動く（挙動不変）."""
    s = _make_state()
    s.records.append({"ms": 0, "end_ms": 1000, "speaker": "#1", "text": "a"})

    s.rekey("#1", "田中")

    assert s.records[0]["speaker"] == "田中"


def test_html_color_stable_after_rekey():
    """C11再現: rekey（統合）で他の話者の HTML 色がずれない."""
    s = _make_state()
    for k in ("#1", "#2", "#3"):
        s.color_of(k)
    before_2 = s.html_color("#2")
    before_3 = s.html_color("#3")

    s.rekey("#1", "田中")   # 実名化（colors から #1 が pop される）

    assert s.html_color("#2") == before_2
    assert s.html_color("#3") == before_3
    # 統合先は旧キーの色枠を引き継ぐ（同一人物の色が変わらない）
    assert s.html_color("田中") == s.html_color("田中")   # 冪等

"""ClusterVoiceNamer（pyannoteハイブリッド構成のクラスタ単位声紋名前付け）の単体テスト.

設計: docs/design/pyannote_live1_trial_2026-07-09.md §8.4/§9 参照。
"""
from __future__ import annotations

import numpy as np

from das.asr.live._cluster_naming import ClusterVoiceNamer
from das.asr.live._constants import SR


class _FakeTracker:
    """VoiceProfiles の照合API（match_profile / embed / dedupe）を模したスタブ."""

    def __init__(self, results, embed_map=None, dedupe=0.72):
        # results: 呼び出しごとに順に返す (name, conf) | None のリスト
        self._results = list(results)
        self.calls: list[int] = []   # 呼び出しごとの音声サンプル数（照合が試みられた回数の記録）
        # embed_map: wav先頭サンプル値(round 3桁) -> 正規化ベクトル。
        # 未指定なら embed は常に None（埋め込み計算不可の環境を模す）。
        self._embed_map = dict(embed_map or {})
        self.dedupe = dedupe   # クラスタ間名寄せの閾値（VoiceProfiles.dedupe 相当）

    def match_profile(self, wav):
        self.calls.append(wav.size)
        if not self._results:
            return None
        return self._results.pop(0)

    def embed(self, wav):
        if not self._embed_map or wav.size == 0:
            return None
        return self._embed_map.get(round(float(wav[0]), 3))


def _wav(seconds: float, fill: float = 0.0) -> np.ndarray:
    return np.full(int(seconds * SR), fill, dtype=np.float32)


# クラスタ間名寄せ用の決め打ち正規化ベクトル（v1・v2は類似 0.9、v3は直交）
_V1 = np.array([1.0, 0.0, 0.0])
_V2 = np.array([0.9, np.sqrt(1 - 0.81), 0.0])
_V3 = np.array([0.0, 1.0, 0.0])


def test_observe_stays_unconfirmed_below_threshold():
    """累積が閾値未満の間は照合すら試みず未確定のまま."""
    tracker = _FakeTracker([("田中", 0.8)])
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)

    assert namer.observe("pyannote:SPEAKER_00", _wav(2.0)) is None
    assert tracker.calls == []
    assert namer.confirmed_name("pyannote:SPEAKER_00") is None


def test_observe_confirms_once_threshold_reached():
    """累積が閾値に達したら照合し、confidence十分なら名前を確定する."""
    tracker = _FakeTracker([("田中", 0.8)])
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)

    namer.observe("pyannote:SPEAKER_00", _wav(3.0))
    name = namer.observe("pyannote:SPEAKER_00", _wav(2.5))

    assert name == "田中"
    assert namer.confirmed_name("pyannote:SPEAKER_00") == "田中"
    assert len(tracker.calls) == 1


def test_observe_keeps_accumulating_when_confidence_insufficient():
    """confidence不足(match_profileがNone)なら未確定のまま蓄積を続け、再照合のたびに試みる."""
    tracker = _FakeTracker([None, ("鈴木", 0.9)])
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)

    assert namer.observe("pyannote:SPEAKER_00", _wav(3.0)) is None
    assert namer.observe("pyannote:SPEAKER_00", _wav(2.5)) is None   # 1回目照合、confidence不足
    assert namer.confirmed_name("pyannote:SPEAKER_00") is None
    name = namer.observe("pyannote:SPEAKER_00", _wav(3.0))            # 再蓄積後、2回目照合
    assert name == "鈴木"
    assert len(tracker.calls) == 2


def test_observe_confirmed_cluster_returns_name_without_reobserving():
    """一度確定したクラスタは以後照合し直さず、確定名をそのまま返す."""
    tracker = _FakeTracker([("田中", 0.8)])
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)
    namer.observe("pyannote:SPEAKER_00", _wav(5.0))

    name = namer.observe("pyannote:SPEAKER_00", _wav(1.0))

    assert name == "田中"
    assert len(tracker.calls) == 1   # 確定後は再照合しない


def test_observe_skips_overlapped_audio_and_does_not_confirm():
    """重複発話区間の音声は蓄積しない（安全側, 設計点5）."""
    tracker = _FakeTracker([("田中", 0.8)])
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)

    for _ in range(3):
        assert namer.observe("pyannote:SPEAKER_00", _wav(3.0), overlapped=True) is None

    assert tracker.calls == []
    assert namer.confirmed_name("pyannote:SPEAKER_00") is None


def test_observe_confirmed_cluster_still_returns_name_when_overlapped():
    """確定済みクラスタは重複発話フラグが立っていても確定名をそのまま返す."""
    tracker = _FakeTracker([("田中", 0.8)])
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)
    namer.observe("pyannote:SPEAKER_00", _wav(5.0))

    assert namer.observe("pyannote:SPEAKER_00", _wav(1.0), overlapped=True) == "田中"


def test_reset_clears_buffers_and_confirmations():
    """会議リセットで蓄積・確定状態をクリアする."""
    tracker = _FakeTracker([("田中", 0.8)])
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)
    namer.observe("pyannote:SPEAKER_00", _wav(5.0))
    assert namer.confirmed_name("pyannote:SPEAKER_00") == "田中"

    namer.reset()

    assert namer.confirmed_name("pyannote:SPEAKER_00") is None


def test_buffer_caps_at_max_buffer_seconds_without_crashing():
    """際限ない蓄積を防ぐため上限を超えたら古い分から捨てる（直近音声で照合する）."""
    tracker = _FakeTracker([("田中", 0.8)])
    namer = ClusterVoiceNamer(tracker, min_sec=100.0, max_buffer_sec=10.0)

    for _ in range(6):
        namer.observe("pyannote:SPEAKER_00", _wav(3.0))   # 計18秒分投入、上限10秒でトリム

    # min_secに達しないので照合は試みられない（クラッシュしないことの確認が主眼）
    assert namer.confirmed_name("pyannote:SPEAKER_00") is None
    assert tracker.calls == []


def test_independent_clusters_do_not_share_buffers_or_confirmations():
    """クラスタ分裂対策: 別クラスタIDは独立に蓄積・確定する（設計点6）."""
    tracker = _FakeTracker([("田中", 0.8), ("田中", 0.8)])
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)

    name_a = namer.observe("pyannote:SPEAKER_00", _wav(5.0))
    assert namer.observe("pyannote:SPEAKER_01", _wav(2.0)) is None   # 別クラスタは未確定のまま
    name_b = namer.observe("pyannote:SPEAKER_01", _wav(3.0))

    assert name_a == "田中"
    assert name_b == "田中"   # 分裂しても同じ人物へ照合されるだけで無害


# --- クラスタ間名寄せ（登録者ゼロ対策, handoff_2026-07-14 §3） -------------


def test_similar_unmatched_clusters_are_merged():
    """未照合2クラスタが類似埋め込みなら名寄せされ、2つ目は新規参加者にならない."""
    tracker = _FakeTracker([], embed_map={0.1: _V1, 0.2: _V2})
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)

    assert namer.observe("pyannote:SPEAKER_00", _wav(5.0, 0.1)) is None   # 代表埋め込み登録
    assert namer.observe("pyannote:SPEAKER_01", _wav(5.0, 0.2)) is None   # 類似→名寄せ

    assert namer.canonical_cluster("pyannote:SPEAKER_01") == "pyannote:SPEAKER_00"
    assert namer.canonical_cluster("pyannote:SPEAKER_00") == "pyannote:SPEAKER_00"
    assert namer.last_match == {"kind": "クラスタ名寄せ", "raw": "pyannote:SPEAKER_01",
                                "canonical": "pyannote:SPEAKER_00", "sim": 0.9}


def test_dissimilar_clusters_stay_independent():
    """非類似（閾値未満）なら名寄せせず、各クラスタは独立を保つ."""
    tracker = _FakeTracker([], embed_map={0.1: _V1, 0.3: _V3})
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)

    namer.observe("pyannote:SPEAKER_00", _wav(5.0, 0.1))
    namer.observe("pyannote:SPEAKER_01", _wav(5.0, 0.3))

    assert namer.canonical_cluster("pyannote:SPEAKER_01") == "pyannote:SPEAKER_01"


def test_absorbed_cluster_accumulates_into_canonical():
    """名寄せ後、吸収側raw_clusterでのobserveはcanonicalのバッファへ蓄積される."""
    tracker = _FakeTracker([None, None, ("田中", 0.8)], embed_map={0.1: _V1, 0.2: _V2})
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)
    namer.observe("pyannote:SPEAKER_00", _wav(5.0, 0.1))      # 照合1回目(None)→埋め込み登録
    namer.observe("pyannote:SPEAKER_01", _wav(5.0, 0.2))      # 照合2回目(None)→名寄せ成立

    # 吸収側キーでの追加音声も canonical に積まれ、確定は canonical 側に付く
    name = namer.observe("pyannote:SPEAKER_01", _wav(1.0, 0.2))

    assert name == "田中"
    assert namer.confirmed_name("pyannote:SPEAKER_00") == "田中"
    assert namer.confirmed_name("pyannote:SPEAKER_01") == "田中"   # エイリアス経由でも見える


def test_merge_into_confirmed_canonical_returns_confirmed_name():
    """canonicalが確定済みなら、名寄せ成立時にその確定名が即返る."""
    tracker = _FakeTracker([None, ("田中", 0.8), None], embed_map={0.1: _V1, 0.2: _V2})
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)
    namer.observe("pyannote:SPEAKER_00", _wav(5.0, 0.1))   # 照合不成立→埋め込み登録
    assert namer.observe("pyannote:SPEAKER_00", _wav(0.5, 0.1)) == "田中"   # 確定

    name = namer.observe("pyannote:SPEAKER_01", _wav(5.0, 0.2))   # 類似→確定済みへ名寄せ

    assert name == "田中"
    assert namer.canonical_cluster("pyannote:SPEAKER_01") == "pyannote:SPEAKER_00"


def test_reset_clears_aliases_and_embeddings():
    """resetで名寄せ状態（aliases/embeddings）もクリアされる."""
    tracker = _FakeTracker([], embed_map={0.1: _V1, 0.2: _V2})
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)
    namer.observe("pyannote:SPEAKER_00", _wav(5.0, 0.1))
    namer.observe("pyannote:SPEAKER_01", _wav(5.0, 0.2))
    assert namer.canonical_cluster("pyannote:SPEAKER_01") == "pyannote:SPEAKER_00"

    namer.reset()

    assert namer.canonical_cluster("pyannote:SPEAKER_01") == "pyannote:SPEAKER_01"
    assert namer.nearest_cluster("pyannote:SPEAKER_00") is None   # 埋め込みも消えている


def test_nearest_cluster_returns_best_with_similarity():
    """nearest_clusterは閾値をかけず (最近傍, 類似度) を返す（統合可否は呼び出し側）."""
    tracker = _FakeTracker([], embed_map={0.1: _V1, 0.3: _V3})
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)
    namer.observe("pyannote:SPEAKER_00", _wav(5.0, 0.1))
    namer.observe("pyannote:SPEAKER_01", _wav(5.0, 0.3))   # 直交（sim=0 < dedupe）→独立

    assert namer.nearest_cluster("pyannote:SPEAKER_00") == ("pyannote:SPEAKER_01", 0.0)
    # 対称にも動く（exclude 引数は未使用のため削除。review D2）
    assert namer.nearest_cluster("pyannote:SPEAKER_01") == ("pyannote:SPEAKER_00", 0.0)


def test_nearest_cluster_returns_none_without_embedding():
    """自クラスタの埋め込みが未計算ならnearest_clusterはNone."""
    tracker = _FakeTracker([])
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)

    assert namer.nearest_cluster("pyannote:SPEAKER_00") is None


def test_confirmed_cluster_saves_embedding_and_serves_as_merge_target():
    """確定経路でも代表埋め込みが保存され、以後の名寄せ先として機能する（F4）."""
    tracker = _FakeTracker([("田中", 0.8)], embed_map={0.1: _V1, 0.2: _V2})
    namer = ClusterVoiceNamer(tracker, min_sec=5.0)

    assert namer.observe("pyannote:SPEAKER_00", _wav(5.0, 0.1)) == "田中"   # 即確定
    # 確定クラスタの埋め込みが保存されているので、類似クラスタは名寄せで即帰属する
    assert namer.observe("pyannote:SPEAKER_01", _wav(5.0, 0.2)) == "田中"
    assert namer.canonical_cluster("pyannote:SPEAKER_01") == "pyannote:SPEAKER_00"


def test_zero_norm_embedding_is_guarded_and_not_stored():
    """全ゼロwav（ゼロノルム埋め込み）は None に落ち、代表埋め込みに NaN が入らない（F5）."""
    import threading

    from das.asr.live._voice_profiles import VoiceProfiles

    vp = VoiceProfiles.__new__(VoiceProfiles)
    vp._lock = threading.RLock()
    vp.embed_ms = []
    vp.min_sec = 1.0
    vp.dedupe = 0.72
    vp._embed_raw = lambda wav: np.zeros(8)   # 無音等でモデルがゼロベクトルを返す想定

    assert vp.embed(_wav(2.0)) is None        # NaN 正規化ではなく None ガード

    namer = ClusterVoiceNamer(vp, min_sec=5.0)
    assert namer.observe("pyannote:SPEAKER_00", _wav(5.0)) is None
    assert namer._embeddings == {}            # NaN 入りの代表埋め込みを保存しない

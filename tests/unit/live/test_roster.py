"""人数を教えられていないときに参加者の集合を自分で作る部分のテスト.

守るべき性質:

  - 明らかに分かれている声はちゃんと分かれる
  - 人数は推定する（上限として与えると序盤に別人を混ぜて壊れるため。§41.1）
  - 群が1つしか無い答えは選ばない（「全部同じ人」に落ちない）
  - scipy の平均連結法と同じ分け方になる（eval 側と食い違わせない）
"""
from __future__ import annotations

import numpy as np
import pytest

from das.asr.live import _roster as roster


def _unit(v) -> np.ndarray:
    a = np.array(v, dtype=np.float64)
    return a / np.linalg.norm(a)


def _people(n_each: int = 4, jitter: float = 0.05, seed: int = 0):
    """3人ぶんの声紋を作る（同じ人は近く、別人は遠い）."""
    rng = np.random.default_rng(seed)
    base = [_unit([1, 0, 0, 0]), _unit([0, 1, 0, 0]), _unit([0, 0, 1, 0])]
    embs, who = [], []
    for i, b in enumerate(base):
        for _ in range(n_each):
            v = b + rng.normal(0, jitter, size=4)
            embs.append(v / np.linalg.norm(v))
            who.append(i)
    return np.array(embs), np.array(who)


def _same_partition(a, b) -> bool:
    """群番号の付け方が違っても、分け方が同じなら True."""
    def parts(x):
        return {frozenset(np.where(x == v)[0]) for v in np.unique(x)}
    return parts(a) == parts(b)


# ---------------------------------------------------------- 平均連結法


def test_clearly_separate_voices_are_split() -> None:
    emb, who = _people()
    lab = roster.average_linkage(emb @ emb.T, 3)
    assert _same_partition(lab, who)


def test_asking_for_one_group_returns_one_group() -> None:
    emb, _who = _people()
    assert len(set(roster.average_linkage(emb @ emb.T, 1))) == 1


def test_more_groups_than_points_gives_each_point_its_own() -> None:
    emb, _who = _people(n_each=1)
    lab = roster.average_linkage(emb @ emb.T, 10)
    assert len(set(lab)) == len(emb)


# ---------------------------------------------------------- 人数の推定


def test_the_number_of_people_is_estimated() -> None:
    """人数を渡さなくても3つに分かれる（推定させるのが既定。§41.1）."""
    emb, who = _people()
    assert _same_partition(roster.cluster(emb), who)


def test_a_single_group_answer_is_never_chosen() -> None:
    """まとまりが1つだけの答えは選ばない（全員同じ人にしない）."""
    emb, _who = _people()
    assert len(set(roster.cluster(emb))) >= 2


def test_one_utterance_is_left_alone() -> None:
    emb, _who = _people(n_each=1)
    assert list(roster.cluster(emb[:1])) == [0]


def test_silhouette_says_worst_for_a_single_group() -> None:
    emb, _who = _people()
    sim = emb @ emb.T
    assert roster.silhouette(sim, np.zeros(len(emb), dtype=int)) == -1.0
    assert roster.silhouette(sim, roster.average_linkage(sim, 3)) > 0.5


# ------------------------------------------------------------ 重心


def test_centroids_are_unit_length_and_in_group_order() -> None:
    emb, _who = _people()
    lab = roster.average_linkage(emb @ emb.T, 3)
    cen = roster.centroids(emb, lab)
    assert cen.shape == (3, 4)
    assert np.allclose(np.linalg.norm(cen, axis=1), 1.0)
    # 各群の重心は、その群の点といちばん似ている
    for i, v in enumerate(emb):
        assert int((cen @ v).argmax()) == lab[i]


def test_centroids_of_nothing_is_empty() -> None:
    emb, _who = _people()
    cen = roster.centroids(emb[:0], np.zeros(0, dtype=int))
    assert cen.shape[0] == 0


# --------------------------------------------- eval 側（scipy）との一致


def test_matches_scipy_average_linkage() -> None:
    """scipy と同じ分け方になる.

    eval 側（`roster_free.py`）は scipy で測っており、本番は numpy で動く。
    ここがずれると「測った数字」と「動く実装」が別物になる（§29 で一度
    やった失敗）。
    """
    sch = pytest.importorskip("scipy.cluster.hierarchy")
    ssd = pytest.importorskip("scipy.spatial.distance")
    for seed in range(3):
        emb, _who = _people(n_each=5, jitter=0.25, seed=seed)
        sim = emb @ emb.T
        d = np.clip(1.0 - sim, 0, None)
        np.fill_diagonal(d, 0.0)
        z = sch.linkage(ssd.squareform((d + d.T) / 2, checks=False),
                        method="average")
        for k in (2, 3, 4, 5):
            mine = roster.average_linkage(sim, k)
            theirs = sch.fcluster(z, k, criterion="maxclust")
            assert _same_partition(mine, theirs), f"seed={seed} k={k}"

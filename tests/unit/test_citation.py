"""citation_rate 計算のテスト。"""

from __future__ import annotations

import pytest

from das.eval.citation import (
    aggregate_citation_stats,
    compute_citation_stats,
    is_cited,
)
from das.types import Utterance


# --- is_cited basic --------------------------------------------------


def test_is_cited_when_target_repeats_source_substantially() -> None:
    src = "X 大学では紙容器導入後 2 年目にコストが解消された"
    target = "X 大学では紙容器導入後 2 年目にコストが解消されたという事例がある"
    assert is_cited(src, target) is True


def test_is_cited_returns_false_for_unrelated_texts() -> None:
    src = "プラスチック容器のコストは増加する"
    target = "今日は天気が良い"
    assert is_cited(src, target) is False


def test_is_cited_threshold_respected() -> None:
    """ほんのわずかしか被ってないなら False。"""

    src = "とても専門的な内容"
    target = "天気の話"
    assert is_cited(src, target, coverage_threshold=0.5) is False


# --- compute_citation_stats ------------------------------------------


def _utts(speakers: list[str], texts: list[str]) -> list[Utterance]:
    return [
        Utterance(turn_id=i + 1, speaker=s, text=t)
        for i, (s, t) in enumerate(zip(speakers, texts, strict=True))
    ]


def test_no_interventions_yields_zero() -> None:
    transcript = _utts(["A", "B"], ["主張 1", "反論 1"])
    stats = compute_citation_stats(transcript, [])
    assert stats.n_items_presented == 0
    assert stats.overall_rate == 0.0


def test_l1_intervention_cited_in_next_utterance_by_addressee() -> None:
    """L1 で提示した文書が、次の addressed_to 発話で再現されると引用済とカウント。"""

    transcript = _utts(
        ["A", "B", "A"],
        [
            "プラ容器を廃止すべき",
            "コストが懸念",
            "X 大学では紙容器導入後 2 年目にコストが解消されている",
        ],
    )
    interventions = [
        {
            "kind": "l1",
            "turn_id": 2,
            "addressed_to": "A",
            "items": [
                {
                    "source_text": "X 大学では紙容器導入後 2 年目にコストが解消された",
                    "source_kind": "document",
                }
            ],
        }
    ]
    stats = compute_citation_stats(transcript, interventions)
    assert stats.n_items_presented == 1
    assert stats.n_items_cited == 1
    assert "document" in stats.by_kind
    assert stats.by_kind["document"].rate == 1.0


def test_l1_not_cited_returns_zero_rate() -> None:
    transcript = _utts(
        ["A", "B", "A"],
        ["主張", "反論", "全く別の話題に逸れる"],
    )
    interventions = [
        {
            "kind": "l1",
            "turn_id": 2,
            "addressed_to": "A",
            "items": [
                {"source_text": "X 大学のコスト構造分析", "source_kind": "document"}
            ],
        }
    ]
    stats = compute_citation_stats(transcript, interventions)
    assert stats.n_items_presented == 1
    assert stats.n_items_cited == 0


def test_skip_interventions_ignored() -> None:
    transcript = _utts(["A", "B"], ["a", "b"])
    interventions = [
        {"kind": "skip", "turn_id": 1, "items": []},
    ]
    stats = compute_citation_stats(transcript, interventions)
    assert stats.n_items_presented == 0


def test_by_kind_breakdown() -> None:
    """source_kind 別に集計される。"""

    transcript = _utts(
        ["A", "B", "A", "B"],
        [
            "主張",
            "反論",
            "X 大学の事例について言及する",  # cited document
            "そういえば自治体の Web 記事もありますね",  # not citing the web item
        ],
    )
    interventions = [
        {
            "kind": "l1",
            "turn_id": 2,
            "addressed_to": "A",
            "items": [
                {
                    "source_text": "X 大学の事例における 2 年目の費用構造",
                    "source_kind": "document",
                }
            ],
        },
        {
            "kind": "l1",
            "turn_id": 3,
            "addressed_to": "B",
            "items": [
                {
                    "source_text": "シンガポールの環境局による報告書",
                    "source_kind": "web",
                }
            ],
        },
    ]
    stats = compute_citation_stats(transcript, interventions)
    assert stats.n_items_presented == 2
    assert stats.by_kind["document"].n_items_cited == 1
    assert stats.by_kind["web"].n_items_cited == 0


def test_aggregate_combines_runs() -> None:
    s1 = compute_citation_stats(
        _utts(["A", "A"], ["x", "提示テキストの再現"]),
        [
            {
                "kind": "l1",
                "turn_id": 1,
                "addressed_to": "A",
                "items": [
                    {"source_text": "提示テキストの再現", "source_kind": "document"}
                ],
            }
        ],
    )
    s2 = compute_citation_stats(
        _utts(["B", "B"], ["y", "全く別"]),
        [
            {
                "kind": "l1",
                "turn_id": 1,
                "addressed_to": "B",
                "items": [{"source_text": "z", "source_kind": "web"}],
            }
        ],
    )
    agg = aggregate_citation_stats([s1, s2])
    assert agg["n_runs"] == 2
    assert agg["n_items_presented_total"] == 2
    assert agg["n_items_cited_total"] == 1
    assert agg["overall_rate"] == pytest.approx(0.5)

"""合意検出 (consensus.py) と SessionRunner 早期終了のテスト。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from das.eval.consensus import detect_consensus
from das.eval.controller import SessionConfig, SessionRunner
from das.eval.persona import build_persona
from das.graph.schema import Edge, Node
from das.graph.store import NetworkXGraphStore
from das.llm import OpenAIClient
from das.types import Utterance


def _utterances(speakers: list[str], texts: list[str]) -> list[Utterance]:
    return [
        Utterance(turn_id=i + 1, speaker=s, text=t)
        for i, (s, t) in enumerate(zip(speakers, texts, strict=True))
    ]


# --- detect_consensus ----------------------------------------------------


def test_no_consensus_with_short_transcript() -> None:
    transcript = _utterances(["A", "B"], ["x", "y"])
    report = detect_consensus(transcript)
    assert report.consensus_reached is False
    assert report.signal == "none"


def test_no_consensus_when_disagreeing() -> None:
    transcript = _utterances(
        ["A", "B", "C", "A", "B"],
        [
            "プラ容器を廃止すべき",
            "コストが高すぎる",
            "別の選択肢もある",
            "それでもやはり廃止が筋",
            "コスト懸念は無視できない",
        ],
    )
    report = detect_consensus(transcript)
    assert report.consensus_reached is False


def test_explicit_agreement_detected() -> None:
    """直近 3 ターンの 60% 以上が合意キーワードを含むと検出される。"""

    transcript = _utterances(
        ["A", "B", "C", "A", "B", "C"],
        [
            "プラ容器を廃止すべき",
            "コストの懸念がある",
            "折衷案が良いかも",
            "なるほど、それは確かに納得できます",
            "賛成です。歩み寄りましょう",
            "私もその通りだと思います",
        ],
    )
    report = detect_consensus(transcript)
    assert report.consensus_reached is True
    assert report.signal == "explicit_agreement"
    assert report.detected_at_turn == 6


def test_structural_signal_with_store() -> None:
    """合意キーワードはなくても、新規 claim と attack が止まれば構造合意を返す。"""

    store = NetworkXGraphStore()
    # 旧ターン (turn_id=1) に claim
    n1 = Node(
        text="主張1",
        node_type="claim",
        source="utterance",
        author="A",
        metadata={"turn_id": 1},
    )
    n2 = Node(
        text="主張2",
        node_type="claim",
        source="utterance",
        author="B",
        metadata={"turn_id": 2},
    )
    store.add_node(n1)
    store.add_node(n2)
    store.add_edge(Edge(src_id=n2.id, dst_id=n1.id, relation="attack", confidence=0.8))

    transcript = _utterances(
        ["A", "B", "A", "B", "C", "A"],
        [
            "主張1",
            "主張2",
            "では",
            "話を整理しよう",
            "中立",
            "終わりかな",
        ],
    )
    # 直近 4 ターンに対応する claim 新規ノードが store にない & 攻撃も無い
    report = detect_consensus(transcript, store=store, stall_window=4)
    assert report.consensus_reached is True
    assert "new_claim_stalled" in report.fired_signals
    assert "no_new_attacks" in report.fired_signals


# --- SessionRunner 早期終了 -----------------------------------------------


def _fake_llm() -> OpenAIClient:
    client = OpenAIClient(client=MagicMock())
    client.chat = AsyncMock(return_value="発言")  # type: ignore[method-assign]
    return client


async def test_run_streaming_stops_on_consensus() -> None:
    """stop_condition が True を返したら max_turns 未満でも終了する。"""

    runner = SessionRunner(
        [build_persona(name="A"), build_persona(name="B")],
        SessionConfig(topic="t", max_turns=10, temperature=0.0),
        llm=_fake_llm(),
    )

    def stop_at_3(history: list[Utterance]) -> bool:
        return len(history) >= 3

    transcript = await runner.run(stop_condition=stop_at_3)
    assert len(transcript) == 3


async def test_run_streaming_runs_full_when_no_stop() -> None:
    runner = SessionRunner(
        [build_persona(name="A")],
        SessionConfig(topic="t", max_turns=4, temperature=0.0),
        llm=_fake_llm(),
    )
    transcript = await runner.run()
    assert len(transcript) == 4


async def test_run_streaming_stop_using_detect_consensus() -> None:
    """detect_consensus を stop_condition に使った統合動作。"""

    llm = _fake_llm()
    # 4 ターン目から合意キーワードを並べる
    replies = [
        "プラ容器を廃止すべき",
        "コストが高い",
        "折衷案",
        "なるほど納得です",
        "賛成です",
        "その通りです",
    ]
    llm.chat = AsyncMock(side_effect=replies)  # type: ignore[method-assign]

    runner = SessionRunner(
        [
            build_persona(name="A"),
            build_persona(name="B"),
            build_persona(name="C"),
        ],
        SessionConfig(topic="t", max_turns=10, temperature=0.0),
        llm=llm,
    )

    def stop(history: list[Utterance]) -> bool:
        return detect_consensus(history).consensus_reached

    transcript = await runner.run(stop_condition=stop)
    # 合意が検出されたターンで停止 (max_turns=10 より少ない)
    assert len(transcript) < 10
    assert any("賛成" in u.text or "納得" in u.text for u in transcript)


def test_consensus_report_no_match_under_threshold() -> None:
    transcript = _utterances(
        ["A", "B", "C", "A"],
        ["主張1", "反論", "別の論点", "なるほど"],
    )
    # 直近 3 ターンのうち 1 件しか合意キーワード無し → 33% < 60%
    report = detect_consensus(transcript, agreement_window=3, agreement_threshold=0.6)
    # 構造シグナルなし & 明示シグナル不足
    assert report.consensus_reached is False


def test_negation_after_agreement_keyword_is_filtered() -> None:
    """「確かに〜が、」「なるほど、しかし」は合意と扱わない (LLM 譲歩前置きパターン)。"""

    transcript = _utterances(
        ["A", "B", "C", "A", "B", "C"],
        [
            "プラ容器を廃止すべき",
            "コスト懸念がある",
            "折衷案も",
            "確かに環境配慮は重要だが、コストが問題だ",
            "なるほど、しかし学生負担が大きい",
            "その通り、ただし現実的でない",
        ],
    )
    report = detect_consensus(transcript)
    assert report.consensus_reached is False
    assert report.signal != "explicit_agreement"


def test_genuine_agreement_without_negation_still_detected() -> None:
    """逆接が無ければ通常通り検出される。"""

    transcript = _utterances(
        ["A", "B", "C", "A", "B", "C"],
        [
            "プラ容器を廃止すべき",
            "コスト懸念がある",
            "折衷案も",
            "なるほど、納得です",
            "賛成、歩み寄りましょう",
            "その通りだと思います",
        ],
    )
    report = detect_consensus(transcript)
    assert report.consensus_reached is True
    assert report.signal == "explicit_agreement"


def test_min_turns_default_blocks_too_early() -> None:
    """既定の min_turns=6 で 5 ターン未満は合意成立しない。"""

    transcript = _utterances(
        ["A", "B", "C", "A", "B"],
        [
            "なるほど",
            "賛成です",
            "その通り",
            "納得",
            "合意",
        ],
    )
    report = detect_consensus(transcript)  # 既定 min_turns_before_consensus=6
    assert report.consensus_reached is False
    assert "ターン数が不足" in report.rationale


def test_consensus_high_threshold_no_false_positive() -> None:
    transcript = _utterances(
        ["A", "B", "C", "A", "B"],
        [
            "主張1",
            "反論",
            "別の論点",
            "なるほど",
            "その通りです",
        ],
    )
    # 高いしきい値だと合意とみなさない
    report = detect_consensus(
        transcript, agreement_window=3, agreement_threshold=0.95
    )
    # 直近 3 ターンの 2/3 = 0.67 < 0.95
    assert report.consensus_reached is False

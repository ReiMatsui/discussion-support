"""DiscussionSimulator._parse_turn の話者分離テスト."""
from __future__ import annotations

from das.asr.live.agents._simulator import DiscussionSimulator


def _sim():
    return DiscussionSimulator(api_key="x", topic="テーマ")


def test_single_speaker_line():
    sp, utt = _sim()._parse_turn("松井: コストを議論しましょう。")
    assert sp == "松井"
    assert utt == "コストを議論しましょう。"


def test_fullwidth_colon():
    sp, utt = _sim()._parse_turn("田中：賛成です。")
    assert sp == "田中"
    assert utt == "賛成です。"


def test_multi_speaker_takes_first_only():
    """複数話者が混ざった応答でも、最初の1人だけを採用する（声の分離維持）."""
    text = "松井: メリットは効率化です。\n田中: でもコストが心配です。\n佐藤: 確かに。"
    sp, utt = _sim()._parse_turn(text)
    assert sp == "松井"
    assert utt == "メリットは効率化です。"
    assert "田中" not in utt  # 2人目以降は取り込まない


def test_unknown_speaker_is_rejected():
    assert _sim()._parse_turn("不明: なにか") == (None, None)


def test_non_speaker_format_rejected():
    assert _sim()._parse_turn("ただのテキスト") == (None, None)


def test_leading_blank_lines_skipped():
    sp, utt = _sim()._parse_turn("\n\n佐藤: そうですね。")
    assert sp == "佐藤"
    assert utt == "そうですね。"

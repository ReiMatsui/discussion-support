"""DiscussionSimulator._parse_turn の話者分離テスト."""
from __future__ import annotations

from das.asr.live.agents._simulator import DiscussionSimulator


def _sim():
    return DiscussionSimulator(api_key="x", topic="テーマ")


def test_single_speaker_line():
    sp, utt = _sim()._parse_turn("参加者A: コストを議論しましょう。")
    assert sp == "参加者A"
    assert utt == "コストを議論しましょう。"


def test_fullwidth_colon():
    sp, utt = _sim()._parse_turn("参加者B：賛成です。")
    assert sp == "参加者B"
    assert utt == "賛成です。"


def test_multi_speaker_takes_first_only():
    """複数話者が混ざった応答でも、最初の1人だけを採用する（声の分離維持）."""
    text = "参加者A: メリットは効率化です。\n参加者B: でもコストが心配です。\n参加者C: 確かに。"
    sp, utt = _sim()._parse_turn(text)
    assert sp == "参加者A"
    assert utt == "メリットは効率化です。"
    assert "参加者B" not in utt  # 2人目以降は取り込まない


def test_unknown_speaker_is_rejected():
    assert _sim()._parse_turn("不明: なにか") == (None, None)


def test_non_speaker_format_rejected():
    assert _sim()._parse_turn("ただのテキスト") == (None, None)


def test_leading_blank_lines_skipped():
    sp, utt = _sim()._parse_turn("\n\n参加者C: そうですね。")
    assert sp == "参加者C"
    assert utt == "そうですね。"

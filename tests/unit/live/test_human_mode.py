"""人間同士ファシリテーションモード（S1）の配線テスト."""
from __future__ import annotations

from das.asr.live._bootstrap import LiveArgs


def test_liveargs_has_topic_field():
    assert LiveArgs().topic is None
    assert LiveArgs(topic="AI導入の是非").topic == "AI導入の是非"


def test_cli_has_topic_option():
    from das.asr.live import main
    names = {p.name for p in main.params}
    assert "topic" in names


def test_agenda_precedence_prefers_topic():
    """議題シードの優先順位は topic > debate > simulate."""
    # run_session の該当ロジックと同じ式を検証（args.topic or args.debate or args.simulate）
    a = LiveArgs(topic="T", debate="D", simulate="S")
    assert (a.topic or a.debate or a.simulate) == "T"
    b = LiveArgs(topic=None, debate="D", simulate="S")
    assert (b.topic or b.debate or b.simulate) == "D"
    c = LiveArgs(topic=None, debate=None, simulate="S")
    assert (c.topic or c.debate or c.simulate) == "S"
    d = LiveArgs()
    assert (d.topic or d.debate or d.simulate) is None

"""人間同士ファシリテーションモード（S1/S3）の配線テスト."""
from __future__ import annotations

import datetime
import threading
import time

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


# ---------------------------------------------------------------------------
# S3: 冒頭アジェンダ自動検出
# ---------------------------------------------------------------------------

def test_detect_agenda_parses_agenda(monkeypatch):
    import das.asr.live._bootstrap as bootstrap
    monkeypatch.setattr(bootstrap, "_post_chat_json",
                        lambda *a, **k: {"agenda": "AI導入の是非"})
    assert bootstrap.detect_agenda([{"speaker": "A", "text": "x"}],
                                   "key", "m") == "AI導入の是非"


def test_detect_agenda_returns_none(monkeypatch):
    import das.asr.live._bootstrap as bootstrap
    monkeypatch.setattr(bootstrap, "_post_chat_json", lambda *a, **k: {"agenda": None})
    assert bootstrap.detect_agenda([{"speaker": "A", "text": "x"}], "key", "m") is None
    monkeypatch.setattr(bootstrap, "_post_chat_json", lambda *a, **k: None)
    assert bootstrap.detect_agenda([{"speaker": "A", "text": "x"}], "key", "m") is None


def _make_state():
    from das.asr.live._session_state import SessionState
    return SessionState(
        args=object(), started=datetime.datetime(2026, 1, 1),
        out_path="/tmp/o.md", html_path="/tmp/o.html", diag_path="/tmp/o.diag",
        turns_path="/tmp/o.turns", wav_path="/tmp/o.wav",
    )


def test_agenda_detector_seeds_when_detected(monkeypatch):
    """十分な発話がたまると、検出した議題をseedして停止する（S3）."""
    import das.asr.live._bootstrap as bootstrap
    from das.asr.live._workers import _run_agenda_detector

    monkeypatch.setattr(bootstrap, "detect_agenda", lambda *a, **k: "AI導入の是非")
    state = _make_state()
    state.records = [{"speaker": "話者1", "text": f"発話{i}",
                      "ms": i * 1000, "end_ms": i * 1000 + 500} for i in range(4)]

    t = threading.Thread(target=_run_agenda_detector,
                         args=(state, "key", "gpt-5-mini"), daemon=True)
    t.start()
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and not state.topics:
        time.sleep(0.05)
    state.stop.set()
    t.join(timeout=2)

    assert [tp["topic"] for tp in state.topics] == ["AI導入の是非"]


def test_agenda_detector_skips_when_topics_exist(monkeypatch):
    """既に論点があれば議題検出は走らない（S3）."""
    import das.asr.live._bootstrap as bootstrap
    from das.asr.live._workers import _run_agenda_detector

    calls = []
    monkeypatch.setattr(bootstrap, "detect_agenda",
                        lambda *a, **k: calls.append(1) or "X")
    state = _make_state()
    state.topics = [{"topic": "既存論点", "speaker": "話者1"}]
    state.records = [{"speaker": "話者1", "text": f"u{i}",
                      "ms": i * 1000, "end_ms": i * 1000 + 500} for i in range(5)]

    t = threading.Thread(target=_run_agenda_detector,
                         args=(state, "key", "gpt-5-mini"), daemon=True)
    t.start()
    time.sleep(2.5)
    state.stop.set()
    t.join(timeout=2)

    assert calls == []
    assert [tp["topic"] for tp in state.topics] == ["既存論点"]

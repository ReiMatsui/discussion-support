"""ターミナル出力のログ保存（_console_log）."""
from __future__ import annotations

import io
import sys
import threading

from das.asr.live._console_log import ConsoleLog, _Tee
from das.asr.live._constants import CLEAR_LINE


def _tee():
    screen, log = io.StringIO(), io.StringIO()
    return _Tee(screen, log, threading.Lock()), screen, log


def test_completed_lines_are_copied_without_ansi():
    t, screen, log = _tee()
    t.write("# AI Agent: \x1b[96m接続完了\x1b[0m\n")
    t.write("HTTP Request: POST ... 200 OK\n")
    assert screen.getvalue().startswith("# AI Agent: \x1b[96m")
    assert log.getvalue() == "# AI Agent: 接続完了\nHTTP Request: POST ... 200 OK\n"


def test_partial_transcript_overwrites_are_dropped():
    """画面で消される途中経過（\\r\\x1b[K で上書き）はログに残さず、確定行だけ残す."""
    t, _screen, log = _tee()
    t.write(CLEAR_LINE + "?: AIツールの導入が")
    t.write(CLEAR_LINE + "?: AIツールの導入が業務効率を")
    t.write(CLEAR_LINE + "[00:01] 参加者A: AIツールの導入が業務効率を上げます。\n")
    t.write(CLEAR_LINE + "?: 次の途中")
    t.write("# [diag] agent: pending=1\n")
    assert log.getvalue() == ("[00:01] 参加者A: AIツールの導入が業務効率を上げます。\n"
                              "# [diag] agent: pending=1\n")


def test_console_log_restores_streams_and_flushes_tail(tmp_path):
    path = tmp_path / "x.log"
    out, err = sys.stdout, sys.stderr
    with ConsoleLog(str(path)):
        print("一行目")
        print("書きかけ", end="", file=sys.stderr)
        assert sys.stdout is not out
    assert sys.stdout is out and sys.stderr is err
    assert path.read_text(encoding="utf-8") == "一行目\n書きかけ\n"

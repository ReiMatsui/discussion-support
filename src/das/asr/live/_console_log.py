"""ターミナル出力を議事録と同じ場所にログとして残す.

`python -m das.asr.live` の実行中に画面へ出るもの（`#` で始まる進行ログ、
HTTP のログ、例外）をそのまま `transcripts/<日時>.log` にも書く。あとから
「あのとき何が起きたか」をターミナルの貼り付けなしで追えるようにするため。

ターミナル向けの制御（`\\r\\x1b[K` で行を消して書き直す途中経過の文字起こし、
色の指定）はファイルには残さない。行を消す指示が来たら、その行の書きかけを
捨てる——画面に最終的に残るものだけがログに残る。
"""
from __future__ import annotations

import contextlib
import re
import sys
import threading
from typing import IO

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_LINE_RESET = re.compile(r"\r\x1b\[K|\r")


class _Tee:
    """元のストリームへ流しつつ、完成した行だけをログファイルへ書く."""

    def __init__(self, stream: IO[str], log: IO[str], lock: threading.Lock):
        self._stream = stream
        self._log = log
        self._lock = lock
        self._buf = ""
        self._partial = False   # 書きかけが「行を消して書き直す途中経過」か

    def write(self, s: str) -> int:
        n = self._stream.write(s)
        with self._lock:
            self._ingest(s)
        return n

    def _ingest(self, s: str) -> None:
        if self._partial and not _LINE_RESET.match(s):
            # 途中経過の上に別の出力が続いた。画面では同じ行にくっつくが、
            # 途中経過はどうせ書き直されるものなのでログには残さない
            self._buf = ""
            self._partial = False
        for piece in s.split("\n")[:-1]:
            self._buf = self._reset(self._buf + piece)
            self._emit(self._buf)
            self._buf = ""
            self._partial = False
        tail = s.split("\n")[-1]
        if _LINE_RESET.search(tail):
            self._partial = True
        self._buf = self._reset(self._buf + tail)

    @staticmethod
    def _reset(line: str) -> str:
        # 行を消す指示より前の書きかけは画面に残らない。ログにも残さない
        parts = _LINE_RESET.split(line)
        return parts[-1]

    def _emit(self, line: str) -> None:
        text = _ANSI.sub("", line).rstrip()
        if not text:
            return
        with contextlib.suppress(Exception):
            self._log.write(text + "\n")
            self._log.flush()

    def flush(self) -> None:
        self._stream.flush()
        with contextlib.suppress(Exception):
            self._log.flush()

    def close_log(self) -> None:
        with self._lock:
            if self._buf.strip():
                self._emit(self._buf)
                self._buf = ""

    def __getattr__(self, name):  # isatty / fileno / encoding などは元のまま
        return getattr(self._stream, name)


class ConsoleLog:
    """stdout / stderr をログファイルへ複写する。`with` で囲む."""

    def __init__(self, path: str):
        self.path = path
        self._fh: IO[str] | None = None
        self._orig: tuple[IO[str], IO[str]] | None = None
        self._tees: list[_Tee] = []

    def __enter__(self) -> ConsoleLog:
        self._fh = open(self.path, "a", encoding="utf-8")
        lock = threading.Lock()
        self._orig = (sys.stdout, sys.stderr)
        self._tees = [_Tee(sys.stdout, self._fh, lock), _Tee(sys.stderr, self._fh, lock)]
        sys.stdout = self._tees[0]  # type: ignore[assignment]
        sys.stderr = self._tees[1]  # type: ignore[assignment]
        return self

    def __exit__(self, *exc) -> None:
        for t in self._tees:
            t.close_log()
        if self._orig is not None:
            sys.stdout, sys.stderr = self._orig
        if self._fh is not None:
            with contextlib.suppress(Exception):
                self._fh.close()

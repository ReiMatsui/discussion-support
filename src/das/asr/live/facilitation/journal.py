"""介入イベントをJSONLへ保存する追記専用ジャーナル."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .events import FacilitationEvent


class FacilitationJournal:
    """複数スレッドから安全に介入イベントを追記する."""

    def __init__(self, path: str | os.PathLike[str]):
        self._path = Path(path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def set_path(self, path: str | os.PathLike[str]) -> None:
        """新しい会議の出力先へ切り替える."""
        with self._lock:
            self._path = Path(path)

    def append(
        self,
        event: FacilitationEvent,
        *,
        path: str | os.PathLike[str] | None = None,
    ) -> None:
        """イベントをUTF-8 JSONLとして1行追記する."""
        line = json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            target = Path(path) if path is not None else self._path
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as file:
                file.write(line)
                file.write("\n")

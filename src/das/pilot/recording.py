"""MediaRecorderの順序付きチャンクを保存する。単独チャンクではなく全体を連結する。"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

MAX_CHUNK_BYTES = 16 * 1024 * 1024
MAX_VIDEO_BYTES = 512 * 1024 * 1024


class VideoRecording:
    def __init__(self, directory: Path, mime: str):
        if mime not in {"video/webm", "video/mp4"}:
            raise ValueError("対応していない動画形式です")
        self.path = directory / ("camera.webm" if mime == "video/webm" else "camera.mp4")
        self.manifest_path = directory / "video.json"
        self.mime = mime
        self.chunks = 0
        self.bytes = 0
        self.complete = False
        self.lock = threading.Lock()
        self.file = self.path.open("xb")
        self._save()

    def _save(self):
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "file": self.path.name, "mime": self.mime, "chunks": self.chunks,
            "bytes": self.bytes, "complete": self.complete,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)

    def append(self, sequence: int, data: bytes):
        with self.lock:
            if self.file.closed or self.complete:
                raise ValueError("この録画は終了しています")
            if sequence != self.chunks:
                raise ValueError("動画データの順序が一致しません。録画を終了して確認してください")
            if not data or len(data) > MAX_CHUNK_BYTES or self.bytes + len(data) > MAX_VIDEO_BYTES:
                raise ValueError("動画データのサイズ上限です")
            self.file.write(data)
            self.file.flush()
            os.fsync(self.file.fileno())
            self.chunks += 1
            self.bytes += len(data)
            self._save()

    def finish(self, expected_chunks: int):
        with self.lock:
            if expected_chunks != self.chunks or self.chunks == 0:
                raise ValueError("動画の保存件数が一致しないか、まだデータがありません")
            self.complete = True
            self.file.close()
            self._save()

    def close_incomplete(self):
        with self.lock:
            if not self.file.closed:
                self.file.close()
            self._save()

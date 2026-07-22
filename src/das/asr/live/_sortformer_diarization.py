"""ローカル Streaming Sortformer diarization provider（サブプロセス方式）.

NVIDIA Streaming Sortformer (HF: nvidia/diar_streaming_sortformer_4spk-v2.1)
を、NeMo 専用 venv のサブプロセス（scripts/sortformer_worker.py）として起動し、
既存の :class:`DiarizationProvider` プロトコルに載せる。本体の依存に NeMo を
入れないための分離で、通信は
  stdin  : 16kHz mono PCM16 生バイト
  stdout : JSON Lines（{"e":"ready"} / {"e":"start","ms":..,"spk":..} /
           {"e":"end","ms":..,"spk":..}）
のみ（プロトコルはワーカー側 docstring と対）。

位置づけ（docs/design/sortformer_feasibility_2026-07-22.md）:
クリーン音源では現行構成を大差で上回り、マイク残響環境では劣る実測。
opt-in（--diarization sortformer）の検証用プロバイダであり、既定の
pyannote / 単独モードの挙動には一切影響しない。

タイムスタンプは「このプロバイダに送った音声の先頭からの ms」で、
pyannote provider と同じ基準（セッション音声クロック）。ワーカーの
プリセットは low レイテンシ（約1秒）を既定にする。

障害時の設計: ワーカーが起動失敗・途中死した場合、このプロバイダは
以後イベントを返さない不活性状態に落ちる（例外で本体を巻き込まない）。
ライブは Soniox ラベル＋声紋のみで継続する。
"""
from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from ._diarization import DiarizationEvent

_WORKER_PATH = (Path(__file__).resolve().parents[4] / "scripts"
                / "sortformer_worker.py")

# ワーカー venv の python の探索順: 引数 > 環境変数 > 既定パス。
_PYTHON_ENV_VAR = "SORTFORMER_PYTHON"
_DEFAULT_VENV_PYTHON = "~/.venvs/sortformer/bin/python"


def resolve_worker_python(explicit: str | None = None) -> str:
    """ワーカーを動かす python 実行体を決める（存在確認はしない）."""
    cand = explicit or os.environ.get(_PYTHON_ENV_VAR) or _DEFAULT_VENV_PYTHON
    return os.path.expanduser(cand)


class SortformerLocalDiarizationProvider:
    """Streaming Sortformer をサブプロセスで駆動する DiarizationProvider."""

    def __init__(self, *, python_path: str | None = None,
                 model: str = "nvidia/diar_streaming_sortformer_4spk-v2.1",
                 latency: str = "low", device: str = "cpu",
                 max_speakers: int | None = None,
                 worker_path: str | os.PathLike | None = None) -> None:
        self._python = resolve_worker_python(python_path)
        self._model = model
        self._latency = latency
        self._device = device
        # モデル仕様上 4 話者固定。ヒントは受け取るだけ（他 provider と同形）。
        self._max_speakers = max_speakers
        self._worker_path = str(worker_path or _WORKER_PATH)
        self._proc: subprocess.Popen | None = None
        self._events: queue.Queue[DiarizationEvent] = queue.Queue()
        self._active_starts: dict[str, int] = {}
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._dead = False
        self._ready = threading.Event()

    @property
    def name(self) -> str:
        return "sortformer"

    # ------------------------------------------------------------------
    def start(self) -> None:
        """ワーカーを起動する。close() 後の再 start()（会議リセット/STT再接続,
        _bootstrap の close→start 対）では全状態を作り直す。モデル再読込の
        ため再開まで数十秒かかる点は既知のコスト（検証用 opt-in の割り切り）。
        """
        if self._proc is not None and self._proc.poll() is None:
            with contextlib.suppress(Exception):
                self._proc.kill()
        self._proc = None
        self._events = queue.Queue()
        with self._lock:
            self._active_starts.clear()
        self._dead = False
        self._ready = threading.Event()
        cmd = [self._python, self._worker_path,
               "--model", self._model, "--latency", self._latency,
               "--device", self._device]
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=sys.stderr.fileno() if hasattr(sys.stderr, "fileno")
                else subprocess.DEVNULL,
            )
        except OSError as exc:
            self._dead = True
            print(f"# 話者分離(sortformer): ワーカー起動失敗のため無効化: {exc}\n"
                  f"#   python={self._python}\n"
                  f"#   セットアップ: docs/design/sortformer_live_setup_2026-07-22.md",
                  flush=True)
            return
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        for raw in proc.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("{"):
                continue   # 防御: 万一のログ混入は読み飛ばす
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = d.get("e")
            if kind == "ready":
                self._ready.set()
                print("# 話者分離(sortformer): ワーカー準備完了", flush=True)
                continue
            spk = str(d.get("spk") or "")
            ms = int(d.get("ms") or 0)
            if kind == "start" and spk:
                with self._lock:
                    self._active_starts[spk] = ms
            elif kind == "end" and spk:
                with self._lock:
                    start_ms = self._active_starts.pop(spk, None)
                if start_ms is not None and ms > start_ms:
                    self._events.put(DiarizationEvent(
                        start_ms=start_ms, end_ms=ms,
                        speaker=spk, source=self.name))
        # stdout EOF = ワーカー終了
        code = proc.poll()
        if not self._dead and code not in (0, None):
            print(f"# 話者分離(sortformer): ワーカーが終了しました (code={code})。"
                  f"以後はSTT+声紋のみで継続します", flush=True)
        self._dead = True

    # ------------------------------------------------------------------
    def send_audio(self, pcm16k: bytes) -> None:
        proc = self._proc
        if self._dead or proc is None or proc.stdin is None:
            return
        try:
            proc.stdin.write(pcm16k)
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            if not self._dead:
                self._dead = True
                print("# 話者分離(sortformer): ワーカーへの送信失敗。"
                      "以後はSTT+声紋のみで継続します", flush=True)

    def drain_events(self) -> list[DiarizationEvent]:
        events: list[DiarizationEvent] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def active_events(self) -> list[DiarizationEvent]:
        with self._lock:
            return [DiarizationEvent(start_ms, None, speaker, self.name)
                    for speaker, start_ms in self._active_starts.items()]

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        # stdin EOF でワーカーは残りを出し切って自然終了する。
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        self._dead = True

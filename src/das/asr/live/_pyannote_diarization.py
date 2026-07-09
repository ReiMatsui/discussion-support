"""pyannoteAI streaming diarization (Live-1) provider.

2026-07-09 時点の docs.pyannote.ai/tutorials/streaming-real-time および
docs.pyannote.ai/api-reference/{create-stream,streaming} (AsyncAPI) で
正式仕様を確認済み。要点:

  - セッション作成: ``POST https://api.pyannote.ai/v1/live`` body ``{}``
    (Authorization: Bearer <key>) -> ``{"id": "...", "url": "<ws url>"}``。
    ``url`` はワンタイムトークン入りで、そのままWS接続に使える
    （追加ヘッダ不要）。旧実装のこの部分は仕様と一致していたため変更なし。
  - 音声フォーマット: PCM float32 little-endian (pcm_f32le)、16kHz、mono、
    **1チャンク=100ms（1600サンプル/6400バイト）固定**。WAVヘッダ等は付けず
    生バイトのみをバイナリWSフレームで送る。サーバは最大5秒のバッファを
    許容するのみで、実時間より先行して送ると切断される。
    -> 呼び出し元（_workers.py）はマイク経由では100ms(1600サンプル)刻みで
    send_audio() を呼ぶが、WAV再生シミュレーション経路は120ms刻みで呼ぶため
    そのまま転送すると仕様の100ms固定チャンクに違反しうる。本改修で内部に
    100ms境界のリングバッファを持ち、常に6400バイト単位で送信するように変更。
  - 終了: JSON テキストフレーム ``{"type": "end_of_stream"}`` を送ると
    サーバが確定イベントを出し切ってから close code 1000 で切断する
    （生ソケットを黙って閉じるのは非推奨）。旧実装のこの部分も仕様通り。
  - 受信イベント: ``diarization_speaker_start`` / ``diarization_speaker_end``
    ({"type": ..., "data": {"timestamp": <秒>, "speaker": "SPEAKER_00"}}) は
    旧実装のパースがそのまま仕様と一致。加えて ``error``
    ({"type": "error", "message": "..."}) が定義されているが旧実装は無視して
    いた（黙って握りつぶすこと自体は許容範囲だが、原因追跡できないため今回
    ログに出すよう変更）。
  - 話者数: 最大8人まで同時追跡（`data.speaker` は "SPEAKER_00".."SPEAKER_07"
    相当のセッション内固定ラベル）。
"""
from __future__ import annotations

import contextlib
import json
import logging
import queue
import threading
import urllib.request
from typing import Any

import numpy as np

from ._constants import SR
from ._diarization import DiarizationEvent

logger = logging.getLogger(__name__)

# Live-1 の必須チャンク粒度: 16kHz mono PCM16 で 100ms = 1600サンプル = 3200バイト。
# f32le に変換すると 6400バイトになる。
_CHUNK_MS = 100
_CHUNK_SAMPLES = SR * _CHUNK_MS // 1000
_CHUNK_BYTES_PCM16 = _CHUNK_SAMPLES * 2


class PyannoteStreamingDiarizationProvider:
    """pyannoteAI のリアルタイム話者分離 (Live-1) WebSocket provider.

    入力側の共通形式は既存のライブ処理に合わせて 16kHz PCM16 bytes とし、
    pyannoteAI Live-1 が要求する 16kHz mono float32 little-endian・100ms固定
    チャンクに内部変換して送る。
    """

    _CREATE_URL = "https://api.pyannote.ai/v1/live"

    def __init__(self, api_key: str, *, create_url: str | None = None) -> None:
        self.api_key = api_key
        self.create_url = create_url or self._CREATE_URL
        self.stream_id: str | None = None
        self._ws: Any = None
        self._events: queue.Queue[DiarizationEvent] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()
        self._active_starts: dict[str, int] = {}
        self._pcm_buf = bytearray()

    @property
    def name(self) -> str:
        return "pyannote"

    def start(self) -> None:
        from websockets.sync.client import connect

        self._stop.clear()
        self._active_starts.clear()
        self._pcm_buf.clear()
        req = urllib.request.Request(self.create_url, data=b"{}", method="POST")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read())
        url = payload["url"]
        self.stream_id = payload.get("id")
        self._ws = connect(url)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def send_audio(self, pcm16k: bytes) -> None:
        """16kHz mono PCM16 bytes を受け取り、Live-1 仕様の100ms固定 f32le
        チャンク(6400バイト)に再分割して送信する。

        呼び出し元のチャンク境界（マイク100ms / WAVシミュレーション120ms等）
        は仕様の100ms固定と一致しないことがあるため、内部バッファで吸収する。
        """
        if self._ws is None:
            return
        self._pcm_buf.extend(pcm16k)
        while len(self._pcm_buf) >= _CHUNK_BYTES_PCM16:
            chunk = bytes(self._pcm_buf[:_CHUNK_BYTES_PCM16])
            del self._pcm_buf[:_CHUNK_BYTES_PCM16]
            payload = pcm16_to_pyannote_f32(chunk)
            if payload:
                self._ws.send(payload)

    def drain_events(self) -> list[DiarizationEvent]:
        events: list[DiarizationEvent] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def active_events(self) -> list[DiarizationEvent]:
        return [
            DiarizationEvent(start_ms, None, speaker, self.name)
            for speaker, start_ms in self._active_starts.items()
        ]

    def close(self) -> None:
        """end_of_stream を送り、サーバが確定イベントを出し切って自発的に
        close(code 1000)するのを少し待ってからソケットを閉じる。

        仕様(docs.pyannote.ai/tutorials/streaming-real-time)は
        「end_of_stream送信後、サーバは残りのイベントを出し切ってから閉じる。
        生ソケットを即座に閉じると最終出力を失いうる」と明記しているため、
        _stop を即セットしてreaderを止めるのではなく、reader(recvループ)が
        サーバ側クローズで自然終了するのを timeout 付きで待ってから閉じる。
        """
        if self._ws is not None:
            # 100ms境界に満たない端数(< 3200バイトPCM16)が残っていれば、
            # 失うよりはそのまま送る（サーバはend_of_stream前の最終フレーム
            # サイズを厳密検証しない。仕様上は100ms固定が基本だが、
            # ストリーム終端の端数フレームまでは拒否されない想定）。
            if self._pcm_buf:
                with contextlib.suppress(Exception):
                    payload = pcm16_to_pyannote_f32(bytes(self._pcm_buf))
                    if payload:
                        self._ws.send(payload)
                self._pcm_buf.clear()
            with contextlib.suppress(Exception):
                self._ws.send(json.dumps({"type": "end_of_stream"}))
            if self._reader is not None:
                self._reader.join(timeout=5.0)
            self._stop.set()
            with contextlib.suppress(Exception):
                self._ws.close()
        else:
            self._stop.set()
        if self._reader is not None:
            self._reader.join(timeout=1.0)

    def _read_loop(self) -> None:
        while not self._stop.is_set() and self._ws is not None:
            try:
                raw = self._ws.recv()
            except Exception:
                break
            event = self._parse_message(raw)
            if event is not None:
                self._events.put(event)

    def _parse_message(self, raw: str | bytes) -> DiarizationEvent | None:
        msg = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        typ = msg.get("type")
        if typ == "error":
            logger.warning("pyannote Live-1 error event: %s", msg.get("message"))
            return None
        if typ not in {"diarization_speaker_start", "diarization_speaker_end"}:
            return None
        data = msg.get("data") or {}
        speaker = data.get("speaker")
        timestamp = data.get("timestamp")
        if not isinstance(speaker, str) or not isinstance(timestamp, int | float):
            return None
        ms = int(float(timestamp) * 1000)
        if typ == "diarization_speaker_start":
            self._active_starts[speaker] = ms
            return None
        start_ms = self._active_starts.pop(speaker, ms)
        return DiarizationEvent(
            start_ms=start_ms,
            end_ms=ms,
            speaker=speaker,
            source=self.name,
        )


def pcm16_to_pyannote_f32(pcm16k: bytes) -> bytes:
    """テストしやすいPCM16→pyannote入力形式変換."""
    samples = np.frombuffer(pcm16k, dtype="<i2").astype(np.float32) / 32768.0
    if SR != 16000:
        raise ValueError("pyannote streaming provider expects 16kHz audio")
    return samples.astype("<f4").tobytes()

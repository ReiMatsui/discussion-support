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
  - 話者数ヒント: ``POST /v1/live`` の body スキーマは
    ``application/json`` の ``object`` 型で、プロパティは一切定義されて
    いない（2026-07-09時点、docs.pyannote.ai/api-reference/create-stream
    のOpenAPIスキーマで確認。``maxSpeakers``/``numSpeakers`` 等は存在しない）。
    つまり Live-1 は話者数ヒントを受け付けない仕様であり、本provider内で
    最大話者数を指定しても pyannoteAI 側には送れない。以下の
    ``max_speakers`` 引数はこの事実を踏まえた上での「配線だけ用意」で
    あり、API へは送信されない（将来 API が対応した場合に備えたプレース
    ホルダ、および将来ローカルでのクラスタ数制限に使うためのフック）。
    実際にセッション内の人間話者数を抑制したい場合は、既存の
    ``--diarization-max-speakers`` → ``SessionState.constrain_human_speaker_key``
    /``VoiceProfiles.set_max_human_speakers`` の経路（_bootstrap.py）が
    セッションレベルで機能する。
"""
from __future__ import annotations

import contextlib
import json
import logging
import queue
import threading
import urllib.error
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

    自動再接続:
      サーバ切断（close code 1011 等）は scripts/test_pyannote_live.py で
      先行実装・検証済みだったロジック（新セッション作成＋タイムスタンプ
      オフセット補正）を本体に移植したもの。``send_audio()`` 中に送信が
      失敗した場合、``max_reconnects`` 回まで自動的に新しい Live-1 セッション
      を作り直し、そのセッション内タイムスタンプに「これまでに送信できた
      音声の累計ms」を加算して連続したタイムラインに補正する
      （``_session_base_ms``）。
      pyannoteAI のセッション内話者ラベル(SPEAKER_00等)は新セッションでは
      ラベル空間が変わる（同じ人が別ラベルになりうる）ため、再接続後の
      ラベルには ``R{epoch}:`` を前置して衝突を避ける
      （``_label_epoch``、epoch=0は前置なし）。この「再接続直後に新しい
      ラベルが出現する」挙動は、SessionState側の参加者化ヒステリシス
      (``PYANNOTE_PARTICIPANT_HYSTERESIS_S``, 既定3.0秒。
      ``_session_state.py`` の ``key_for_diarization_speaker`` 参照)が
      吸収する設計。再接続直後の短い揺れでは偽参加者を作らず、既存参加者の
      発話が3秒以上そのラベルに乗り続けた場合のみ新規参加者として確定する。
    """

    _CREATE_URL = "https://api.pyannote.ai/v1/live"

    def __init__(
        self,
        api_key: str,
        *,
        create_url: str | None = None,
        max_speakers: int | None = None,
        max_reconnects: int = 3,
        auto_reconnect: bool = True,
    ) -> None:
        self.api_key = api_key
        self.create_url = create_url or self._CREATE_URL
        # Live-1 の `POST /v1/live` はボディにプロパティを持たない(objectのみ)
        # ため話者数ヒントを送る手段が無い。ここでの保持は将来API対応時の
        # フックであり、現状はAPI呼び出しに一切反映されない
        # （クラスdocstring冒頭「話者数ヒント」節を参照）。
        self.max_speakers = max_speakers
        self.max_reconnects = max_reconnects
        self.auto_reconnect = auto_reconnect
        self.stream_id: str | None = None
        self._ws: Any = None
        self._events: queue.Queue[DiarizationEvent] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()
        self._active_starts: dict[str, int] = {}
        self._pcm_buf = bytearray()
        self._reconnects = 0
        self._session_base_ms = 0
        self._label_epoch = 0
        self._sent_audio_ms = 0
        self._started_once = False

    @property
    def name(self) -> str:
        return "pyannote"

    def start(self) -> None:
        self._stop.clear()
        self._active_starts.clear()
        self._pcm_buf.clear()
        self._reconnects = 0
        # ラベルepoch・タイムライン基点は start() でリセットしない（2026-07-15
        # レビュー F3）。_bootstrap の STT切断復旧は同一インスタンスに対して
        # provider.close(); provider.start() を行うため、epoch を 0 に戻すと
        # 新セッションの SPEAKER_00 が旧セッションのラベル空間
        # （ClusterVoiceNamer._confirmed / SessionState.diarization_speaker_keys の
        # "pyannote:SPEAKER_00" 等）と衝突し、再起動後の別人が旧確定名へ即誤帰属
        # し得る。再接続時（_handle_disconnect）と同様に epoch をインクリメント
        # すれば、既存の R{epoch}: 前置（クラスdocstring「自動再接続」節）で旧キー
        # と自然に区別され、_session_base_ms の引き継ぎでタイムラインも会議内で
        # 単調のまま保たれる。初回 start のみ epoch=0（前置なし）で従来どおり。
        if self._started_once:
            self._label_epoch += 1
            self._session_base_ms += self._sent_audio_ms
        self._started_once = True
        self._sent_audio_ms = 0
        self._connect()

    def _connect(self) -> None:
        """新しい Live-1 セッションを作成しWS接続する（初回start/再接続共通）."""
        from websockets.sync.client import connect

        req = urllib.request.Request(self.create_url, data=b"{}", method="POST")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            # 本文に理由が書かれている。捨てると「400 Bad Request」だけが出て
            # 原因が分からない（LLM側で実際にそうなった。handoff §43）。
            # 例外はそのまま上げる——起動を止める判断は呼び出し側が持つ。
            detail = ""
            with contextlib.suppress(Exception):
                detail = e.read().decode("utf-8", "replace")[:600]
            logger.error("pyannote Live-1: セッション作成が %s で失敗: %s",
                         e.code, detail or e.reason)
            raise
        url = payload["url"]
        self.stream_id = payload.get("id")
        self._ws = connect(url)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _handle_disconnect(self, exc: Exception) -> None:
        """送信失敗を検知した際に自動再接続を試みる（再接続数上限あり）."""
        logger.warning("pyannote Live-1: 送信中に切断を検知しました (%s)。", exc)
        if not self.auto_reconnect or self._reconnects >= self.max_reconnects:
            logger.error(
                "pyannote Live-1: 再接続を行いません（auto_reconnect=%s, %d/%d回）。",
                self.auto_reconnect, self._reconnects, self.max_reconnects,
            )
            self._ws = None
            return
        with contextlib.suppress(Exception):
            if self._ws is not None:
                self._ws.close()
        if self._reader is not None:
            self._reader.join(timeout=1.0)
        self._reconnects += 1
        # これまでに送信できた音声の累計msをオフセットとして次セッションに引き継ぐ。
        self._session_base_ms += self._sent_audio_ms
        self._sent_audio_ms = 0
        self._label_epoch += 1
        self._active_starts.clear()
        self._stop.clear()
        try:
            self._connect()
            logger.warning(
                "pyannote Live-1: 新セッションで再接続しました (%d/%d回目、"
                "音声内位置 %dms から再開、ラベルepoch=%d)。",
                self._reconnects, self.max_reconnects,
                self._session_base_ms, self._label_epoch,
            )
        except Exception:
            logger.exception("pyannote Live-1: 再接続に失敗しました。")
            self._ws = None

    def send_audio(self, pcm16k: bytes) -> None:
        """16kHz mono PCM16 bytes を受け取り、Live-1 仕様の100ms固定 f32le
        チャンク(6400バイト)に再分割して送信する。

        呼び出し元のチャンク境界（マイク100ms / WAVシミュレーション120ms等）
        は仕様の100ms固定と一致しないことがあるため、内部バッファで吸収する。
        送信中にサーバ切断を検知した場合、``auto_reconnect`` が有効なら
        新セッションを作って同じチャンクを送り直す（自動再接続）。
        """
        if self._ws is None:
            return
        self._pcm_buf.extend(pcm16k)
        while len(self._pcm_buf) >= _CHUNK_BYTES_PCM16:
            chunk = bytes(self._pcm_buf[:_CHUNK_BYTES_PCM16])
            del self._pcm_buf[:_CHUNK_BYTES_PCM16]
            payload = pcm16_to_pyannote_f32(chunk)
            if not payload:
                continue
            try:
                self._ws.send(payload)
            except Exception as exc:
                self._handle_disconnect(exc)
                if self._ws is None:
                    raise
                self._ws.send(payload)
            self._sent_audio_ms += _CHUNK_MS

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
        raw_speaker = data.get("speaker")
        timestamp = data.get("timestamp")
        if not isinstance(raw_speaker, str) or not isinstance(timestamp, int | float):
            return None
        # 再接続後(epoch>0)はラベル空間が変わるため前置して衝突を避ける。
        # クラスdocstring「自動再接続」節参照。
        speaker = raw_speaker if self._label_epoch == 0 else f"R{self._label_epoch}:{raw_speaker}"
        ms = int(float(timestamp) * 1000) + self._session_base_ms
        if typ == "diarization_speaker_start":
            self._active_starts[speaker] = ms
            return None
        # diarization_speaker_end。仕様(streaming-real-time)上は
        # start/end が必ず対で来る想定だが、実測ではサーバ側の重複end送信・
        # 再接続直後の取りこぼし等で「対応するstartが無いend」が届くことが
        # あった。以前の実装は `pop(speaker, ms)` で end の timestamp 自体を
        # フォールバックのstartに使っており、start_ms == end_ms の縮退
        # セグメント（0ms区間）を量産していた（DiarizationEvent.closed()は
        # end<=startを弾くため下流には影響しないが、drain_events()経由で
        # 生ログ・統計に混入し、イベントペアリングの不整合を隠していた）。
        # 対応するstartが無いendは実区間を再構成できないため、ここで
        # ログを残した上で捨てる（Noneを返す）。
        if speaker not in self._active_starts:
            logger.warning(
                "pyannote Live-1: speaker=%s の speaker_end (ts=%.3fs) に対応する"
                " speaker_start がありません。縮退セグメント化を避けるため破棄します。",
                speaker, timestamp if isinstance(timestamp, int | float) else -1.0,
            )
            return None
        start_ms = self._active_starts.pop(speaker)
        if ms <= start_ms:
            logger.warning(
                "pyannote Live-1: speaker=%s の区間が非正 (start_ms=%d end_ms=%d)。"
                " 縮退セグメントとして破棄します。",
                speaker, start_ms, ms,
            )
            return None
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

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
import time
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
_SR_BYTES_PER_MS = _CHUNK_BYTES_PCM16 // _CHUNK_MS   # 16kHz PCM16 = 32 バイト/ms


class PyannoteStreamingDiarizationProvider:
    """pyannoteAI のリアルタイム話者分離 (Live-1) WebSocket provider.

    入力側の共通形式は既存のライブ処理に合わせて 16kHz PCM16 bytes とし、
    pyannoteAI Live-1 が要求する 16kHz mono float32 little-endian・100ms固定
    チャンクに内部変換して送る。

    自動再接続:
      サーバ切断（close code 1011 等）への対応は、先行スクリプト（旧
      scripts/test_pyannote_live.py、本体移植後に削除）で検証済みだった
      ロジック（新セッション作成＋タイムスタンプオフセット補正）の移植。``send_audio()`` 中に送信が
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

      再接続は別スレッドで行い（``_reconnect_worker``）、送信スレッドは止めない。
      再接続中に届いた音声は分離には送らず時刻の基点だけ進める。溜め込んで
      再接続後に一気に送るとサーバの「実時間より 5 秒以上先行」で切られる。
      2026-09-28 の実測（YouTube 再生 4 本中 3 本、過去の AMI 一括実行の多く）
      では、10 分前後でサーバ都合の 1011 が来た後、再接続→溜まり分の全捨て→
      5 秒無音で 1008→再接続、の連鎖で 3 回の上限を使い切って分離が死んでいた。
      原因は先行判定に捨てた音声まで数えていたこと（``send_audio`` 内の注記）。
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
        # 再接続カウンタを忘れるまでの安定送信時間。_sent_audio_ms は再接続で
        # 0 に戻るので「直近の再接続からの安定時間」をそのまま表す。
        self._RECONNECT_FORGET_MS = 60_000
        self._MAX_AHEAD_MS = 3_500   # サーバの上限 5,000ms に対する余裕
        self.stream_id: str | None = None
        self._ws: Any = None
        self._events: queue.Queue[DiarizationEvent] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()
        self._active_starts: dict[str, int] = {}
        # _active_starts は3スレッドが触る（readerが挿入/pop、送信スレッドが
        # clear、recv側が active_events で走査）。無ロックだと走査中の挿入で
        # RuntimeError（dictionary changed size during iteration）になり、
        # 発話処理ごと落ちる（レビュー 2026-07-30）。
        self._active_lock = threading.Lock()
        self._pcm_buf = bytearray()
        self._reconnects = 0
        self._session_base_ms = 0
        self._label_epoch = 0
        self._sent_audio_ms = 0
        self._started_once = False
        self._connected_at = 0.0     # 現セッションの接続時刻（実時間より先行しない送り方の基準）
        self._dropped_ms = 0         # 先行しすぎて送らなかった音声（タイムラインは進める）
        # 再接続は送信スレッドを止めずに別スレッドで行う。送信スレッドは STT にも
        # 音声を流しているので、ここで数秒止まると文字起こしまで遅れる。再接続中
        # に届いた音声は分離には送らず（送れないので）時刻の基点だけ進める。
        self._reconnecting = False
        self._reconnect_thread: threading.Thread | None = None
        self._timeline_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "pyannote"

    def reset_timeline(self) -> None:
        """「新しい会議」でSTTの時刻が0に戻るのに合わせ、分離側の時刻も0に戻す.

        start() は STT 切断復旧の名残で前セッション分を引き継ぐが、会議の
        リセットでは STT 側（asr_pcm_total_bytes）が 0 から数え直すので、
        引き継ぐと2会議目以降の区間が丸ごとずれて重なりが取れない。
        close() の後、start() の前に呼ぶ。ラベルの epoch はそのまま進める。
        """
        self._session_base_ms = 0
        self._sent_audio_ms = 0
        self._dropped_ms = 0
        self._pcm_buf.clear()

    def start(self) -> None:
        self._stop.clear()
        with self._active_lock:
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
        try:
            self._connect()
        except Exception as exc:
            # 開始時にセッション作成や WS のハンドシェイクが失敗しても会議は止めない
            # （2026-09-28: 同時に流した 2 本が両方ともハンドシェイク待ちで落ち、
            # 文字起こしごと死んだ）。分離なしで始め、裏で接続を試し続ける。
            # 接続できるまでの音声は send_audio が時刻の基点に足す。
            logger.warning("pyannote Live-1: 開始時の接続に失敗しました (%s)。"
                           "分離なしで続行し、裏で接続を試します。", exc)
            self._ws = None
            self._reconnecting = True
            self._reconnect_thread = threading.Thread(
                target=self._reconnect_worker, args=(None, 0), daemon=True)
            self._reconnect_thread.start()

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
        self._connected_at = time.monotonic()
        self._dropped_ms = 0
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _handle_disconnect(self, exc: Exception) -> None:
        """送信失敗を検知した際に自動再接続を試みる（再接続数上限あり）.

        再接続そのもの（HTTP でセッション作成→WS 接続、1〜数秒）は別スレッドで
        行い、送信スレッドはすぐ戻る。再接続中は ``_ws`` が None なので
        ``send_audio`` は音声を捨てて基点だけ進める。溜め込んで後で一気に送ると
        サーバの「実時間より 5 秒以上先行」で切られるため、溜めない方が正しい。
        """
        logger.warning("pyannote Live-1: 送信中に切断を検知しました (%s)。", exc)
        old_ws = self._ws
        self._ws = None
        if not self.auto_reconnect or self._reconnects >= self.max_reconnects:
            logger.error(
                "pyannote Live-1: 再接続を行いません（auto_reconnect=%s, %d/%d回）。",
                self.auto_reconnect, self._reconnects, self.max_reconnects,
            )
            return
        self._reconnects += 1
        self._reconnecting = True
        with self._timeline_lock:
            # これまでに送信できた音声の累計msをオフセットとして次セッションに引き継ぐ。
            self._session_base_ms += self._sent_audio_ms
            self._sent_audio_ms = 0
        self._label_epoch += 1
        with self._active_lock:
            self._active_starts.clear()
        self._pcm_buf.clear()
        attempt = self._reconnects
        self._reconnect_thread = threading.Thread(
            target=self._reconnect_worker, args=(old_ws, attempt), daemon=True)
        self._reconnect_thread.start()

    _RETRY_BACKOFF_S = (2.0, 4.0, 8.0)

    def _reconnect_worker(self, old_ws: Any, attempt: int) -> None:
        """別スレッドで新セッションに繋ぎ直す。失敗したら間を置いて上限まで試す.

        attempt は「この呼び出しが何回目の再接続か」（0 は開始時の接続失敗からの
        再試行）。1 回の失敗で諦めると、サーバが数秒応答しないだけで会議の残り
        全部で分離が止まる。
        """
        with contextlib.suppress(Exception):
            if old_ws is not None:
                old_ws.close()
        if self._reader is not None and self._reader is not threading.current_thread():
            self._reader.join(timeout=1.0)
        self._stop.clear()
        try:
            while True:
                try:
                    self._connect()
                except Exception as exc:
                    if self._reconnects >= self.max_reconnects:
                        logger.error("pyannote Live-1: 再接続に失敗しました (%s)。"
                                     "上限 %d 回に達したので諦めます。", exc, self.max_reconnects)
                        self._ws = None
                        return
                    wait = self._RETRY_BACKOFF_S[min(self._reconnects, len(self._RETRY_BACKOFF_S) - 1)]
                    self._reconnects += 1
                    attempt = self._reconnects
                    logger.warning("pyannote Live-1: 接続に失敗しました (%s)。%.0f 秒後に再試行"
                                   "（%d/%d回目）。", exc, wait, attempt, self.max_reconnects)
                    if self._stop.wait(timeout=wait):
                        return
                    continue
                logger.warning(
                    "pyannote Live-1: 新セッションで%s (%d/%d回目、"
                    "音声内位置 %dms から再開、ラベルepoch=%d)。",
                    "接続しました" if attempt == 0 else "再接続しました",
                    attempt, self.max_reconnects,
                    self._session_base_ms, self._label_epoch,
                )
                return
        finally:
            self._reconnecting = False

    def send_audio(self, pcm16k: bytes) -> None:
        """16kHz mono PCM16 bytes を受け取り、Live-1 仕様の100ms固定 f32le
        チャンク(6400バイト)に再分割して送信する。

        呼び出し元のチャンク境界（マイク100ms / WAVシミュレーション120ms等）
        は仕様の100ms固定と一致しないことがあるため、内部バッファで吸収する。
        送信中にサーバ切断を検知した場合、``auto_reconnect`` が有効なら
        新セッションを作って同じチャンクを送り直す（自動再接続）。
        """
        if self._ws is None:
            # 未接続（接続処理中・再接続中・諦めた後）でも会議の時間は進んでいる。
            # 送らなかった分をタイムラインに足しておかないと、以後の区間の時刻が
            # STT より早い側にずれて重なり判定が崩れる（レビュー 2026-09-13:
            # 既定の UI 起動では STT 接続→分離接続の間の約1秒がこれに当たる）
            with self._timeline_lock:
                self._session_base_ms += len(pcm16k) // _SR_BYTES_PER_MS
            return
        self._pcm_buf.extend(pcm16k)
        while len(self._pcm_buf) >= _CHUNK_BYTES_PCM16:
            chunk = bytes(self._pcm_buf[:_CHUNK_BYTES_PCM16])
            del self._pcm_buf[:_CHUNK_BYTES_PCM16]
            payload = pcm16_to_pyannote_f32(chunk)
            if not payload:
                continue
            # Live-1 は「実時間より 5 秒以上先行した音声」を policy violation で切る。
            # 再接続の間に溜まった音声を一気に流すと必ずこれに当たり、再接続→
            # 溜まる→切断の悪循環で3回の上限を使い切って分離が死ぬ（2026-09-12 の
            # 実走と AMI 一括実行で再現）。先行しすぎる分は送らずに捨て、
            # その長さをタイムラインのオフセットに足して以後の区間の時刻を
            # 会議の時間に合わせ続ける。
            # 先行量は「サーバに実際に送った音声」と実時間の差で見る。捨てた音声を
            # 数えると、捨てても先行量が減らず、一度しきい値を超えた後は以後の
            # 全チャンクを捨て続けてサーバに何も届かなくなる。すると 5 秒の無音で
            # 1008（client timeout）で切られ、再接続→溜まる→全部捨てる→1008 の
            # 連鎖で 3 回の上限を使い切って分離が死ぬ（2026-09-28 の YouTube
            # 再生ラン 4 本中 3 本で再現。切断の起点はサーバ都合の 1011 だった）。
            elapsed_ms = (time.monotonic() - self._connected_at) * 1000.0
            if self._sent_audio_ms - elapsed_ms > self._MAX_AHEAD_MS:
                self._dropped_ms += _CHUNK_MS
                with self._timeline_lock:
                    self._session_base_ms += _CHUNK_MS
                if self._dropped_ms == _CHUNK_MS:
                    logger.warning("pyannote Live-1: 実時間より先行した音声を捨てて追従します"
                                   "（再接続直後の溜まり分）。")
                continue
            try:
                self._ws.send(payload)
            except Exception as exc:
                self._handle_disconnect(exc)
                # このチャンクと、バッファに残った端数は送れない。時刻だけ進める
                with self._timeline_lock:
                    self._session_base_ms += _CHUNK_MS + len(self._pcm_buf) // _SR_BYTES_PER_MS
                self._pcm_buf.clear()
                return
            self._sent_audio_ms += _CHUNK_MS
            # 1分間安定して送れたら再接続カウンタを忘れる（§48.5）。
            # 上限3回は「連続失敗の暴走止め」であって生涯回数ではない。
            # 忘れないと、数時間の会議で散発的な瞬断が3回起きただけで
            # 分離が残り全部で無言のまま死ぬ（send_audio は例外を
            # 呼び出し側で握り潰されるため気づけない）。
            if self._reconnects and self._sent_audio_ms >= self._RECONNECT_FORGET_MS:
                self._reconnects = 0

    @property
    def alive(self) -> bool:
        """接続が生きているか（再接続中は True、諦めた後は False, §48.5）."""
        return self._ws is not None or self._reconnecting

    def drain_events(self) -> list[DiarizationEvent]:
        events: list[DiarizationEvent] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def active_events(self) -> list[DiarizationEvent]:
        with self._active_lock:
            items = list(self._active_starts.items())
        return [DiarizationEvent(start_ms, None, speaker, self.name)
                for speaker, start_ms in items]

    def close(self) -> None:
        """end_of_stream を送り、サーバが確定イベントを出し切って自発的に
        close(code 1000)するのを少し待ってからソケットを閉じる。

        仕様(docs.pyannote.ai/tutorials/streaming-real-time)は
        「end_of_stream送信後、サーバは残りのイベントを出し切ってから閉じる。
        生ソケットを即座に閉じると最終出力を失いうる」と明記しているため、
        _stop を即セットしてreaderを止めるのではなく、reader(recvループ)が
        サーバ側クローズで自然終了するのを timeout 付きで待ってから閉じる。
        """
        if self._reconnect_thread is not None and self._reconnect_thread.is_alive():
            if self._ws is None:
                self._stop.set()          # 再試行の待ちを打ち切る
            self._reconnect_thread.join(timeout=5.0)
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
            try:
                event = self._parse_message(raw)
            except Exception:
                # 不正なフレーム（非JSON・非UTF-8・想定外の型）1つで reader
                # スレッドが死ぬと、以後の diarization イベントが無言で止まり、
                # 帰属が STT ラベルへ静かに縮退する（レビュー 2026-07-30）。
                # フレームは捨てて読み続ける。
                logger.warning("pyannote Live-1: 解釈できないフレームを破棄: %r",
                               raw[:120] if isinstance(raw, (str, bytes)) else raw)
                continue
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
            with self._active_lock:
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
        with self._active_lock:
            start_ms = self._active_starts.pop(speaker, None)
        if start_ms is None:
            logger.warning(
                "pyannote Live-1: speaker=%s の speaker_end (ts=%.3fs) に対応する"
                " speaker_start がありません。縮退セグメント化を避けるため破棄します。",
                speaker, timestamp if isinstance(timestamp, int | float) else -1.0,
            )
            return None
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

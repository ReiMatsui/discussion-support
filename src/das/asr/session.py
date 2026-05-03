"""1 セッション分の AudioProcessor ラッパ。

責務:
  - マイクから来る PCM/エンコード済みバイト列を ``push_audio`` で投入する
  - WhisperLiveKit が返す ``lines`` (確定行) を差分検出して、新しく
    確定した行を ``Utterance`` として ``iter_utterances`` から yield する
  - 中間 (途中の文字起こし) は ``on_partial`` コールバックで通知

WhisperLiveKit の出力スキーマ (要点だけ):
  - ``lines``: ``[{"speaker": int, "text": str, "start": "H:MM:SS", "end": ...}]``
    既定の ``mode=full`` では毎フレーム全件が入っているので、こちらで
    「直近何行目まで emit したか」を記憶して差分を取る
  - ``buffer_transcription``: 途中の (まだ確定していない) 文字列
  - ``status``: ``"active_transcription"`` / ``"no_audio_detected"``
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, Union

from das.asr.engine import get_engine
from das.logging import get_logger
from das.types import Utterance

_log = get_logger("das.asr.session")

PartialCallback = Callable[[str], Union[Awaitable[None], None]]


class LiveAsrSession:
    """1 マイク入力 = 1 インスタンス。

    使い方::

        session = LiveAsrSession()
        await session.start()
        # 別タスクでマイク → push_audio
        async for utt in session.iter_utterances():
            await bus.publish(utt)
        await session.cleanup()
    """

    def __init__(
        self,
        *,
        engine: Any | None = None,
        speaker: str = "speaker_1",
        on_partial: PartialCallback | None = None,
    ) -> None:
        # 遅延 import: [asr] extras なしの環境では import エラーになるが、
        # その場合 LiveAsrSession を生成しなければ実害はない (本体テストは安全)
        from whisperlivekit import AudioProcessor

        self._engine = engine or get_engine()
        self._processor = AudioProcessor(transcription_engine=self._engine)
        self._speaker = speaker
        self._on_partial = on_partial
        self._next_turn_id = 0
        self._results_generator: AsyncIterator[Any] | None = None

    async def start(self) -> None:
        """背後タスクを起動する。``iter_utterances`` の前に呼ぶ。"""

        if self._results_generator is not None:
            raise RuntimeError("LiveAsrSession is already started")
        self._results_generator = await self._processor.create_tasks()
        _log.info("asr.session.started", speaker=self._speaker)

    async def push_audio(self, chunk: bytes) -> None:
        """マイクから来た音声バイト列を投入する。"""

        await self._processor.process_audio(chunk)

    async def stop(self) -> None:
        """音声入力終了を通知する (空フレーム = EOS)。

        ``iter_utterances`` のループはこの後、確定残りを吐き出してから終わる。
        """

        try:
            await self._processor.process_audio(b"")
        except Exception:  # pragma: no cover - 防御的
            _log.warning("asr.session.eos_failed", exc_info=True)

    async def cleanup(self) -> None:
        """AudioProcessor を破棄する。"""

        try:
            await self._processor.cleanup()
        except Exception:  # pragma: no cover
            _log.warning("asr.session.cleanup_failed", exc_info=True)

    async def iter_utterances(self) -> AsyncIterator[Utterance]:
        """確定行を ``Utterance`` として 1 件ずつ yield する。

        生成終わりの判定は WhisperLiveKit の results_generator 終了に委ねる。
        """

        if self._results_generator is None:
            raise RuntimeError("LiveAsrSession.start() を先に呼んでください")

        async for response in self._results_generator:
            payload = self._to_dict(response)

            # 1) 確定済み lines を差分で emit
            lines = payload.get("lines") or []
            for i in range(self._next_turn_id, len(lines)):
                line = lines[i]
                text = (line.get("text") or "").strip()
                if not text:
                    # 沈黙プレースホルダ (text=None, speaker=-2) などはスキップ
                    continue
                yield Utterance(
                    turn_id=i,
                    speaker=self._resolve_speaker(line),
                    text=text,
                    timestamp=datetime.now(timezone.utc),
                )
            self._next_turn_id = len(lines)

            # 2) 途中バッファをコールバックに通知
            partial = payload.get("buffer_transcription")
            if partial and self._on_partial is not None:
                result = self._on_partial(partial)
                if asyncio.iscoroutine(result):
                    await result

    # --- 内部ヘルパ ---------------------------------------------------

    def _resolve_speaker(self, line: dict[str, Any]) -> str:
        """``line.speaker`` (int) を文字列話者IDに正規化する。

        ダイアライゼーション無効時は基本的に 0 か 1 のどちらかなので、
        負/未指定なら固定話者にフォールバック。
        """

        speaker_id = line.get("speaker")
        if isinstance(speaker_id, int) and speaker_id > 0:
            return f"speaker_{speaker_id}"
        return self._speaker

    @staticmethod
    def _to_dict(response: Any) -> dict[str, Any]:
        if hasattr(response, "to_dict"):
            return response.to_dict()  # type: ignore[no-any-return]
        if isinstance(response, dict):
            return response
        return {}


__all__ = ["LiveAsrSession", "PartialCallback"]

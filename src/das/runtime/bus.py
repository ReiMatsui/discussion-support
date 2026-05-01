"""非常にシンプルな asyncio ベースのイベントバス。

設計:
  - handler はイベント型に対して登録 (exact 型マッチ)
  - ``publish(event)`` で該当ハンドラを ``asyncio.create_task`` する
  - ``drain()`` は in-flight タスクが全て終わるまでループ
    (handler 内で publish された二次イベントも待つ)
  - handler が例外を投げた場合は drain 時に最初の例外を再 raise

研究プロトタイプ向けに最小限。スレッドセーフ性は保証しない (asyncio 単一ループ前提)。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from das.logging import get_logger

EventHandler = Callable[[Any], Awaitable[None]]


class EventBus:
    """exact 型マッチの軽量 pub/sub。"""

    def __init__(self) -> None:
        self._handlers: dict[type, list[EventHandler]] = {}
        self._inflight: set[asyncio.Task[None]] = set()
        self._errors: list[BaseException] = []
        self._log = get_logger("das.runtime.bus")

    # --- 登録 ----------------------------------------------------------

    def subscribe(self, event_type: type, handler: EventHandler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    # --- 発行 ----------------------------------------------------------

    async def publish(self, event: Any) -> None:
        """イベントを配信する。マッチするハンドラがいなければ no-op。"""

        handlers = self._handlers.get(type(event), [])
        if not handlers:
            return
        for handler in handlers:
            task = asyncio.create_task(self._invoke(handler, event))
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)

    async def drain(self) -> None:
        """全ての in-flight ハンドラが終わるまで待つ。

        ハンドラ内で新しい publish が発生した場合、それも拾って待ち続ける。
        全タスク完了後、いずれかが例外で終わっていれば最初の 1 件を再 raise する。
        """

        while self._inflight:
            current = list(self._inflight)
            await asyncio.gather(*current, return_exceptions=True)

        if self._errors:
            error = self._errors[0]
            self._errors = []
            raise error

    # --- 内部 ----------------------------------------------------------

    async def _invoke(self, handler: EventHandler, event: Any) -> None:
        try:
            await handler(event)
        except Exception as exc:
            self._log.error(
                "bus.handler_error",
                handler=getattr(handler, "__qualname__", repr(handler)),
                event_type=type(event).__name__,
                error=str(exc),
            )
            self._errors.append(exc)


__all__ = ["EventBus", "EventHandler"]

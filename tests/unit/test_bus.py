"""EventBus のユニットテスト。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from das.runtime.bus import EventBus


@dataclass(frozen=True)
class _Foo:
    value: int


@dataclass(frozen=True)
class _Bar:
    text: str


async def test_publish_invokes_matching_handler() -> None:
    bus = EventBus()
    received: list[int] = []

    async def handler(event: _Foo) -> None:
        received.append(event.value)

    bus.subscribe(_Foo, handler)
    await bus.publish(_Foo(value=42))
    await bus.drain()

    assert received == [42]


async def test_publish_does_nothing_for_unmatched_type() -> None:
    bus = EventBus()
    called: list[object] = []

    async def handler(event: _Foo) -> None:
        called.append(event)

    bus.subscribe(_Foo, handler)
    await bus.publish(_Bar(text="x"))
    await bus.drain()

    assert called == []


async def test_multiple_handlers_invoked_for_same_event() -> None:
    bus = EventBus()
    h1: list[int] = []
    h2: list[int] = []

    async def first(event: _Foo) -> None:
        h1.append(event.value)

    async def second(event: _Foo) -> None:
        h2.append(event.value * 10)

    bus.subscribe(_Foo, first)
    bus.subscribe(_Foo, second)
    await bus.publish(_Foo(value=3))
    await bus.drain()

    assert h1 == [3]
    assert h2 == [30]


async def test_handler_can_publish_secondary_event() -> None:
    """ハンドラ内 publish も drain で待たれることを確認。"""

    bus = EventBus()
    seen_bars: list[str] = []

    async def on_foo(event: _Foo) -> None:
        await bus.publish(_Bar(text=f"from-{event.value}"))

    async def on_bar(event: _Bar) -> None:
        seen_bars.append(event.text)

    bus.subscribe(_Foo, on_foo)
    bus.subscribe(_Bar, on_bar)

    await bus.publish(_Foo(value=1))
    await bus.publish(_Foo(value=2))
    await bus.drain()

    assert sorted(seen_bars) == ["from-1", "from-2"]


async def test_handler_exceptions_surface_in_drain() -> None:
    bus = EventBus()

    async def boom(event: _Foo) -> None:
        raise RuntimeError("boom")

    bus.subscribe(_Foo, boom)
    await bus.publish(_Foo(value=1))

    with pytest.raises(RuntimeError, match="boom"):
        await bus.drain()


async def test_drain_with_no_publishes_is_noop() -> None:
    bus = EventBus()
    await bus.drain()  # 例外なく終わる


async def test_handlers_run_concurrently() -> None:
    """複数ハンドラが直列ではなく並行に実行されることを確認 (asyncio スケジューリング)。"""

    bus = EventBus()
    timeline: list[str] = []

    async def slow(event: _Foo) -> None:
        timeline.append("slow.start")
        await asyncio.sleep(0.05)
        timeline.append("slow.end")

    async def fast(event: _Foo) -> None:
        timeline.append("fast.start")
        timeline.append("fast.end")

    bus.subscribe(_Foo, slow)
    bus.subscribe(_Foo, fast)
    await bus.publish(_Foo(value=1))
    await bus.drain()

    # fast.end は slow.end より前に出る (並行実行されている)
    assert timeline.index("fast.end") < timeline.index("slow.end")

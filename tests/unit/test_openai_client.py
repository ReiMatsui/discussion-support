"""OpenAIClient のユニットテスト。

外部 API は呼ばず、AsyncMock で AsyncOpenAI を差し替えてラッパの挙動を確認する。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import APITimeoutError, RateLimitError
from pydantic import BaseModel

from das.llm import OpenAIClient


def _completion_response(content: str = "hi") -> SimpleNamespace:
    """``chat.completions.create`` の応答を模した構造を返す。"""

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
    )


def _parse_response(parsed: BaseModel) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(parsed=parsed),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=20,
        ),
    )


def _fake_async_client(
    *,
    create: AsyncMock | None = None,
    parse: AsyncMock | None = None,
) -> MagicMock:
    """AsyncOpenAI を模したオブジェクトを返す。"""

    fake = MagicMock()
    fake.chat = MagicMock()
    fake.chat.completions = MagicMock()
    fake.chat.completions.create = create or AsyncMock()
    fake.beta = MagicMock()
    fake.beta.chat = MagicMock()
    fake.beta.chat.completions = MagicMock()
    fake.beta.chat.completions.parse = parse or AsyncMock()
    return fake


def _rate_limit_error(message: str = "rate limited") -> RateLimitError:
    """SDK の RateLimitError を実体化するためのヘルパ。"""

    response = httpx.Response(
        status_code=429,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    return RateLimitError(message=message, response=response, body=None)


def _timeout_error() -> APITimeoutError:
    return APITimeoutError(
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    )


# --- モデル解決 -----------------------------------------------------------


def test_default_models_from_settings() -> None:
    client = OpenAIClient(client=_fake_async_client())
    assert client.fast_model == "gpt-4o-mini"
    assert client.smart_model == "gpt-4o"


# --- chat (plain text) ----------------------------------------------------


async def test_chat_returns_content_and_uses_fast_model_by_default() -> None:
    create = AsyncMock(return_value=_completion_response("こんにちは"))
    fake = _fake_async_client(create=create)
    client = OpenAIClient(client=fake)

    result = await client.chat([{"role": "user", "content": "hi"}])

    assert result == "こんにちは"
    create.assert_awaited_once()
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == client.fast_model
    assert kwargs["temperature"] == 0.0


async def test_chat_respects_explicit_model() -> None:
    create = AsyncMock(return_value=_completion_response())
    fake = _fake_async_client(create=create)
    client = OpenAIClient(client=fake)

    await client.chat([{"role": "user", "content": "x"}], model="gpt-5-mini")

    assert create.await_args.kwargs["model"] == "gpt-5-mini"


async def test_chat_returns_empty_string_when_content_is_none() -> None:
    create = AsyncMock(return_value=_completion_response(content=None))  # type: ignore[arg-type]
    fake = _fake_async_client(create=create)
    client = OpenAIClient(client=fake)

    assert await client.chat([{"role": "user", "content": "x"}]) == ""


# --- chat_structured ------------------------------------------------------


class _DummyOutput(BaseModel):
    relation: str
    confidence: float


async def test_chat_structured_returns_parsed_pydantic() -> None:
    expected = _DummyOutput(relation="support", confidence=0.92)
    parse = AsyncMock(return_value=_parse_response(expected))
    fake = _fake_async_client(parse=parse)
    client = OpenAIClient(client=fake)

    result = await client.chat_structured(
        [{"role": "user", "content": "判定して"}],
        response_format=_DummyOutput,
    )

    assert result == expected
    parse.assert_awaited_once()
    kwargs = parse.await_args.kwargs
    assert kwargs["response_format"] is _DummyOutput
    assert kwargs["model"] == client.fast_model


async def test_chat_structured_raises_when_parsed_is_none() -> None:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=None))],
        usage=None,
    )
    parse = AsyncMock(return_value=response)
    fake = _fake_async_client(parse=parse)
    client = OpenAIClient(client=fake)

    with pytest.raises(RuntimeError):
        await client.chat_structured(
            [{"role": "user", "content": "x"}],
            response_format=_DummyOutput,
        )


# --- リトライ ------------------------------------------------------------


async def test_retry_succeeds_after_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # tenacity の wait を 0 にして高速化
    import tenacity

    monkeypatch.setattr(tenacity, "nap", SimpleNamespace(sleep=lambda _: None))

    create = AsyncMock(side_effect=[_rate_limit_error(), _completion_response("ok")])
    fake = _fake_async_client(create=create)
    client = OpenAIClient(client=fake, max_retries=3)

    result = await client.chat([{"role": "user", "content": "x"}])
    assert result == "ok"
    assert create.await_count == 2


async def test_retry_gives_up_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    import tenacity

    monkeypatch.setattr(tenacity, "nap", SimpleNamespace(sleep=lambda _: None))

    create = AsyncMock(side_effect=_timeout_error())
    fake = _fake_async_client(create=create)
    client = OpenAIClient(client=fake, max_retries=2)

    with pytest.raises(APITimeoutError):
        await client.chat([{"role": "user", "content": "x"}])
    assert create.await_count == 2


async def test_non_retryable_error_propagates_immediately() -> None:
    create = AsyncMock(side_effect=ValueError("boom"))
    fake = _fake_async_client(create=create)
    client = OpenAIClient(client=fake, max_retries=3)

    with pytest.raises(ValueError):
        await client.chat([{"role": "user", "content": "x"}])
    assert create.await_count == 1

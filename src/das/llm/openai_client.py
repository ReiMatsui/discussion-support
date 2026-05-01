"""OpenAI API への薄いラッパ。

- 既定モデルは Settings (`OPENAI_MODEL_FAST` / `OPENAI_MODEL_SMART`) から解決
- リトライ: tenacity で RateLimit / Connection / Timeout の 3 種だけ再試行
- トークン使用量を構造化ログに記録
- 構造化出力: pydantic モデルを ``response_format`` に渡し、パース済みインスタンスを返す
- テスト容易性のため、内部の ``AsyncOpenAI`` を DI 可能にしてある
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from das.logging import get_logger
from das.settings import Settings, get_settings

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

T = TypeVar("T", bound=BaseModel)


_RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
)


class OpenAIClient:
    """非同期前提の OpenAI クライアントラッパ。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: AsyncOpenAI | None = None,
        max_retries: int = 3,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client or AsyncOpenAI(api_key=self._settings.openai_api_key)
        self._log = get_logger("das.llm.openai")
        self._max_retries = max_retries

    # --- モデル解決 -----------------------------------------------------

    @property
    def fast_model(self) -> str:
        """抽出・連結など量産で使う既定モデル。"""

        return self._settings.openai_model_fast

    @property
    def smart_model(self) -> str:
        """ファシリテーションなど推論重視で使う上位モデル。"""

        return self._settings.openai_model_smart

    @property
    def embedding_model(self) -> str:
        """既定の埋め込みモデル。"""

        return "text-embedding-3-small"

    # --- 公開メソッド ---------------------------------------------------

    async def chat(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """プレーンテキストの応答を返す。"""

        chosen_model = model or self.fast_model
        response: Any = None
        async for attempt in self._retrier():
            with attempt:
                response = await self._client.chat.completions.create(
                    model=chosen_model,
                    messages=messages,
                    temperature=temperature,
                )
        if response is None:  # pragma: no cover - 防御的
            raise RuntimeError("openai client returned no response")
        self._log_usage(response, chosen_model)
        return response.choices[0].message.content or ""

    async def chat_structured(
        self,
        messages: list[ChatCompletionMessageParam],
        response_format: type[T],
        *,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> T:
        """pydantic モデルとして構造化出力を返す。

        ``response_format`` には ``BaseModel`` のサブクラスを渡す。
        OpenAI 側で JSON Schema に変換されてバリデーション済みのまま返ってくる。
        """

        chosen_model = model or self.fast_model
        response: Any = None
        async for attempt in self._retrier():
            with attempt:
                response = await self._client.beta.chat.completions.parse(
                    model=chosen_model,
                    messages=messages,
                    response_format=response_format,
                    temperature=temperature,
                )
        if response is None:  # pragma: no cover - 防御的
            raise RuntimeError("openai client returned no response")
        self._log_usage(response, chosen_model)
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("structured output returned no parsed result")
        return parsed

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """テキスト群の埋め込みベクトルを返す。"""

        if not texts:
            return []
        chosen_model = model or self.embedding_model
        response: Any = None
        async for attempt in self._retrier():
            with attempt:
                response = await self._client.embeddings.create(
                    model=chosen_model,
                    input=texts,
                )
        if response is None:  # pragma: no cover - 防御的
            raise RuntimeError("openai client returned no response")
        self._log_embedding_usage(response, chosen_model, len(texts))
        return [item.embedding for item in response.data]

    async def embed_one(self, text: str, *, model: str | None = None) -> list[float]:
        """単一テキストの埋め込み。"""

        result = await self.embed([text], model=model)
        return result[0]

    # --- 内部 -----------------------------------------------------------

    def _retrier(self) -> AsyncRetrying:
        return AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
            reraise=True,
        )

    def _log_usage(self, response: Any, model: str) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self._log.info(
            "openai.usage",
            model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )

    def _log_embedding_usage(self, response: Any, model: str, n_inputs: int) -> None:
        usage = getattr(response, "usage", None)
        self._log.info(
            "openai.embedding_usage",
            model=model,
            n_inputs=n_inputs,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            total_tokens=getattr(usage, "total_tokens", None) if usage else None,
        )


__all__ = ["OpenAIClient"]

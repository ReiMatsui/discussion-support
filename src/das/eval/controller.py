"""議論シミュレーションのターン制コントローラ。

Persona 群とトピックを受け取り、round-robin で発話させて transcript を作る。
``InfoProvider`` を差し替えると、各ターンの発話直前に「参考情報」を挿入できる。
これが M2.3 の 3 条件 (None / FlatRAG / FullProposal) の差し込み口になる。

API:
  - ``SessionRunner.run(...)``           : 同期的に走らせて transcript を返す
  - ``SessionRunner.run_streaming(...)`` : 各 Utterance が出来るたびに yield する
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from das.eval.persona import PersonaAgent, PersonaSpec
from das.llm import OpenAIClient
from das.logging import get_logger
from das.types import Utterance

# 履歴と次の発話者の persona から参考情報を返す関数。None なら情報なし。
InfoProvider = Callable[[list[Utterance], PersonaSpec], Awaitable[str | None]]


@dataclass(frozen=True)
class SessionConfig:
    """1 セッションの実行パラメータ。"""

    topic: str
    max_turns: int = 8
    temperature: float = 0.7
    model: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class SessionRunner:
    """ターン制で persona に発話させるシミュレーションコントローラ。"""

    def __init__(
        self,
        personas: list[PersonaSpec],
        config: SessionConfig,
        *,
        llm: OpenAIClient | None = None,
    ) -> None:
        if not personas:
            raise ValueError("personas must not be empty")
        if config.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        self._personas = personas
        self._config = config
        self._llm = llm or OpenAIClient()
        self._log = get_logger("das.eval.controller")

    @property
    def personas(self) -> list[PersonaSpec]:
        return list(self._personas)

    @property
    def config(self) -> SessionConfig:
        return self._config

    async def run(
        self,
        *,
        info_provider: InfoProvider | None = None,
    ) -> list[Utterance]:
        """全ターンをまとめて走らせ、transcript を返す。"""

        transcript: list[Utterance] = []
        async for u in self.run_streaming(info_provider=info_provider):
            transcript.append(u)
        return transcript

    async def run_streaming(
        self,
        *,
        info_provider: InfoProvider | None = None,
    ) -> AsyncIterator[Utterance]:
        """各ターンの ``Utterance`` を完成次第 yield する。

        ``info_provider`` は ``await`` 可能。条件ごとに異なる情報注入をする際に使う。
        """

        agents = [PersonaAgent(spec, llm=self._llm) for spec in self._personas]
        history: list[Utterance] = []
        self._log.info(
            "session.start",
            n_personas=len(agents),
            max_turns=self._config.max_turns,
            topic_chars=len(self._config.topic),
        )

        for turn_id in range(1, self._config.max_turns + 1):
            agent = agents[(turn_id - 1) % len(agents)]
            info = await info_provider(history, agent.spec) if info_provider is not None else None
            utterance = await agent.utter(
                history,
                self._config.topic,
                info=info,
                turn_id=turn_id,
                model=self._config.model,
                temperature=self._config.temperature,
            )
            history.append(utterance)
            self._log.info(
                "session.turn",
                turn_id=turn_id,
                speaker=utterance.speaker,
                info_provided=info is not None,
            )
            yield utterance

        self._log.info("session.done", n_utterances=len(history))


__all__ = ["InfoProvider", "SessionConfig", "SessionRunner"]

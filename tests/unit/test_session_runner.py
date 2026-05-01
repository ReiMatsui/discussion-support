"""SessionRunner のユニットテスト。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from das.eval import (
    PersonaSpec,
    SessionConfig,
    SessionRunner,
    build_persona,
)
from das.llm import OpenAIClient
from das.types import Utterance


def _fake_llm(reply: str | list[str] = "発言") -> OpenAIClient:
    client = OpenAIClient(client=MagicMock())
    if isinstance(reply, str):
        client.chat = AsyncMock(return_value=reply)  # type: ignore[method-assign]
    else:
        client.chat = AsyncMock(side_effect=reply)  # type: ignore[method-assign]
    return client


@pytest.fixture
def personas() -> list[PersonaSpec]:
    return [
        build_persona(name="A", stance="pro", focus="環境"),
        build_persona(name="B", stance="con", focus="コスト"),
    ]


def _config(**kwargs: object) -> SessionConfig:
    defaults: dict = {"topic": "プラ容器", "max_turns": 4, "temperature": 0.0}
    defaults.update(kwargs)  # type: ignore[arg-type]
    return SessionConfig(**defaults)  # type: ignore[arg-type]


# --- ガード ------------------------------------------------------------


def test_runner_rejects_empty_personas() -> None:
    with pytest.raises(ValueError):
        SessionRunner([], _config(), llm=_fake_llm())


def test_runner_rejects_zero_turns(personas: list[PersonaSpec]) -> None:
    with pytest.raises(ValueError):
        SessionRunner(personas, _config(max_turns=0), llm=_fake_llm())


# --- 基本動作 ---------------------------------------------------------


async def test_run_returns_max_turns_utterances(personas: list[PersonaSpec]) -> None:
    llm = _fake_llm("ok")
    runner = SessionRunner(personas, _config(max_turns=5), llm=llm)
    transcript = await runner.run()
    assert len(transcript) == 5
    assert all(isinstance(u, Utterance) for u in transcript)


async def test_run_round_robin_speaker(personas: list[PersonaSpec]) -> None:
    """話者は personas を順番に巡回する。"""

    llm = _fake_llm("ok")
    runner = SessionRunner(personas, _config(max_turns=5), llm=llm)
    transcript = await runner.run()
    speakers = [u.speaker for u in transcript]
    assert speakers == ["A", "B", "A", "B", "A"]


async def test_run_streaming_yields_each_turn(
    personas: list[PersonaSpec],
) -> None:
    llm = _fake_llm(["t1", "t2", "t3"])
    runner = SessionRunner(personas, _config(max_turns=3), llm=llm)
    seen: list[Utterance] = []
    async for u in runner.run_streaming():
        seen.append(u)
    assert [u.text for u in seen] == ["t1", "t2", "t3"]
    assert [u.turn_id for u in seen] == [1, 2, 3]


async def test_run_streaming_passes_history(personas: list[PersonaSpec]) -> None:
    """各ターンの persona 呼び出しで、累積履歴が渡されている。"""

    llm = _fake_llm(["t1", "t2", "t3"])
    runner = SessionRunner(personas, _config(max_turns=3), llm=llm)
    await runner.run()

    # llm.chat が 3 回呼ばれ、3 回目の messages には turn 1, 2 が含まれる
    third_call = llm.chat.await_args_list[2]
    user_content = third_call.args[0][1]["content"]
    assert "[turn 1]" in user_content
    assert "[turn 2]" in user_content


# --- info_provider ----------------------------------------------------


async def test_info_provider_called_per_turn(personas: list[PersonaSpec]) -> None:
    llm = _fake_llm("ok")
    runner = SessionRunner(personas, _config(max_turns=3), llm=llm)

    seen_personas: list[str] = []

    async def provider(history: list[Utterance], spec: PersonaSpec) -> str | None:
        seen_personas.append(spec.name)
        return f"参考情報 turn{len(history) + 1}"

    await runner.run(info_provider=provider)
    # 3 ターンで 3 回呼ばれる
    assert seen_personas == ["A", "B", "A"]

    # info が prompt に入っていることを確認
    first_call_user = llm.chat.await_args_list[0].args[0][1]["content"]
    assert "参考情報 turn1" in first_call_user


async def test_info_provider_none_means_no_info(personas: list[PersonaSpec]) -> None:
    llm = _fake_llm("ok")
    runner = SessionRunner(personas, _config(max_turns=2), llm=llm)
    await runner.run(info_provider=None)
    # 参考情報セクションが含まれていない
    user_content = llm.chat.await_args_list[0].args[0][1]["content"]
    assert "参考情報" not in user_content

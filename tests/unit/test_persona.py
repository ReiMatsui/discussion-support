"""PersonaAgent / PersonaSpec のユニットテスト。"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from das.eval.persona import PersonaAgent, PersonaSpec, build_persona
from das.eval.presets import cafeteria_personas, policy_ai_lecture_personas
from das.llm import OpenAIClient
from das.types import Utterance


def _fake_llm() -> OpenAIClient:
    return OpenAIClient(client=MagicMock())


# --- PersonaSpec / factory ----------------------------------------------


def test_build_persona_defaults() -> None:
    spec = build_persona(name="A", stance="pro", focus="環境")
    assert spec.name == "A"
    assert spec.stance == "pro"
    assert spec.focus == "環境"
    assert spec.personality  # 既定値が入っている
    assert spec.metadata == {}


def test_persona_spec_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    spec = build_persona(name="X")
    with pytest.raises(FrozenInstanceError):
        spec.name = "Y"  # type: ignore[misc]


# --- プリセット ---------------------------------------------------------


def test_cafeteria_presets_have_three_stances() -> None:
    personas = cafeteria_personas()
    assert len(personas) == 3
    stances = {p.stance for p in personas}
    assert stances == {"pro", "con", "neutral"}


def test_policy_presets_at_least_four() -> None:
    personas = policy_ai_lecture_personas()
    assert len(personas) >= 3
    # 賛成・反対・中立の 3 立場すべてが揃う
    assert {p.stance for p in personas} == {"pro", "con", "neutral"}


# --- PersonaAgent.utter -------------------------------------------------


@pytest.fixture
def spec_a() -> PersonaSpec:
    return build_persona(
        name="A",
        stance="pro",
        focus="環境負荷",
        personality="長期視点",
        extra="廃棄物問題に問題意識",
    )


async def test_utter_returns_utterance(spec_a: PersonaSpec) -> None:
    llm = _fake_llm()
    llm.chat = AsyncMock(return_value="プラ容器を廃止すべきだと思います。")  # type: ignore[method-assign]

    agent = PersonaAgent(spec_a, llm=llm)
    history: list[Utterance] = []
    u = await agent.utter(history, topic="プラ容器を廃止すべきか")

    assert isinstance(u, Utterance)
    assert u.speaker == "A"
    assert u.text == "プラ容器を廃止すべきだと思います。"
    assert u.turn_id == 1


async def test_utter_strips_whitespace(spec_a: PersonaSpec) -> None:
    llm = _fake_llm()
    llm.chat = AsyncMock(return_value="  発言です  \n")  # type: ignore[method-assign]
    agent = PersonaAgent(spec_a, llm=llm)
    u = await agent.utter([], topic="t")
    assert u.text == "発言です"


async def test_utter_uses_history_in_prompt(spec_a: PersonaSpec) -> None:
    llm = _fake_llm()
    captured = AsyncMock(return_value="新しい発言")
    llm.chat = captured  # type: ignore[method-assign]
    agent = PersonaAgent(spec_a, llm=llm)
    history = [
        Utterance(turn_id=1, speaker="B", text="コストが心配です"),
        Utterance(turn_id=2, speaker="C", text="折衷案を考えましょう"),
    ]
    await agent.utter(history, topic="t")

    messages = captured.await_args.args[0]
    user_content = messages[1]["content"]
    assert "B: コストが心配です" in user_content
    assert "C: 折衷案を考えましょう" in user_content


async def test_utter_includes_info_when_provided(spec_a: PersonaSpec) -> None:
    llm = _fake_llm()
    captured = AsyncMock(return_value="ok")
    llm.chat = captured  # type: ignore[method-assign]
    agent = PersonaAgent(spec_a, llm=llm)
    await agent.utter([], topic="t", info="X 大学では 2 年目にコストが解消した")

    user_content = captured.await_args.args[0][1]["content"]
    assert "参考情報" in user_content
    assert "X 大学" in user_content


async def test_utter_explicit_turn_id(spec_a: PersonaSpec) -> None:
    llm = _fake_llm()
    llm.chat = AsyncMock(return_value="hi")  # type: ignore[method-assign]
    agent = PersonaAgent(spec_a, llm=llm)
    u = await agent.utter([], topic="t", turn_id=42)
    assert u.turn_id == 42


async def test_system_prompt_contains_persona_fields(spec_a: PersonaSpec) -> None:
    llm = _fake_llm()
    captured = AsyncMock(return_value="ok")
    llm.chat = captured  # type: ignore[method-assign]
    agent = PersonaAgent(spec_a, llm=llm)
    await agent.utter([], topic="プラ容器を廃止すべきか")

    system_msg = captured.await_args.args[0][0]
    assert system_msg["role"] == "system"
    sys_content = system_msg["content"]
    assert "A" in sys_content
    assert "環境負荷" in sys_content
    assert "長期視点" in sys_content
    assert "プラ容器を廃止すべきか" in sys_content

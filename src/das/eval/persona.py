"""議論シミュレーション用の Persona 定義と発話生成。

設計:
  - ``PersonaSpec``: 名前・立場・関心点・性格を保持する frozen dataclass
  - ``build_persona(...)``: パラメータから Spec を組み立てる factory
  - ``PersonaAgent``: Spec と LLM クライアントを束ね、対話履歴と
    (任意の) 参考情報から次の発話を生成する
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from das.llm import OpenAIClient
from das.logging import get_logger
from das.types import Utterance

_PROMPTS_DIR = Path(__file__).parent / "prompts"

Stance = Literal["pro", "con", "neutral"]

_STANCE_DESCRIPTION: dict[Stance, str] = {
    "pro": "提案や変更に賛成の立場 (現状を変える側を後押しする)",
    "con": "提案や変更に反対の立場 (現状維持またはコスト・リスクを重視する)",
    "neutral": "中立。両方の論点に耳を傾け、論拠の強さで判断する",
}


@dataclass(frozen=True)
class PersonaSpec:
    """1 人の議論参加者の設計仕様。"""

    name: str
    stance: Stance
    focus: str
    """重視している論点や関心領域 (短く 1 文)。"""

    personality: str = "落ち着いて論理的"
    extra: str = ""
    """追加で含めたい背景情報 (例: 役職、過去の経験など)。空可。"""

    metadata: dict[str, str] = field(default_factory=dict)


def build_persona(
    name: str,
    *,
    stance: Stance = "neutral",
    focus: str = "総合的なバランス",
    personality: str = "落ち着いて論理的",
    extra: str = "",
    metadata: dict[str, str] | None = None,
) -> PersonaSpec:
    """``PersonaSpec`` の factory。引数のデフォルトを 1 箇所で管理する。"""

    return PersonaSpec(
        name=name,
        stance=stance,
        focus=focus,
        personality=personality,
        extra=extra,
        metadata=metadata or {},
    )


def _load_system_template() -> str:
    return (_PROMPTS_DIR / "persona.md").read_text(encoding="utf-8")


def _format_history(history: list[Utterance], max_turns: int = 12) -> str:
    if not history:
        return "(まだ発言なし — 最初の発言として、トピックに対する自分の立場を 1〜2 文で述べてください)"
    recent = history[-max_turns:]
    lines = [f"[turn {u.turn_id}] {u.speaker}: {u.text}" for u in recent]
    return "\n".join(lines)


class PersonaAgent:
    """Persona に従って議論ターンを生成するエージェント。"""

    def __init__(self, spec: PersonaSpec, llm: OpenAIClient | None = None) -> None:
        self.spec = spec
        self.llm = llm or OpenAIClient()
        self.log = get_logger(f"das.eval.persona.{spec.name}")
        self._template = _load_system_template()

    def _system_prompt(self, topic: str) -> str:
        extra_block = ""
        if self.spec.extra:
            extra_block = f"- 補足: {self.spec.extra}"
        return self._template.format(
            name=self.spec.name,
            stance_description=_STANCE_DESCRIPTION[self.spec.stance],
            focus=self.spec.focus,
            personality=self.spec.personality,
            extra_block=extra_block,
            topic=topic,
        )

    async def utter(
        self,
        history: list[Utterance],
        topic: str,
        *,
        info: str | None = None,
        turn_id: int | None = None,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> Utterance:
        """次の発言を生成して ``Utterance`` を返す。"""

        user_parts = [f"これまでの発言:\n{_format_history(history)}"]
        if info:
            user_parts.append(f"\n参考情報:\n{info}")
        user_parts.append("\nあなたの次の発言:")
        user_content = "\n".join(user_parts)

        messages = [
            {"role": "system", "content": self._system_prompt(topic)},
            {"role": "user", "content": user_content},
        ]
        text = await self.llm.chat(
            messages,  # type: ignore[arg-type]
            model=model,
            temperature=temperature,
        )
        text = text.strip()
        next_turn = turn_id if turn_id is not None else len(history) + 1

        self.log.info(
            "persona.utter",
            speaker=self.spec.name,
            turn_id=next_turn,
            n_chars=len(text),
        )
        return Utterance(turn_id=next_turn, speaker=self.spec.name, text=text)


__all__ = [
    "PersonaAgent",
    "PersonaSpec",
    "Stance",
    "build_persona",
]

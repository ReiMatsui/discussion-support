"""Stance polling エージェント (DEBATE benchmark 流: 公開立場 vs 私的立場)。

各ペルソナに対して以下を Likert (-3..+3) で測定する:

  - **Pre / Public**: 議論前にトピックに対する公開立場 (議論で表明する見込み)
  - **Pre / Private**: 議論前にトピックに対する私的立場 (内心どう思っているか)
  - **Post / Public**: 議論後にトピックに対する公開立場
  - **Post / Private**: 議論後にトピックに対する私的立場

ここで「公開」と「私的」のギャップ (public-private gap) は、参加者が
社会的圧力で表面的に同調しているだけの量を測る (DEBATE benchmark, 2025)。

LLM 同士の議論は表層で合意しがちなので、**私的立場が動いたか / 動いてないか**
を見ることで、提案手法が「見かけの合意」ではなく「実質的な意見変化」を
もたらしたかが測れる。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from das.agents.base import BaseAgent
from das.eval.persona import PersonaSpec
from das.types import Utterance

Phase = Literal["pre", "post"]


class StanceMeasurement(BaseModel):
    """1 ペルソナ × 1 フェーズ (pre/post) の立場測定。"""

    public_stance: int = Field(
        ge=-3, le=3,
        description="議論で表明する立場 (-3=強く反対, 0=中立, +3=強く賛成)",
    )
    private_stance: int = Field(
        ge=-3, le=3,
        description="内心で実際に思う立場 (-3=強く反対, 0=中立, +3=強く賛成)",
    )
    public_reason: str = Field(
        default="", description="公開立場の理由 (1-2 文)"
    )
    private_reason: str = Field(
        default="", description="私的立場の理由 (1-2 文)。公開と異なる場合はそのギャップの理由"
    )

    @property
    def public_private_gap(self) -> int:
        return abs(self.public_stance - self.private_stance)


class StanceAgent(BaseAgent):
    """ペルソナごとの Pre/Post × Public/Private 立場を LLM で測定する。"""

    name = "stance"

    async def measure(
        self,
        *,
        persona: PersonaSpec,
        topic: str,
        phase: Phase,
        transcript: list[Utterance] | None = None,
        model: str | None = None,
    ) -> StanceMeasurement:
        """指定フェーズで 1 ペルソナの立場を測定する。"""

        if phase == "pre":
            user = (
                f"# トピック\n{topic}\n\n"
                f"# あなたのペルソナ\n"
                f"- 名前: {persona.name}\n"
                f"- 立場: {persona.stance}\n"
                f"- 重視している論点: {persona.focus}\n"
                f"- 性格: {persona.personality}\n\n"
                "# 指示\n"
                "議論を **始める前** の段階で、このトピックに対して:\n"
                "- 議論の場で表明する **公開立場 (public_stance)** を -3 (強く反対) "
                "から +3 (強く賛成) で答えてください\n"
                "- 内心で実際に思う **私的立場 (private_stance)** を同じスケールで答えてください\n"
                "- それぞれ短い理由 (1〜2 文) を public_reason / private_reason に書いてください\n"
                "- 多くの場合 public と private は近い値になりますが、"
                "「人前では穏当に見せたい」「立場を強く表明したい / したくない」"
                "等の理由で **両者がずれる** こともあります。素直に答えてください"
            )
        else:  # post
            transcript_block = "\n".join(
                f"[t{u.turn_id}] {u.speaker}: {u.text}"
                for u in (transcript or [])
            )
            user = (
                f"# トピック\n{topic}\n\n"
                f"# あなたのペルソナ\n"
                f"- 名前: {persona.name}\n"
                f"- 立場 (議論前): {persona.stance}\n"
                f"- 重視している論点: {persona.focus}\n\n"
                f"# 議論内容 (全 transcript)\n{transcript_block}\n\n"
                "# 指示\n"
                "議論を **終えた直後** に、このトピックに対して:\n"
                "- 議論の場で表明している (or 表明した) **公開立場 (public_stance)** を "
                "-3 から +3 で答えてください\n"
                "- 内心で実際に思う **私的立場 (private_stance)** を同じスケールで答えてください\n"
                "- それぞれ短い理由を public_reason / private_reason に書いてください\n"
                "- 議論を通じて公開立場と私的立場が **ずれた** 場合 (例: "
                "「合意の流れに表面では同調したが、内心では納得していない」)、"
                "そのギャップを正直に表現してください"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "あなたは議論シミュレーションの参加者として、"
                    "公開立場と私的立場を別々に内省します。社会的圧力に流されて "
                    "表面的に合意したかどうかを見抜くため、両者を分けて測ります。"
                ),
            },
            {"role": "user", "content": user},
        ]

        result = await self.llm.chat_structured(
            messages,  # type: ignore[arg-type]
            response_format=StanceMeasurement,
            model=model,
        )
        self.log.info(
            "stance.measured",
            persona=persona.name,
            phase=phase,
            public=result.public_stance,
            private=result.private_stance,
            gap=result.public_private_gap,
        )
        return result


__all__ = ["Phase", "StanceAgent", "StanceMeasurement"]

"""LLM-judge による合意検出エージェント (Sirota et al. SIGDIAL 2025 路線)。

研究計画書 §5.1「合意形成までの時間」を、表面キーワードに依存しない LLM 判定で
求める。手順は **Finding Common Ground** (Sirota et al., 2025) の
**stance detection + agreement detection** を踏襲:

  1. 各参加者の直近発言から **立場 (position) + 極性 (polarity)** を抽出
  2. それらを横断して **全員が同じ結論に賛同しているか** を判定
  3. 構造化出力 (pydantic) で ``ConsensusJudgement`` を返す

設計上の選択:
  - LLM 呼び出しは高価なので、ファサード (``detect_consensus_with_llm``) 側で
    **構造的静止が起きたときだけ**呼び出す二段構成にする
  - 出力は配信チャネル非依存 (UI / ログ / 自動停止のいずれにも使える)
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from das.agents.base import BaseAgent
from das.eval.persona import PersonaSpec
from das.types import Utterance

Polarity = Literal["pro", "con", "partial_pro", "partial_con", "neutral"]


class StanceJudgement(BaseModel):
    """1 参加者の立場判定。"""

    speaker: str
    position: str = Field(description="その参加者が現時点で取っている立場の自然文サマリ")
    polarity: Polarity = Field(description="トピックに対する極性")
    confidence: float = Field(ge=0.0, le=1.0, description="判定の確信度 0-1")


class ConsensusJudgement(BaseModel):
    """合意成否の構造化判定。"""

    consensus_reached: bool
    consensus_position: str = Field(
        description="合意した内容の自然文サマリ (合意していなければ空文字)"
    )
    n_agreeing: int = Field(ge=0, description="その立場に賛同しているとみなせる参加者数")
    n_total: int = Field(ge=0, description="参加者数")
    confidence: float = Field(ge=0.0, le=1.0, description="合意判定の確信度 0-1")
    rationale: str = Field(description="判定の根拠 (なぜそう判断したか)")
    stances: list[StanceJudgement] = Field(default_factory=list)


class ConsensusAgent(BaseAgent):
    """LLM ベースの合意検出エージェント。"""

    name = "consensus"

    async def judge(
        self,
        *,
        topic: str,
        transcript: list[Utterance],
        personas: list[PersonaSpec],
        recent_window: int = 6,
        model: str | None = None,
    ) -> ConsensusJudgement:
        """直近 ``recent_window`` ターンの発話から合意判定を返す。"""

        recent = transcript[-recent_window:] if transcript else []
        recent_block = "\n".join(
            f"[t{u.turn_id}] {u.speaker}: {u.text}" for u in recent
        )
        persona_block = "\n".join(
            f"- {p.name} (stance={p.stance}, focus={p.focus})" for p in personas
        )

        system = (
            "あなたは議論の合意状況を分析する公平な判定者です。"
            "表面的な合意キーワード (確かに、なるほど、賛成 等) だけで判断せず、"
            "各参加者が実際にどの立場を取っているかを直近発言から読み解き、"
            "全員が同じ結論に賛同しているかを判定してください。"
            "「確かに〜が、しかし〜」のような譲歩 → 反論パターンは合意ではないことに注意。"
        )
        user = (
            f"# トピック\n{topic}\n\n"
            f"# 参加者\n{persona_block}\n\n"
            f"# 直近の発言 (最大 {recent_window} ターン)\n{recent_block}\n\n"
            "# 指示\n"
            "1. 各参加者の現時点の立場 (position) と極性 (polarity) を判定\n"
            "2. 全員が実質的に同じ結論を支持しているなら consensus_reached=true\n"
            "3. 合意した結論を consensus_position に自然文で書く (合意なしなら空文字)\n"
            "4. 部分賛成 (partial_pro / partial_con) は条件付き賛同とみなす — その条件が "
            "全員で一致しているなら合意とみなして良い\n"
            "5. 判定の根拠を rationale に短く書く"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        result = await self.llm.chat_structured(
            messages,  # type: ignore[arg-type]
            response_format=ConsensusJudgement,
            model=model,
        )
        self.log.info(
            "consensus_agent.judged",
            consensus=result.consensus_reached,
            n_agreeing=result.n_agreeing,
            n_total=result.n_total,
            confidence=round(result.confidence, 3),
        )
        return result


def judgement_to_dict(j: ConsensusJudgement) -> dict:
    """ログ書き出し用の辞書化 (run_meta.json 等で使う)。"""

    return json.loads(j.model_dump_json())


__all__ = [
    "ConsensusAgent",
    "ConsensusJudgement",
    "Polarity",
    "StanceJudgement",
    "judgement_to_dict",
]

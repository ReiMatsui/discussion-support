"""シミュレーション評価サブシステム。

LLM 多人数エージェントによる議論シミュレーションと、3 条件比較による
評価フレームワーク (M2 で構築)。
"""

from das.eval.conditions import (
    Condition,
    ConditionFlatRAG,
    ConditionFullProposal,
    ConditionNone,
)
from das.eval.controller import InfoProvider, SessionConfig, SessionRunner
from das.eval.persona import PersonaAgent, PersonaSpec, Stance, build_persona
from das.eval.presets import cafeteria_personas, policy_ai_lecture_personas

__all__ = [
    "Condition",
    "ConditionFlatRAG",
    "ConditionFullProposal",
    "ConditionNone",
    "InfoProvider",
    "PersonaAgent",
    "PersonaSpec",
    "SessionConfig",
    "SessionRunner",
    "Stance",
    "build_persona",
    "cafeteria_personas",
    "policy_ai_lecture_personas",
]

"""シミュレーション評価サブシステム。

LLM 多人数エージェントによる議論シミュレーションと、3 条件比較による
評価フレームワーク (M2 で構築)。
"""

from das.eval.conditions import (
    Condition,
    ConditionFlatRAG,
    ConditionFullProposal,
    ConditionNone,
    FlatRAGItem,
    InfoItem,
    InterventionLogEntry,
    write_intervention_log,
)
from das.eval.controller import InfoProvider, SessionConfig, SessionRunner
from das.eval.metrics import (
    GraphMetrics,
    TranscriptMetrics,
    gini_coefficient,
    graph_metrics,
    transcript_metrics,
)
from das.eval.persona import PersonaAgent, PersonaSpec, Stance, build_persona
from das.eval.presets import cafeteria_personas, policy_ai_lecture_personas

__all__ = [
    "Condition",
    "ConditionFlatRAG",
    "ConditionFullProposal",
    "ConditionNone",
    "FlatRAGItem",
    "GraphMetrics",
    "InfoItem",
    "InfoProvider",
    "InterventionLogEntry",
    "PersonaAgent",
    "PersonaSpec",
    "SessionConfig",
    "SessionRunner",
    "Stance",
    "TranscriptMetrics",
    "build_persona",
    "cafeteria_personas",
    "gini_coefficient",
    "graph_metrics",
    "policy_ai_lecture_personas",
    "transcript_metrics",
    "write_intervention_log",
]

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
from das.eval.consensus import ConsensusReport, ConsensusSignal, detect_consensus
from das.eval.controller import InfoProvider, SessionConfig, SessionRunner, StopCondition
from das.eval.judge import (
    AggregatedScores,
    JudgeAgent,
    JudgeReport,
    JudgeScores,
    aggregate_reports,
)
from das.eval.metrics import (
    GraphMetrics,
    TranscriptMetrics,
    gini_coefficient,
    graph_metrics,
    transcript_metrics,
)
from das.eval.persona import PersonaAgent, PersonaSpec, Stance, build_persona
from das.eval.presets import cafeteria_personas, policy_ai_lecture_personas
from das.eval.run_eval import (
    ConditionFactory,
    EvalResult,
    SingleRunResult,
    run_eval,
)

__all__ = [
    "AggregatedScores",
    "Condition",
    "ConditionFactory",
    "ConditionFlatRAG",
    "ConditionFullProposal",
    "ConditionNone",
    "ConsensusReport",
    "ConsensusSignal",
    "EvalResult",
    "FlatRAGItem",
    "GraphMetrics",
    "InfoItem",
    "InfoProvider",
    "InterventionLogEntry",
    "JudgeAgent",
    "JudgeReport",
    "JudgeScores",
    "PersonaAgent",
    "PersonaSpec",
    "SessionConfig",
    "SessionRunner",
    "SingleRunResult",
    "Stance",
    "StopCondition",
    "TranscriptMetrics",
    "aggregate_reports",
    "build_persona",
    "cafeteria_personas",
    "detect_consensus",
    "gini_coefficient",
    "graph_metrics",
    "policy_ai_lecture_personas",
    "run_eval",
    "transcript_metrics",
    "write_intervention_log",
]

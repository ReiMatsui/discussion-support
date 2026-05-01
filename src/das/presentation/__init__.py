"""4 層情報提示 (L1〜L4)。

- L1 参加者向け 個別化通知: ``das.eval.conditions.ConditionFullProposal.last_items``
  (FacilitationAgent が選定した InfoItem を提示)
- L2 ファシリテーター向け フルビュー: ``das.ui.streamlit_app`` (グラフ + 統計)
- L3 議論区切りでの自然言語要約: ``presentation.summary``
- L4 議論後の振り返り個別化: ``presentation.retrospective``
"""

from das.presentation.retrospective import (
    IncomingAttack,
    OutgoingAttack,
    ParticipantRetrospective,
    retrospective_for,
    retrospectives_by_speaker,
)
from das.presentation.summary import (
    SessionSummary,
    llm_summary,
    programmatic_summary,
    summarize_session,
)

__all__ = [
    "IncomingAttack",
    "OutgoingAttack",
    "ParticipantRetrospective",
    "SessionSummary",
    "llm_summary",
    "programmatic_summary",
    "retrospective_for",
    "retrospectives_by_speaker",
    "summarize_session",
]

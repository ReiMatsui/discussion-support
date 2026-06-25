"""音声ファシリテーターの介入ライフサイクルイベント."""
from __future__ import annotations

import datetime
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class FacilitationEventType(StrEnum):
    """Phase V0で観測する介入イベント種別."""

    TRIGGER_REQUESTED = "trigger_requested"
    TRIGGER_SUPPRESSED = "trigger_suppressed"
    RESPONSE_REQUESTED = "response_requested"
    SPEECH_STARTED = "speech_started"
    SPEECH_COMPLETED = "speech_completed"
    UTTERANCE_COMPLETED = "utterance_completed"
    RESPONSE_COMPLETED = "response_completed"
    INTERRUPTED = "interrupted"
    NOOP = "noop"
    ERROR = "error"


def new_intervention_id() -> str:
    """介入試行を一意に識別するIDを生成する."""
    return f"int_{uuid.uuid4().hex}"


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="milliseconds")


@dataclass(frozen=True, slots=True)
class FacilitationEvent:
    """1回の介入に属する追記専用イベント.

    Phase V1以降の診断情報は details に追加し、V0のイベント形式を壊さない。
    """

    intervention_id: str
    event_type: FacilitationEventType
    trigger_kind: str | None = None
    trigger_reason: str | None = None
    input_utterances: tuple[dict[str, str], ...] = ()
    model_output: str | None = None
    playback: bool | None = None
    details: dict[str, Any] = field(default_factory=dict)
    occurred_at: str = field(default_factory=_utc_now_iso)
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex}")
    schema_version: int = 1

    @classmethod
    def create(
        cls,
        event_type: FacilitationEventType,
        intervention_id: str,
        **kwargs: Any,
    ) -> FacilitationEvent:
        """列挙型を使ったイベント生成の短縮形."""
        return cls(
            intervention_id=intervention_id,
            event_type=event_type,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSONへ保存可能な辞書へ変換する."""
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["input_utterances"] = list(self.input_utterances)
        return data

"""All provisional thresholds; TOML overrides can be supplied to the CLI."""

import tomllib
from pathlib import Path

from pydantic import Field

from .models import Record


class Config(Record):
    focus_threshold: float = Field(default=0.7, ge=0, le=1)
    focus_turns: int = Field(default=2, ge=2)
    focus_ms: int = Field(default=10000, ge=0)
    stay_prior: float = Field(default=0.8, gt=0, lt=1)
    shift_stay_prior: float = Field(default=0.45, gt=0, lt=1)
    observation_chars: int = Field(default=20, gt=0)
    pending_ms: int = Field(default=60000, gt=0)
    name_threshold: float = Field(default=0.5, ge=0, le=1)
    stance_threshold: float = Field(default=0.8, ge=0, le=1)
    resolution_threshold: float = Field(default=0.8, ge=0, le=1)
    identity_margin: float = Field(default=0.05, ge=0, le=1)
    decision_ms: int = Field(default=10000, ge=0)
    display_ms: int = Field(default=3000, gt=0)
    display_levels: int = Field(default=3, ge=1)
    display_nodes: int = Field(default=15, ge=1)
    issue_chars: int = Field(default=20, ge=1)
    position_chars: int = Field(default=16, ge=1)
    recent_turns: int = Field(default=6, ge=1)
    other_issues: int = Field(default=4, ge=0)
    openai_model: str = "gpt-4.1-mini"
    label_model: str = "gpt-4.1-mini"
    jev_model: str = "jev-1.13.0"
    budget_usd: float = Field(default=2.0, gt=0, le=2)
    input_usd_per_million: float = Field(default=0.4, gt=0)
    output_usd_per_million: float = Field(default=1.6, gt=0)

    @classmethod
    def load(cls, path: Path | None = None):
        return cls.model_validate(tomllib.loads(path.read_text()) if path else {})

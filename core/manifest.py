from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ExperimentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    mode: str = "train"
    seed: int = 0


class ExperimentManifest(BaseModel):
    model_config = ConfigDict(extra="allow")
    experiment: ExperimentSettings
    includes: list[str] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)
    benchmarks: str | None = None
    gates: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)


def load_manifest(path: str | Path) -> ExperimentManifest:
    """Load and validate a top-level manifest; include merging follows in Phase 0."""
    with Path(path).open(encoding="utf-8") as source:
        data = yaml.safe_load(source) or {}
    return ExperimentManifest.model_validate(data)

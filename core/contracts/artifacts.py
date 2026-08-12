from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ArtifactBundle:
    """Versioned record of a run's inputs and outputs."""

    root: Path
    manifest_snapshot: Path
    files: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)

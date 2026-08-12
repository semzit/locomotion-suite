from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TransitionBatch:
    """Simulator-to-algorithm wire format."""

    observations: Any
    actions: Any
    rewards: Any
    next_observations: Any
    terminated: Any
    truncated: Any
    info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

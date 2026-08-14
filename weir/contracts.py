from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from torch import nn


class SimBackend(Protocol):
    """Simulator boundary: reset/step a physics model and describe its interface."""

    def load(self, robot_spec: dict[str, Any], sim_config: dict[str, Any]) -> None: ...
    def reset(self, batch_size: int) -> Any: ...
    def step(self, actions: Any) -> Any: ...
    def observation_spec(self) -> dict[str, Any]: ...
    def action_spec(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


class AlgorithmPlugin(Protocol):
    """Learning and policy boundary."""

    def configure(
        self,
        observation_spec: dict[str, Any],
        action_spec: dict[str, Any],
        config: dict[str, Any],
    ) -> None: ...
    def act(self, observations: Any, deterministic: bool = False) -> Any: ...
    def update(self, batch: TransitionBatch) -> dict[str, float]: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
    def export_policy(self) -> nn.Module: ...


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

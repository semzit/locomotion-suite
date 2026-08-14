from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from torch import nn

Observation = np.ndarray
Action = np.ndarray


@dataclass(frozen=True, slots=True)
class Shape:
    """Dimensions, dtype, and optional per-dimension bounds of an observation or action space."""

    dims: tuple[int, ...]
    dtype: str
    low: np.ndarray | None = None
    high: np.ndarray | None = None


@dataclass(slots=True)
class SimStep:
    """One simulator transition: next observation plus reward and termination flags."""

    observation: Observation
    reward: float
    terminated: bool
    truncated: bool


@runtime_checkable
class SimBackend(Protocol):
    """Simulator boundary: reset/step a physics model and describe its interface."""

    def load(self, agent_config: dict[str, Any], sim_config: dict[str, Any]) -> None: ...
    def reset(self, seed: int | None = None) -> Observation: ...
    def step(self, actions: Action) -> SimStep: ...
    def observation_shape(self) -> Shape: ...
    def action_shape(self) -> Shape: ...
    def close(self) -> None: ...


@runtime_checkable
class AlgorithmPlugin(Protocol):
    """Learning and policy boundary."""

    def configure(
        self,
        observation_shape: Shape,
        action_shape: Shape,
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

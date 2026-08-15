from __future__ import annotations

from typing import Any, SupportsFloat

import numpy as np
from gymnasium import Env, spaces

from weir.core.contracts import Action, Observation, Shape, SimBackend


class GymEnv(Env):
    """Adapt a SimBackend to the gymnasium environment interface."""

    metadata = {"render_modes": []}

    def __init__(self, sim: SimBackend) -> None:
        super().__init__()
        self._sim = sim
        self.observation_space = shape_to_box(sim.observation_shape())
        self.action_space = shape_to_box(sim.action_shape())

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        super().reset(seed=seed)
        observation = self._sim.reset(seed=seed)
        return observation, {}

    def step(self, action: Action) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        step = self._sim.step(action)
        return step.observation, step.reward, step.terminated, step.truncated, {}

    def close(self) -> None:
        self._sim.close()

    @property
    def sim(self) -> SimBackend:
        return self._sim


def shape_to_box(shape: Shape) -> spaces.Box:
    """Build a gymnasium Box space from a Shape (unbounded when low/high are absent)."""
    low = shape.low if shape.low is not None else -np.inf
    high = shape.high if shape.high is not None else np.inf
    return spaces.Box(low=low, high=high, shape=shape.dims, dtype=np.float32)

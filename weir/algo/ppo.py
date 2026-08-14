from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from torch import nn

from weir.algo.utils import sample_action
from weir.contracts import Shape, TransitionBatch


class PPOAlgorithm:
    """Reference PPO algorithm implementation used for the minimal stack."""

    def configure(
        self,
        observation_shape: Shape,
        action_shape: Shape,
        config: dict[str, Any],
    ) -> None:
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.config = config

    def act(self, observations: Any, deterministic: bool = False) -> Any:
        return sample_action(
            self.action_shape, np.random.default_rng(), deterministic=deterministic
        )

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        return {"loss": 0.0, "reward": 0.0}

    def save(self, path: Path) -> None:
        path.write_text("ppo-checkpoint", encoding="utf-8")

    def load(self, path: Path) -> None:
        _ = path

    def export_policy(self) -> nn.Module:
        raise NotImplementedError("export_policy requires torch; not implemented in the stub.")

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

from weir.contracts import TransitionBatch


class PPOAlgorithm:
    """Reference PPO algorithm implementation used for the minimal stack."""

    def configure(
        self,
        observation_shape: dict[str, Any],
        action_shape: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.config = config

    def act(self, observations: Any, deterministic: bool = False) -> Any:
        _ = deterministic
        return observations

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        return {"loss": 0.0, "reward": 0.0 if not batch else 0.0}

    def save(self, path: Path) -> None:
        path.write_text("ppo-checkpoint", encoding="utf-8")

    def load(self, path: Path) -> None:
        _ = path

    def export_policy(self) -> nn.Module:
        raise NotImplementedError("export_policy requires torch; not implemented in the stub.")

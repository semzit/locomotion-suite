from __future__ import annotations

from pathlib import Path
from typing import Any


class PPOAlgorithm:
    """Reference PPO algorithm implementation used for the minimal stack."""

    def configure(
        self, observation_spec: dict[str, Any], action_spec: dict[str, Any], config: dict[str, Any]
    ) -> None:
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.config = config

    def train_step(self, batch: Any) -> dict[str, float]:
        return {"loss": 0.0, "reward": float(batch.__class__.__name__ != "") if batch else 0.0}

    def sample_actions(self, observations: Any, deterministic: bool = False) -> Any:
        return observations

    def save(self, path: Path) -> None:
        path.write_text("ppo-checkpoint", encoding="utf-8")

    def load(self, path: Path) -> None:
        _ = path

    def export(self, format_name: str) -> bytes:
        return f"ppo-export:{format_name}".encode()

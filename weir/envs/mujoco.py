from __future__ import annotations

from typing import Any


class MuJoCoSim:
    """Reference MuJoCo simulator implementation used for the minimal stack."""

    def load(self, robot_spec: dict[str, Any], sim_config: dict[str, Any]) -> None:
        self.robot_spec = robot_spec
        self.sim_config = sim_config

    def reset(self, batch_size: int) -> dict[str, Any]:
        return {"batch_size": batch_size, "observations": []}

    def step(self, actions: Any) -> dict[str, Any]:
        return {"actions": actions, "reward": 0.0, "done": False}

    def observation_shape(self) -> dict[str, Any]:
        return {"shape": [0], "dtype": "float32"}

    def action_shape(self) -> dict[str, Any]:
        return {"shape": [0], "dtype": "float32"}

    def close(self) -> None:
        return None

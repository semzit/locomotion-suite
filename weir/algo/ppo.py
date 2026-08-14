from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from weir.contracts import Shape
from weir.envs.gym_env import shape_to_box


class _SpacesOnly(Env):
    """Minimal env exposing only the spaces SB3 needs to build its policy."""

    def __init__(self, observation_shape: Shape, action_shape: Shape) -> None:
        self.observation_space = shape_to_box(observation_shape)
        self.action_space = shape_to_box(action_shape)


class PPOAlgorithm:
    """PPO via stable-baselines3, exposed behind the AlgorithmPlugin protocol."""

    def configure(
        self,
        observation_shape: Shape,
        action_shape: Shape,
        config: dict[str, Any],
    ) -> None:
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        net_arch = list(config.get("net_arch", [64, 64]))
        self._model = PPO(
            "MlpPolicy",
            _SpacesOnly(observation_shape, action_shape),
            policy_kwargs={"net_arch": net_arch},
            learning_rate=float(config.get("learning_rate", 3e-4)),
            n_steps=int(config.get("n_steps", 2048)),
            batch_size=int(config.get("batch_size", 64)),
            n_epochs=int(config.get("n_epochs", 10)),
            gamma=float(config.get("gamma", 0.99)),
            gae_lambda=float(config.get("gae_lambda", 0.95)),
            clip_range=float(config.get("clip_range", 0.2)),
            ent_coef=float(config.get("ent_coef", 0.0)),
            vf_coef=float(config.get("vf_coef", 0.5)),
            max_grad_norm=float(config.get("max_grad_norm", 0.5)),
        )

    def learn(self, env: Env, total_steps: int) -> dict[str, float]:
        self._require_model()
        self._model.set_env(env)
        self._model.learn(total_timesteps=total_steps, progress_bar=False)
        return {"total_steps": float(total_steps)}

    def act(self, observations: Any, deterministic: bool = False) -> Any:
        self._require_model()
        action, _ = self._model.predict(
            np.asarray(observations, dtype=np.float32), deterministic=deterministic
        )
        return action

    def save(self, path: Path) -> None:
        self._require_model()
        self._model.save(str(path))

    def load(self, path: Path) -> None:
        self._model = PPO.load(str(path))

    def export_policy(self) -> nn.Module:
        self._require_model()
        return _DeterministicPolicy(self._model.policy)

    def _require_model(self) -> None:
        if getattr(self, "_model", None) is None:
            raise RuntimeError("PPOAlgorithm.configure() must be called before use")


class _DeterministicPolicy(nn.Module):
    """Inference-only wrapper: forward maps observations to deterministic mean actions."""

    def __init__(self, policy: ActorCriticPolicy) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        actions, _, _ = self.policy(observations, deterministic=True)
        return actions

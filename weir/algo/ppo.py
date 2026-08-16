from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import Env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from torch import nn

from weir.algo.utils import DeterministicPolicy, SpacesOnly
from weir.core.contracts import AlgorithmPlugin, Shape


class PPOAlgorithm(AlgorithmPlugin):
    """PPO via stable-baselines3, exposed behind the AlgorithmPlugin protocol."""

    def configure(
        self,
        observation_shape: Shape,
        action_shape: Shape,
        config: dict[str, Any],
    ) -> None:
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self._checkpoint_freq: int | None = None
        checkpoint = config.get("checkpoint")
        if checkpoint:
            self._model = PPO.load(str(checkpoint))
            return
        self._checkpoint_freq = config.get("checkpoint_freq")
        net_arch = list(config.get("net_arch", [64, 64]))
        self._model = PPO(
            "MlpPolicy",
            SpacesOnly(observation_shape, action_shape),
            policy_kwargs={"net_arch": net_arch},
            device=str(config.get("device", "auto")),
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

    def learn(
        self,
        env: Env,
        total_steps: int,
        callback: Any | None = None,
    ) -> dict[str, float]:
        self._require_model()
        self._model.set_env(env)
        callbacks: list[Any] = []
        if self._checkpoint_freq:
            callbacks.append(
                CheckpointCallback(
                    save_freq=int(self._checkpoint_freq),
                    save_path=str(Path.cwd()),
                    name_prefix="checkpoint",
                )
            )
        if callback is not None:
            callbacks.append(callback)
        self._model.learn(
            total_timesteps=total_steps,
            progress_bar=False,
            callback=callbacks or None,
        )
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
        return DeterministicPolicy(self._model.policy)

    def _require_model(self) -> None:
        if getattr(self, "_model", None) is None:
            raise RuntimeError("PPOAlgorithm.configure() must be called before use")

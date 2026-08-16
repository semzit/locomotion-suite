from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weir.core.contracts import Action, Observation


@dataclass(slots=True)
class BalanceTask:
    """CartPole balance: reward every step, terminate when the pole falls.

    Ported from Gymnasium's CartPole-v1:

    | Gymnasium (Farama Foundation), "CartPole-v1".
    | https://github.com/Farama-Foundation/Gymnasium — file
    | ``gymnasium/envs/classic_control/cartpole.py``. MIT License.

    Semantics: ``+1`` reward per step while ``|cart x| <= x_threshold`` and
    ``|pole angle| <= theta_threshold`` (12 degrees), terminating otherwise.
    The observation layout is ``[x, theta, x_dot, theta_dot]``.
    """

    x_threshold: float = 2.4
    theta_threshold: float = 12.0 * 2.0 * np.pi / 360.0  # 12 degrees in radians

    def reward(
        self,
        observation: Observation,
        action: Action,
        prev_action: Action | None = None,
    ) -> float:
        return 1.0

    def terminated(self, observation: Observation) -> bool:
        x, theta = float(observation[0]), float(observation[1])
        return bool(abs(x) > self.x_threshold or abs(theta) > self.theta_threshold)

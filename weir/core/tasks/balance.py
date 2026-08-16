from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weir.core.contracts import Action, Observation


@dataclass(slots=True)
class BalanceTask:
    """CartPole balance: reward every step, terminate when the pole falls.

    The reward and termination semantics are exactly the reward-relevant lines
    of Gymnasium's ``CartPoleEnv.step`` (with the default
    ``sutton_barto_reward=False``), plus the two thresholds set in its
    ``__init__``:

    | Gymnasium (Farama Foundation), "CartPole-v1" — MIT License.
    | https://github.com/Farama-Foundation/Gymnasium, file
    | ``gymnasium/envs/classic_control/cartpole.py``, which itself credits
    | "Classic cart-pole system implemented by Rich Sutton et al.",
    | http://incompleteideas.net/sutton/book/code/pole.c
    | (permalink: https://perma.cc/C9ZM-652R).

    Source lines::

        self.theta_threshold_radians = 12 * 2 * math.pi / 360   # __init__
        self.x_threshold = 2.4                                  # __init__
        terminated = bool(                                      # step()
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        if not terminated:
            reward = 0.0 if self._sutton_barto_reward else 1.0  # step()
        elif self.steps_beyond_terminated is None:
            reward = -1.0 if self._sutton_barto_reward else 1.0  # step()
        else:
            reward = -1.0 if self._sutton_barto_reward else 0.0  # step()

    i.e. ``+1`` on every step (including the termination step), terminating
    when ``|x| > 2.4`` or ``|theta| > 12 degrees``. One adaptation: our
    observation layout is MuJoCo's ``qpos||qvel`` order, ``[x, theta,
    x_dot, theta_dot]`` — Gymnasium's state is ``[x, x_dot, theta,
    theta_dot]`` — so ``terminated`` reads ``obs[0]`` and ``obs[1]``.
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

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from weir.core.contracts import Action, Observation
from weir.core.utils import rotate_vector


class Task(Protocol):
    """Environment task: maps simulator state to reward and termination.

    ``prev_action`` is the action from the previous step (None on the first
    step of an episode), enabling action-rate penalties.
    """

    def reward(
        self,
        observation: Observation,
        action: Action,
        prev_action: Action | None = None,
    ) -> float: ...
    def terminated(self, observation: Observation) -> bool: ...


@dataclass(slots=True)
class SurviveTask:
    """Reward survival for every step; never terminates on its own."""

    live_reward: float = 1.0

    def reward(
        self,
        observation: Observation,
        action: Action,
        prev_action: Action | None = None,
    ) -> float:
        return self.live_reward

    def terminated(self, observation: Observation) -> bool:
        return False


@dataclass(slots=True)
class StandingTask:
    """Stay upright and on the floor: survive while standing, terminate on falling.

    Assumes the observation begins with the root freejoint: qpos[2] is the
    body height above the floor.
    """

    min_height: float = 0.8
    live_reward: float = 1.0

    def reward(
        self,
        observation: Observation,
        action: Action,
        prev_action: Action | None = None,
    ) -> float:
        return 0.0 if self.terminated(observation) else self.live_reward

    def terminated(self, observation: Observation) -> bool:
        return float(observation[2]) < self.min_height


@dataclass(slots=True)
class BalanceTask:
    """CartPole balance: reward every step, terminate when the pole falls.

    Ported from Gymnasium's CartPole-v1 (MIT license) semantics: reward is
    ``+1`` per step while ``|cart x| <= x_threshold`` and
    ``|pole angle| <= theta_threshold``, terminating otherwise. Observation
    layout: ``[x, theta, x_dot, theta_dot]``.
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


TaskFactory = Callable[..., Task]


@dataclass(slots=True)
class WalkForwardTask:
    """Walk forward along world +x while staying upright; terminate on falling.

    The reward blend ports the standard legged-locomotion terms from Isaac
    Lab's ``mdp.rewards`` catalog (MIT license), adapted to this project's
    stateless observation layout:

    - ``upright_posture``: cosine of the base tilt (base up vector dot world up)
    - ``heading_consistency``: base forward vector dot goal heading (world +x)
    - ``root_lin_vel`` forward term: forward speed in the heading direction
    - ``alive_reward``: per-step survival bonus
    - ``action_rate_l2``: penalty on the squared change in action (needs the
      previous step's action)

    Assumes the observation begins with the root freejoint: obs[2] is the
    body height above the floor, obs[3:7] is the root quaternion in
    (w, x, y, z) order, and obs[nq] is the root linear x-velocity (qvel[0]).
    """

    nq: int = 19
    min_height: float = 0.9
    forward_coef: float = 2.0
    heading_coef: float = 1.0
    upright_coef: float = 0.5
    alive_reward: float = 1.0
    action_rate_coef: float = 0.02

    def reward(
        self,
        observation: Observation,
        action: Action,
        prev_action: Action | None = None,
    ) -> float:
        if self.terminated(observation):
            return 0.0
        forward_vel = float(observation[self.nq])
        heading = float(rotate_vector(observation[3:7], (1.0, 0.0, 0.0))[0])
        uprightness = float(rotate_vector(observation[3:7], (0.0, 0.0, 1.0))[2])
        action_rate = 0.0
        if prev_action is not None:
            delta = np.asarray(action, dtype=np.float32) - np.asarray(prev_action, dtype=np.float32)
            action_rate = self.action_rate_coef * float(np.sum(np.square(delta)))
        return (
            self.alive_reward
            + self.forward_coef * forward_vel * heading
            + self.heading_coef * heading
            + self.upright_coef * uprightness
            - action_rate
        )

    def terminated(self, observation: Observation) -> bool:
        return float(observation[2]) < self.min_height


TASKS: dict[str, TaskFactory] = {
    "survive": SurviveTask,
    "standing": StandingTask,
    "balance": BalanceTask,
    "walk_forward": WalkForwardTask,
}

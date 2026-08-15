from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from weir.core.contracts import Action, Observation
from weir.core.utils import rotate_vector


class Task(Protocol):
    """Environment task: maps simulator state to reward and termination."""

    def reward(self, observation: Observation, action: Action) -> float: ...
    def terminated(self, observation: Observation) -> bool: ...


@dataclass(slots=True)
class SurviveTask:
    """Reward survival for every step; never terminates on its own."""

    live_reward: float = 1.0

    def reward(self, observation: Observation, action: Action) -> float:
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

    def reward(self, observation: Observation, action: Action) -> float:
        return 0.0 if self.terminated(observation) else self.live_reward

    def terminated(self, observation: Observation) -> bool:
        return float(observation[2]) < self.min_height


TaskFactory = Callable[..., Task]

_ACTION_NORM_SCALE = 2.0


@dataclass(slots=True)
class WalkForwardTask:
    """Walk forward along world +x while staying upright; terminate on falling.

    Assumes the observation begins with the root freejoint: obs[2] is the
    body height above the floor, obs[3:7] is the root quaternion in
    (w, x, y, z) order, and obs[nq] is the root linear x-velocity (qvel[0]).
    """

    nq: int = 19
    min_height: float = 0.9
    forward_coef: float = 1.0
    heading_coef: float = 0.3
    upright_coef: float = 0.5
    action_penalty_coef: float = 0.01
    alive_reward: float = 0.5

    def reward(self, observation: Observation, action: Action) -> float:
        if self.terminated(observation):
            return 0.0
        forward_vel = float(observation[self.nq])
        heading = float(rotate_vector(observation[3:7], (1.0, 0.0, 0.0))[0])
        uprightness = float(rotate_vector(observation[3:7], (0.0, 0.0, 1.0))[2])
        action_penalty = self.action_penalty_coef * float(
            np.sum(np.square(np.asarray(action, dtype=np.float32) / _ACTION_NORM_SCALE))
        )
        return (
            self.alive_reward
            + self.forward_coef * forward_vel * heading
            + self.heading_coef * heading
            + self.upright_coef * uprightness
            - action_penalty
        )

    def terminated(self, observation: Observation) -> bool:
        return float(observation[2]) < self.min_height


TASKS: dict[str, TaskFactory] = {
    "survive": SurviveTask,
    "standing": StandingTask,
    "walk_forward": WalkForwardTask,
}

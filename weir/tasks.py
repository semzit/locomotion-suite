from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from weir.contracts import Action, Observation


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

TASKS: dict[str, TaskFactory] = {
    "survive": SurviveTask,
    "standing": StandingTask,
}

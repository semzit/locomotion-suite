from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weir.core.contracts import Action, Observation
from weir.core.utils import rotate_vector


@dataclass(slots=True)
class WalkForwardTask:
    """Walk forward along world +x while staying upright; terminate on falling.

    The reward blend ports the standard legged-locomotion reward terms from
    Isaac Lab's reward catalog:

    | Isaac Lab (NVIDIA), "Isaac Lab".
    | https://github.com/isaac-sim/IsaacLab — file
    | ``source/isaaclab/isaaclab/envs/mdp/rewards/basic_rewards.py``.
    | MIT License.

    Terms ported (each adapted to this project's stateless observation
    layout — no asset/phase tensors, just the root freejoint slice):

    - ``upright_posture``: cosine of the base tilt (base up vector dot world up)
    - ``heading_consistency``: base forward vector dot goal heading (world +x)
    - ``root_lin_vel`` forward term: forward speed in the heading direction
    - ``alive_reward``: per-step survival bonus
    - ``action_rate_l2``: penalty on the squared change in action (uses the
      previous step's action via ``prev_action``)

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

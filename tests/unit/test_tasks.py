import numpy as np
import pytest

from weir.core.tasks import (
    TASKS,
    BalanceTask,
    StandingTask,
    SurviveTask,
    WalkForwardTask,
)
from weir.core.utils import rotate_vector


def test_survive_never_terminates() -> None:
    task = SurviveTask(live_reward=1.0)
    obs = np.zeros(10, dtype=np.float32)
    assert task.reward(obs, np.zeros(1, dtype=np.float32)) == 1.0
    assert task.terminated(obs) is False


def test_standing_terminates_below_min_height() -> None:
    task = StandingTask(min_height=0.8, live_reward=2.0)
    tall = np.zeros(10, dtype=np.float32)
    tall[2] = 1.2
    short = np.zeros(10, dtype=np.float32)
    short[2] = 0.5

    assert task.terminated(tall) is False
    assert task.terminated(short) is True
    assert task.reward(tall, np.zeros(1, dtype=np.float32)) == 2.0
    assert task.reward(short, np.zeros(1, dtype=np.float32)) == 0.0


def test_tasks_registry_builds_with_params() -> None:
    assert TASKS["survive"] is SurviveTask
    assert TASKS["standing"] is StandingTask
    task = TASKS["standing"](min_height=0.5)
    assert task.min_height == 0.5


def test_balance_rewards_every_step_and_terminates_outside_bounds() -> None:
    task = BalanceTask()
    obs = np.zeros(4, dtype=np.float32)
    action = np.zeros(1, dtype=np.float32)
    assert task.reward(obs, action) == 1.0
    assert task.terminated(obs) is False

    just_over = np.zeros(4, dtype=np.float32)
    just_over[1] = task.theta_threshold + 0.01
    assert task.terminated(just_over) is True

    just_under = np.zeros(4, dtype=np.float32)
    just_under[1] = task.theta_threshold - 0.01
    assert task.terminated(just_under) is False

    cart_over = np.zeros(4, dtype=np.float32)
    cart_over[0] = task.x_threshold + 0.1
    assert task.terminated(cart_over) is True


def test_balance_registry_entry() -> None:
    assert TASKS["balance"] is BalanceTask


def _make_obs(
    *,
    nq: int = 19,
    nv: int = 18,
    height: float,
    yaw: float,
    forward_vel: float,
) -> np.ndarray:
    """Build a root-freejoint observation: qpos then qvel, quat yawed about z."""
    obs = np.zeros(nq + nv, dtype=np.float32)
    obs[2] = height
    half = yaw / 2.0
    obs[3:7] = (np.cos(half), 0.0, 0.0, np.sin(half))
    obs[nq] = forward_vel
    return obs


def test_walk_forward_terminates_below_min_height() -> None:
    task = WalkForwardTask(min_height=0.9, nq=19)
    tall = _make_obs(height=1.2, yaw=0.0, forward_vel=0.0)
    short = _make_obs(height=0.5, yaw=0.0, forward_vel=0.0)

    assert task.terminated(tall) is False
    assert task.terminated(short) is True
    assert task.reward(short, np.zeros(1, dtype=np.float32)) == 0.0


def test_walk_forward_reward_scales_with_forward_velocity() -> None:
    task = WalkForwardTask(nq=19)
    action = np.zeros(1, dtype=np.float32)
    slow = _make_obs(height=1.2, yaw=0.0, forward_vel=1.0)
    fast = _make_obs(height=1.2, yaw=0.0, forward_vel=2.0)

    assert task.reward(fast, action) > task.reward(slow, action)
    # alive 1.0 + forward 2.0*1*1 + heading 1.0*1 + upright 0.5*1
    assert pytest.approx(task.reward(slow, action)) == 4.5


def test_walk_forward_sideways_motion_reduces_reward() -> None:
    task = WalkForwardTask(nq=19)
    action = np.zeros(1, dtype=np.float32)
    aligned = _make_obs(height=1.2, yaw=0.0, forward_vel=1.0)
    sideways = _make_obs(height=1.2, yaw=np.pi / 2.0, forward_vel=1.0)

    assert task.reward(sideways, action) < task.reward(aligned, action)
    # sideways: heading ~0, still upright -> alive 1.0 + upright 0.5
    assert pytest.approx(task.reward(sideways, action)) == 1.5


def test_walk_forward_backwards_motion_reduces_reward() -> None:
    task = WalkForwardTask(nq=19)
    action = np.zeros(1, dtype=np.float32)
    aligned = _make_obs(height=1.2, yaw=0.0, forward_vel=1.0)
    backwards = _make_obs(height=1.2, yaw=np.pi, forward_vel=1.0)

    assert task.reward(backwards, action) < task.reward(aligned, action)
    assert task.reward(backwards, action) < 0.0


def test_walk_forward_action_rate_penalty_reduces_reward() -> None:
    task = WalkForwardTask(nq=19)
    obs = _make_obs(height=1.2, yaw=0.0, forward_vel=1.0)
    zero = np.zeros(1, dtype=np.float32)
    busy = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    first_step = task.reward(obs, busy)  # prev_action is None -> no rate penalty
    assert task.reward(obs, busy, prev_action=busy) == first_step
    with_prev = task.reward(obs, busy, prev_action=zero)
    assert with_prev < first_step
    # rate penalty = 0.02 * (1^2 + 2^2 + 3^2) = 0.02 * 14 = 0.28
    assert pytest.approx(first_step - with_prev) == 0.28


def test_walk_forward_respects_config_params() -> None:
    task = WalkForwardTask(
        nq=13,
        min_height=1.0,
        forward_coef=5.0,
        heading_coef=0.0,
        upright_coef=0.0,
        action_rate_coef=0.1,
        alive_reward=1.0,
    )
    obs = _make_obs(nq=13, nv=12, height=1.2, yaw=0.0, forward_vel=0.5)
    assert task.terminated(obs) is False
    # alive 1.0 + forward 5.0*0.5*1 = 3.5
    assert pytest.approx(task.reward(obs, np.asarray([2.0], dtype=np.float32))) == 3.5

    low = _make_obs(nq=13, nv=12, height=0.8, yaw=0.0, forward_vel=0.5)
    assert task.terminated(low) is True
    assert task.reward(low, np.asarray([2.0], dtype=np.float32)) == 0.0


def test_walk_forward_registry_builds_with_params() -> None:
    assert TASKS["walk_forward"] is WalkForwardTask
    task = TASKS["walk_forward"](nq=13, min_height=0.7)
    assert task.nq == 13
    assert task.min_height == 0.7


def test_rotate_vector_rotates_wxyz_quaternion() -> None:
    quarter_turn = np.asarray([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    forward = rotate_vector(quarter_turn, (1.0, 0.0, 0.0))
    up = rotate_vector(quarter_turn, (0.0, 0.0, 1.0))
    assert np.allclose(forward, [0.0, 1.0, 0.0])
    assert np.allclose(up, [0.0, 0.0, 1.0])

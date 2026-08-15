import numpy as np
import pytest

from weir.tasks import (
    TASKS,
    StandingTask,
    SurviveTask,
    WalkForwardTask,
    _rotate_vector,
)


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
    assert pytest.approx(task.reward(slow, action)) == 2.3


def test_walk_forward_sideways_motion_reduces_reward() -> None:
    task = WalkForwardTask(nq=19)
    action = np.zeros(1, dtype=np.float32)
    aligned = _make_obs(height=1.2, yaw=0.0, forward_vel=1.0)
    sideways = _make_obs(height=1.2, yaw=np.pi / 2.0, forward_vel=1.0)

    assert task.reward(sideways, action) < task.reward(aligned, action)
    assert pytest.approx(task.reward(sideways, action)) == 1.0


def test_walk_forward_backwards_motion_reduces_reward() -> None:
    task = WalkForwardTask(nq=19)
    action = np.zeros(1, dtype=np.float32)
    aligned = _make_obs(height=1.2, yaw=0.0, forward_vel=1.0)
    backwards = _make_obs(height=1.2, yaw=np.pi, forward_vel=1.0)

    assert task.reward(backwards, action) < task.reward(aligned, action)
    assert task.reward(backwards, action) < 0.0


def test_walk_forward_action_penalty_reduces_reward() -> None:
    task = WalkForwardTask(nq=19)
    obs = _make_obs(height=1.2, yaw=0.0, forward_vel=1.0)
    zero = np.zeros(1, dtype=np.float32)
    busy = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    assert task.reward(obs, busy) < task.reward(obs, zero)
    assert pytest.approx(task.reward(obs, zero) - task.reward(obs, busy)) == 0.035


def test_walk_forward_respects_config_params() -> None:
    task = WalkForwardTask(
        nq=13,
        min_height=1.0,
        forward_coef=5.0,
        heading_coef=0.0,
        upright_coef=0.0,
        action_penalty_coef=0.1,
        alive_reward=1.0,
    )
    obs = _make_obs(nq=13, nv=12, height=1.2, yaw=0.0, forward_vel=0.5)
    assert task.terminated(obs) is False
    assert pytest.approx(task.reward(obs, np.asarray([2.0], dtype=np.float32))) == 3.4

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
    forward = _rotate_vector(quarter_turn, (1.0, 0.0, 0.0))
    up = _rotate_vector(quarter_turn, (0.0, 0.0, 1.0))
    assert np.allclose(forward, [0.0, 1.0, 0.0])
    assert np.allclose(up, [0.0, 0.0, 1.0])

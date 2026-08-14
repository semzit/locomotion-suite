import numpy as np

from weir.tasks import TASKS, StandingTask, SurviveTask


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

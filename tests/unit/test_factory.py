import pytest

from weir.factory import create_algorithm, create_sim


def test_create_sim_returns_mujoco() -> None:
    assert create_sim("mujoco").__class__.__name__ == "MuJoCoSim"


def test_create_sim_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown sim backend"):
        create_sim("isaac_lab")


def test_create_algorithm_returns_ppo() -> None:
    assert create_algorithm("ppo").__class__.__name__ == "PPOAlgorithm"


def test_create_algorithm_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown algorithm"):
        create_algorithm("sac")

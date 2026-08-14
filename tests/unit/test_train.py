from __future__ import annotations

import os
from pathlib import Path

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from weir.algo.ppo import PPOAlgorithm
from weir.envs.mujoco import MuJoCoSim
from weir.train import check_conformance, run
from weir.utils import CONFIG_DIR

ROOT = Path(__file__).parents[2]
CONFIG_RELATIVE = str(Path(os.path.relpath(CONFIG_DIR, Path(__file__).parent)))


def make_config(overrides: list[str] | None = None) -> DictConfig:
    with initialize(version_base=None, config_path=CONFIG_RELATIVE):
        return compose(config_name="train", overrides=overrides or [])


def test_config_composes_defaults() -> None:
    cfg = make_config()
    assert cfg.agent.name == "cartpole"
    assert cfg.task.name == "survive"
    assert cfg.sim.plugin == "mujoco"
    assert cfg.algo.plugin == "ppo"
    assert cfg.train.total_steps == 100000


def test_config_agent_override_swaps_model() -> None:
    cfg = make_config(["agent=simple_humanoid"])
    assert cfg.agent.name == "simple_humanoid"
    assert "simple_humanoid.xml" in str(cfg.agent.model)


def test_config_task_override_applies_params() -> None:
    cfg = make_config(["task=standing"])
    assert cfg.task.name == "standing"
    assert cfg.task.params.min_height == 0.8


def test_run_trains_a_short_run(caplog: pytest.LogCaptureFixture) -> None:
    cfg = make_config(["train.total_steps=128", "algo.n_steps=64"])
    with caplog.at_level("INFO", logger="weir"):
        result = run(cfg)

    assert result["steps"] == 128
    events = [record.message for record in caplog.records]
    assert "train.start" in events
    assert "train.complete" in events
    assert caplog.records[0].sim == "mujoco"
    assert caplog.records[0].algo == "ppo"


def test_conformance_passes_for_reference_plugins() -> None:
    check_conformance(MuJoCoSim(), PPOAlgorithm())


def test_conformance_rejects_non_sim() -> None:
    class NotASim: ...

    with pytest.raises(TypeError, match="SimBackend"):
        check_conformance(NotASim(), PPOAlgorithm())  # type: ignore[arg-type]


def test_conformance_rejects_non_algorithm() -> None:
    class NotAnAlgorithm: ...

    with pytest.raises(TypeError, match="AlgorithmPlugin"):
        check_conformance(MuJoCoSim(), NotAnAlgorithm())  # type: ignore[arg-type]

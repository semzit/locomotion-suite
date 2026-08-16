from __future__ import annotations

import os
from pathlib import Path

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from weir.cli.eval import run_eval
from weir.cli.train import run
from weir.core.run import build_sim
from weir.core.utils import CONFIG_DIR
from weir.envs.backends.mujoco import MuJoCoSim
from weir.envs.wrappers.randomized import RandomizedSim

ROOT = Path(__file__).parents[2]
CONFIG_RELATIVE = str(Path(os.path.relpath(CONFIG_DIR, Path(__file__).parent)))


def make_config(overrides: list[str] | None = None) -> DictConfig:
    with initialize(version_base=None, config_path=CONFIG_RELATIVE):
        return compose(config_name="train", overrides=overrides or [])


def test_config_composes_defaults() -> None:
    cfg = make_config()
    assert cfg.agent.name == "cartpole"
    assert cfg.task.name == "balance"
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


def test_run_trains_a_short_run(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = make_config(["train.total_steps=128", "algo.n_steps=64"])
    monkeypatch.chdir(tmp_path)
    with caplog.at_level("INFO", logger="weir"):
        result = run(cfg)

    assert result["steps"] == 128
    assert (tmp_path / "checkpoint.zip").exists()
    assert (tmp_path / "checkpoint.meta.json").exists()
    events = [record.message for record in caplog.records]
    assert "train.start" in events
    assert "train.complete" in events
    assert caplog.records[0].sim == "mujoco"
    assert caplog.records[0].algo == "ppo"


def test_run_manifest_drives_eval(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    run(make_config(["train.total_steps=32", "algo.n_steps=32"]))

    checkpoint = tmp_path / "checkpoint.zip"
    assert (tmp_path / "checkpoint.meta.json").exists()
    metrics = run_eval(checkpoint, episodes=1, seed=0, max_steps=20)
    assert set(metrics) == {
        "mean_reward",
        "mean_episode_length",
        "total_forward_distance",
        "mean_forward_distance_per_episode",
        "episodes_completed",
    }
    assert metrics["mean_episode_length"] == 20.0


def test_run_eval_override_layers_on_manifest_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    run(make_config(["train.total_steps=32", "algo.n_steps=32"]))
    checkpoint = tmp_path / "checkpoint.zip"

    metrics = run_eval(
        checkpoint, episodes=1, seed=0, max_steps=10, overrides=["train.total_steps=5"]
    )
    assert metrics["mean_episode_length"] == 10.0


def test_run_resumes_from_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    run(make_config(["train.total_steps=32", "algo.n_steps=32"]))
    checkpoint = tmp_path / "checkpoint.zip"
    assert checkpoint.exists()

    cfg = make_config(
        [
            "train.total_steps=32",
            "algo.n_steps=32",
            f"algo.checkpoint={checkpoint}",
        ]
    )
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    monkeypatch.chdir(resume_dir)
    result = run(cfg)
    assert result["steps"] == 32


def test_build_sim_wraps_when_robust() -> None:
    assert isinstance(build_sim({"plugin": "mujoco"}), MuJoCoSim)
    hardened = build_sim({"plugin": "mujoco", "robust": True, "randomization": {}})
    assert isinstance(hardened, RandomizedSim)

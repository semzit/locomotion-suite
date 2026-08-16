from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from weir.cli.utils import setup_logging
from weir.core.contracts import Shape
from weir.core.factory import create_algorithm
from weir.core.run import MANIFEST_NAME, Run
from weir.core.utils import CONFIG_DIR, config_to_dict, log_event
from weir.envs.gym_env import GymEnv

logger = logging.getLogger("weir")


def _shape_record(shape: Shape) -> dict[str, object]:
    return {"dims": list(shape.dims), "dtype": shape.dtype}


def _write_manifest(
    checkpoint: Path,
    *,
    agent: dict[str, Any],
    task: dict[str, Any],
    sim_config: dict[str, Any],
    algorithm_config: dict[str, Any],
    seed: int,
    total_steps: int,
    observation_shape: Shape,
    action_shape: Shape,
) -> Path:
    """Write the run manifest so downstream tools can rebuild the environment."""
    manifest = checkpoint.with_name(MANIFEST_NAME)
    payload = {
        "version": 1,
        "agent": agent,
        "task": task,
        "sim": sim_config,
        "algo": algorithm_config,
        "train": {"seed": seed, "total_steps": total_steps},
        "observation_shape": _shape_record(observation_shape),
        "action_shape": _shape_record(action_shape),
    }
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest


def run(cfg: DictConfig) -> dict[str, Any]:
    """Train the algorithm on the simulator for the configured number of steps."""
    agent = config_to_dict(cfg.agent, "model")
    sim_config = config_to_dict(cfg.sim)
    task = config_to_dict(cfg.task)
    algorithm_config = config_to_dict(cfg.algo, "checkpoint")

    sim = Run.build_sim(sim_config)
    algorithm = create_algorithm(str(algorithm_config["plugin"]))

    sim.load(agent, {**sim_config, "task": task})
    observation_shape = sim.observation_shape()
    action_shape = sim.action_shape()
    algorithm.configure(observation_shape, action_shape, algorithm_config)

    total_steps = int(cfg.train.total_steps)

    log_event(
        logger,
        "train.start",
        sim=sim_config["plugin"],
        algo=algorithm_config["plugin"],
        agent=agent.get("name"),
        observation_shape=observation_shape,
        action_shape=action_shape,
        total_steps=total_steps,
    )

    env = GymEnv(sim)
    metrics = algorithm.learn(env, total_steps)

    checkpoint = Path.cwd() / "checkpoint.zip"
    algorithm.save(checkpoint)
    _write_manifest(
        checkpoint,
        agent=agent,
        task=task,
        sim_config=sim_config,
        algorithm_config=algorithm_config,
        seed=int(cfg.train.seed),
        total_steps=total_steps,
        observation_shape=observation_shape,
        action_shape=action_shape,
    )
    env.close()

    log_event(
        logger,
        "train.complete",
        steps=total_steps,
        checkpoint=str(checkpoint),
        **metrics,
    )
    return {"steps": total_steps, **metrics}


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="train")
def main(cfg: DictConfig) -> None:
    setup_logging()
    run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())

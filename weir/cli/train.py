from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from weir.cli.utils import setup_logging
from weir.core.contracts import SimBackend
from weir.core.factory import create_algorithm, create_sim
from weir.core.utils import CONFIG_DIR, config_to_dict, log_event, resolve_model_path
from weir.envs.gym_env import GymEnv
from weir.envs.wrappers.randomized import RandomizedSim

logger = logging.getLogger("weir")


def build_sim(sim_config: dict[str, Any]) -> SimBackend:
    """Construct the simulator named in config, wrapping it when hardened."""
    sim = create_sim(str(sim_config["plugin"]))
    if sim_config.get("robust", False):
        sim = RandomizedSim(sim, sim_config.get("randomization", {}))
    return sim


def run(cfg: DictConfig) -> dict[str, Any]:
    """Train the algorithm on the simulator for the configured number of steps."""
    agent = config_to_dict(cfg.agent)
    if "model" in agent:
        agent["model"] = resolve_model_path(str(agent["model"]))
    sim_config = config_to_dict(cfg.sim)
    task = config_to_dict(cfg.task)
    algorithm_config = config_to_dict(cfg.algo)

    sim = build_sim(sim_config)
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

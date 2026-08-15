from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from weir.core.contracts import AlgorithmPlugin, SimBackend
from weir.core.factory import create_algorithm, create_sim
from weir.core.utils import CONFIG_DIR, config_to_dict, log_event, resolve_model_path
from weir.envs.gym.gym_env import GymEnv
from weir.envs.randomized import RandomizedSim

logger = logging.getLogger("weir")


def build_sim(sim_config: dict[str, Any]) -> SimBackend:
    """Construct the simulator named in config, wrapping it when hardened."""
    sim = create_sim(str(sim_config["plugin"]))
    if sim_config.get("robust", False):
        sim = RandomizedSim(sim, sim_config.get("randomization", {}))
    return sim


def check_conformance(sim: SimBackend, algorithm: AlgorithmPlugin) -> None:
    """Startup guard: fail fast if a configured plugin does not meet its protocol."""
    if not isinstance(sim, SimBackend):
        raise TypeError(f"{type(sim).__name__} does not implement SimBackend")
    if not isinstance(algorithm, AlgorithmPlugin):
        raise TypeError(f"{type(algorithm).__name__} does not implement AlgorithmPlugin")


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
    check_conformance(sim, algorithm)

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
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(cfg)


if __name__ == "__main__":
    main()

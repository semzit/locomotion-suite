from __future__ import annotations

import logging
from typing import Any

import hydra
from omegaconf import DictConfig

from weir.contracts import AlgorithmPlugin, SimBackend
from weir.factory import create_algorithm, create_sim
from weir.utils import CONFIG_DIR, config_to_dict, log_event, resolve_model_path

logger = logging.getLogger("weir")


def check_conformance(sim: SimBackend, algorithm: AlgorithmPlugin) -> None:
    """Startup guard: fail fast if a configured plugin does not meet its protocol."""
    if not isinstance(sim, SimBackend):
        raise TypeError(f"{type(sim).__name__} does not implement SimBackend")
    if not isinstance(algorithm, AlgorithmPlugin):
        raise TypeError(f"{type(algorithm).__name__} does not implement AlgorithmPlugin")


def run(cfg: DictConfig) -> dict[str, Any]:
    """Execute a Milestone-1 rollout: step the simulator with the policy, logging shapes."""
    agent = config_to_dict(cfg.agent)
    if "model" in agent:
        agent["model"] = resolve_model_path(str(agent["model"]))
    sim_config = config_to_dict(cfg.sim)
    task = config_to_dict(cfg.task)
    algorithm_config = config_to_dict(cfg.algo)

    sim = create_sim(str(sim_config["plugin"]))
    algorithm = create_algorithm(str(algorithm_config["plugin"]))
    check_conformance(sim, algorithm)

    sim.load(agent, {**sim_config, "task": task})
    observation_shape = sim.observation_shape()
    action_shape = sim.action_shape()
    algorithm.configure(observation_shape, action_shape, algorithm_config)

    total_steps = int(cfg.train.total_steps)
    log_every = int(cfg.train.log_every)

    log_event(
        logger,
        "rollout.start",
        sim=sim_config["plugin"],
        algo=algorithm_config["plugin"],
        agent=agent.get("name"),
        observation_shape=observation_shape,
        action_shape=action_shape,
        total_steps=total_steps,
    )

    observation = sim.reset(seed=int(cfg.train.seed))
    for step in range(total_steps):
        action = algorithm.act(observation, deterministic=False)
        result = sim.step(action)
        if step % log_every == 0:
            log_event(
                logger,
                "rollout.step",
                step=step,
                reward=result.reward,
                terminated=result.terminated,
                truncated=result.truncated,
            )
        observation = sim.reset() if result.terminated or result.truncated else result.observation

    sim.close()
    log_event(logger, "rollout.complete", steps=total_steps)
    return {"steps": total_steps}


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="train")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(cfg)


if __name__ == "__main__":
    main()

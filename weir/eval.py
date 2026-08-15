from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from hydra import compose, initialize
from omegaconf import DictConfig

from weir.contracts import AlgorithmPlugin, SimBackend
from weir.factory import create_algorithm, create_sim
from weir.utils import CONFIG_DIR, config_to_dict, log_event, resolve_model_path

logger = logging.getLogger("weir")

CONFIG_RELATIVE = str(os.path.relpath(CONFIG_DIR, Path(__file__).parent))


def compose_config(agent: str, task: str) -> DictConfig:
    """Compose the train config with the requested agent and task groups."""
    with initialize(version_base=None, config_path=CONFIG_RELATIVE):
        return compose(config_name="train", overrides=[f"agent={agent}", f"task={task}"])


def rollout_metrics(
    sim: SimBackend,
    algorithm: AlgorithmPlugin,
    episodes: int,
    *,
    seed: int,
    max_steps: int,
) -> dict[str, float]:
    """Roll out deterministic episodes, summarizing reward, length, and forward progress.

    Forward displacement is the change in observation[0] (root x position). An
    episode counts as completed when it ends on a sim signalled termination or
    truncation rather than exhausting the per-episode step budget.
    """
    total_reward = 0.0
    total_length = 0
    total_forward = 0.0
    completed = 0
    for episode in range(episodes):
        observation = sim.reset(seed=seed + episode)
        initial_x = float(np.asarray(observation)[0])
        episode_reward = 0.0
        episode_length = 0
        finished = False
        for _ in range(max_steps):
            action = algorithm.act(observation, deterministic=True)
            step = sim.step(action)
            observation = step.observation
            episode_reward += float(step.reward)
            episode_length += 1
            if step.terminated or step.truncated:
                finished = True
                break
        total_reward += episode_reward
        total_length += episode_length
        total_forward += float(np.asarray(observation)[0]) - initial_x
        if finished:
            completed += 1
    return {
        "mean_reward": total_reward / episodes,
        "mean_episode_length": total_length / episodes,
        "total_forward_distance": total_forward,
        "mean_forward_distance_per_episode": total_forward / episodes,
        "episodes_completed": float(completed),
    }


def run_eval(
    checkpoint: Path,
    *,
    agent: str,
    task: str,
    episodes: int,
    seed: int,
    max_steps: int,
) -> dict[str, float]:
    """Load a checkpoint and evaluate it with deterministic rollouts."""
    cfg = compose_config(agent, task)
    agent_config = config_to_dict(cfg.agent)
    if "model" in agent_config:
        agent_config["model"] = resolve_model_path(str(agent_config["model"]))
    sim_config = config_to_dict(cfg.sim)
    task_config = config_to_dict(cfg.task)
    algorithm_config = config_to_dict(cfg.algo)

    sim = create_sim(str(sim_config["plugin"]))
    algorithm = create_algorithm(str(algorithm_config["plugin"]))
    sim.load(agent_config, {**sim_config, "task": task_config})
    algorithm.load(checkpoint)
    try:
        return rollout_metrics(sim, algorithm, episodes, seed=seed, max_steps=max_steps)
    finally:
        sim.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weir-eval",
        description="Evaluate a trained checkpoint with deterministic rollouts.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to the algorithm checkpoint."
    )
    parser.add_argument(
        "--agent",
        default="humanoid",
        help="Agent config group name (configs/agent/).",
    )
    parser.add_argument(
        "--task",
        default="standing",
        help="Task config group name (configs/task/).",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Episodes to roll out.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the rollouts.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        metrics = run_eval(
            args.checkpoint,
            agent=args.agent,
            task=args.task,
            episodes=args.episodes,
            seed=args.seed,
            max_steps=args.max_steps,
        )
    except Exception as error:
        log_event(logger, "eval.failed", error=str(error))
        print(f"weir-eval: {error}", file=sys.stderr)
        return 1
    log_event(
        logger,
        "eval.complete",
        checkpoint=str(args.checkpoint),
        agent=args.agent,
        task=args.task,
        **metrics,
    )
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

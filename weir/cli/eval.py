from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from weir.cli.utils import add_checkpoint_arg, add_seed_arg, compose_config, guarded_main
from weir.core.contracts import AlgorithmPlugin, SimBackend
from weir.core.factory import create_algorithm, create_sim
from weir.core.run import Run
from weir.core.utils import config_to_dict, resolve_model_path


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
    agent: str | None = None,
    task: str | None = None,
    episodes: int,
    seed: int,
    max_steps: int,
    overrides: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate a checkpoint, driven by its manifest unless flags override it."""
    run = Run.open(checkpoint)
    if run.config is not None and not (agent or task or overrides):
        sim = run.sim()
        run.validate(sim)
        algorithm = run.algorithm()
    else:
        cfg = compose_config(agent or "humanoid", task or "standing", overrides)
        agent_config = config_to_dict(cfg.agent)
        if "model" in agent_config:
            agent_config["model"] = resolve_model_path(str(agent_config["model"]))
        sim_config = config_to_dict(cfg.sim)
        task_config = config_to_dict(cfg.task)
        algorithm_config = config_to_dict(cfg.algo)
        sim = create_sim(str(sim_config["plugin"]))
        sim.load(agent_config, {**sim_config, "task": task_config})
        run.validate(sim)
        algorithm = create_algorithm(str(algorithm_config["plugin"]))
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
    add_checkpoint_arg(parser)
    parser.add_argument(
        "--agent",
        default=None,
        help="Agent config group name; defaults to the run's manifest (configs/agent/).",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task config group name; defaults to the run's manifest (configs/task/).",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Episodes to roll out.")
    add_seed_arg(parser, help="Seed for the rollouts.")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum steps per episode.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override, repeatable (e.g. --override task.params.nq=13).",
    )
    return parser


def _run_eval_from_args(args: argparse.Namespace) -> dict[str, Any]:
    metrics = run_eval(
        args.checkpoint,
        agent=args.agent,
        task=args.task,
        episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        overrides=args.override,
    )
    run = Run.open(args.checkpoint)
    config = run.config or {}
    agent_name = args.agent or str(config.get("agent", {}).get("name", "unknown"))
    task_name = args.task or str(config.get("task", {}).get("name", "unknown"))
    return {
        "metrics": metrics,
        "checkpoint": str(args.checkpoint),
        "agent": agent_name,
        "task": task_name,
    }


def _print_metrics(result: dict[str, Any]) -> None:
    for key, value in result["metrics"].items():
        print(f"{key}: {value:.3f}")


def main(argv: list[str] | None = None) -> int:
    return guarded_main(
        build_parser(),
        _run_eval_from_args,
        "eval",
        on_success=_print_metrics,
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())

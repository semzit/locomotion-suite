from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio

from weir.contracts import AlgorithmPlugin
from weir.envs.mujoco import MuJoCoSim
from weir.factory import create_algorithm
from weir.utils import resolve_model_path


def render_episode(
    sim: MuJoCoSim,
    algo: AlgorithmPlugin,
    output_path: Path,
    *,
    frames_to_capture: int | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    seed: int = 0,
) -> Path:
    """Roll out a policy and write the frames to an mp4 file at output_path."""
    frames: list[Any] = []
    observation = sim.reset(seed=seed)
    while frames_to_capture is None or len(frames) < frames_to_capture:
        action = algo.act(observation, deterministic=False)
        result = sim.step(action)
        frames.append(sim.render_frame(width, height))
        if result.terminated or result.truncated:
            break
    imageio.mimsave(output_path, frames, fps=fps)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weir-render",
        description="Render a MuJoCo rollout to an mp4 video.",
    )
    parser.add_argument("--model", default="weir/models/cartpole.xml")
    parser.add_argument("--task", default="survive")
    parser.add_argument("--time-limit", type=float, default=5.0)
    parser.add_argument("--output", default="video.mp4")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sim = MuJoCoSim()
    try:
        sim.load(
            {"name": "render", "model": resolve_model_path(args.model)},
            {"task": {"name": args.task, "params": {}}, "time_limit": args.time_limit},
        )
        algo = create_algorithm("ppo")
        algo.configure(sim.observation_shape(), sim.action_shape(), {})
        output_path = render_episode(
            sim,
            algo,
            Path(args.output),
            frames_to_capture=args.frames,
            width=args.width,
            height=args.height,
            fps=args.fps,
            seed=args.seed,
        )
    except RuntimeError as error:
        print(f"weir-render: {error}", file=sys.stderr)
        return 1
    finally:
        sim.close()
    print(f"Rendered rollout to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

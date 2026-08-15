from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio

from weir.cli.utils import add_seed_arg
from weir.core.contracts import AlgorithmPlugin
from weir.core.factory import create_algorithm
from weir.core.run import Run
from weir.core.utils import resolve_model_path
from weir.envs.backends.mujoco import MuJoCoSim


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
    parser.add_argument(
        "--model", default=None, help="Model XML path; defaults to the run's manifest."
    )
    parser.add_argument("--task", default=None, help="Task name; defaults to the run's manifest.")
    parser.add_argument(
        "--task-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Task parameter override, repeatable (e.g. --task-param nq=13).",
    )
    parser.add_argument("--time-limit", type=float, default=5.0)
    parser.add_argument("--output", default="video.mp4")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--algo", default="ppo", help="Algorithm name registered in the factory.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Trained algorithm checkpoint to play; defaults to a fresh random policy.",
    )
    add_seed_arg(parser)
    return parser


def _parse_task_params(entries: list[str]) -> dict[str, object]:
    params: dict[str, object] = {}
    for entry in entries:
        key, sep, value = entry.partition("=")
        if not key or not sep:
            raise ValueError(f"Invalid task param (expected KEY=VALUE): {entry!r}")
        try:
            params[key] = float(value) if "." in value else int(value)
        except ValueError:
            params[key] = value
    return params


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run = Run.open(args.checkpoint) if args.checkpoint else None
    overridden = args.model or args.task or args.task_param
    sim = MuJoCoSim()
    try:
        if run is not None and run.config is not None and not overridden:
            sim = run.sim()
            run.validate(sim)
            if not isinstance(sim, MuJoCoSim):
                raise ValueError(
                    f"Render requires a MuJoCoSim, but the manifest uses {type(sim).__name__}"
                )
            algo = run.algorithm()
        else:
            model = args.model or "weir/models/cartpole.xml"
            sim.load(
                {"name": "render", "model": resolve_model_path(model)},
                {
                    "task": {
                        "name": args.task or "survive",
                        "params": _parse_task_params(args.task_param),
                    },
                    "time_limit": args.time_limit,
                },
            )
            if run is not None:
                run.validate(sim)
            algo = create_algorithm(args.algo)
            if args.checkpoint:
                algo.load(args.checkpoint)
            else:
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
    except (RuntimeError, ValueError) as error:
        print(f"weir-render: {error}", file=sys.stderr)
        return 1
    finally:
        sim.close()
    print(f"Rendered rollout to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

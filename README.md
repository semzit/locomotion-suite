<p align="center">
  <img src="assets/weir-hero.jpg" alt="Weir" width="520" />
</p>

# Weir

Train a simulated legged agent to walk using reinforcement learning. A small, self-contained
project: a `SimBackend` and an `AlgorithmPlugin` Protocol, a training loop that never imports a
concrete implementation, Hydra config, and ONNX policy export.

The simulator and the algorithm are both interchangeable, chosen at the command line rather than
hardcoded — so you can compare SB3 PPO against RLtools, or MuJoCo against Isaac Lab, without touching the
training loop. Where the run executes is deliberately *not* interchangeable yet.

## Quick start

```bash
uv run weir-train agent=cartpole task=survive
uv run weir-train agent=simple_humanoid task=standing   # swap the agent, same pipeline
uv run weir-train agent=humanoid                        # menagerie humanoid, default task
```

`weir/cli/train.py` imports only the Protocols; Hydra selects which concrete class satisfies each
interface from the config groups under `configs/` (`agent/`, `task/`, `sim/`, `algo/`).
Sibling entry points round out the workflow:

```bash
uv run weir-eval --checkpoint checkpoint.zip        # rollout metrics on a trained policy
uv run weir-export --checkpoint checkpoint.zip      # export the policy to standalone .onnx
python -m weir.cli.render --model weir/models/humanoid.xml   # render a rollout to mp4
```

## Development checks

Create the dev environment once:

```bash
uv sync --group dev
```

Then run the checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

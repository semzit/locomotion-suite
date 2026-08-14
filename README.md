<p align="center">
  <img src="assets/weir-hero.jpg" alt="Weir" width="520" />
</p>

# Weir

Train a simulated legged agent to walk using reinforcement learning. A small, self-contained
project: a `SimBackend` and an `AlgorithmPlugin` Protocol, a training loop that never imports a
concrete implementation, Hydra config, and ONNX policy export.

The simulator and the algorithm are both interchangeable, chosen at the command line rather than
hardcoded — so you can compare PPO against SAC, or MuJoCo against Isaac Lab, without touching the
training loop. Where the run executes is deliberately *not* interchangeable yet.

## Quick start

```bash
uv run weir-train agent=cartpole task=survive
uv run weir-train agent=simple_humanoid task=standing   # swap the agent, same pipeline
uv run weir-train agent=humanoid                        # menagerie humanoid, default task
```

`train.py` imports only the Protocols; Hydra selects which concrete class satisfies each
interface from the config groups under `configs/` (`agent/`, `task/`, `sim/`, `algo/`).
The trained policy exports to a standalone `.onnx` file via `weir/export.py`.

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

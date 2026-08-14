# Weir

Train a simulated legged robot to walk using reinforcement learning. A small, self-contained
project: a `SimBackend` and an `AlgorithmPlugin` Protocol, a training loop that never imports a
concrete implementation, Hydra config, and ONNX policy export.

See [docs/PLAN.md](docs/PLAN.md) for the full plan, [docs/overview.md](docs/overview.md) for the
short version, and [docs/engineering-rules.md](docs/engineering-rules.md) for engineering rules.

## Quick start

```bash
uv run weir-train robot=cartpole task=survive
uv run weir-train robot=simple_humanoid task=standing   # swap the robot, same pipeline
uv run weir-train robot=humanoid                        # menagerie humanoid, default task
```

`train.py` imports only the Protocols; Hydra selects which concrete class satisfies each
interface from the config groups under `configs/` (`robot/`, `task/`, `sim/`, `algo/`).
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

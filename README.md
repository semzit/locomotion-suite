# Weir

Train a simulated legged robot to walk using reinforcement learning. A small, self-contained
project: a `SimBackend` and an `AlgorithmPlugin` Protocol, a training loop that never imports a
concrete implementation, Hydra config, and ONNX policy export.

See [docs/PLAN.md](docs/PLAN.md) for the full plan, [docs/overview.md](docs/overview.md) for the
short version, and [docs/engineering-rules.md](docs/engineering-rules.md) for engineering rules.

## Quick start

```bash
python weir/train.py sim=mujoco algo=ppo
python weir/train.py sim=mujoco algo=sac         # swap the algorithm, same simulator
python weir/train.py sim=isaac_lab algo=ppo      # swap the simulator, same algorithm
```

`train.py` imports only the Protocols; Hydra selects which concrete class satisfies each interface.
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

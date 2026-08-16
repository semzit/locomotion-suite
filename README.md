# Weir

Train a simulated legged agent to walk using reinforcement learning. A small, self-contained
project: a `SimBackend` and an `AlgorithmPlugin` Protocol, a training loop that never imports a
concrete implementation, Hydra config, and ONNX policy export.

The simulator and the algorithm are both interchangeable, chosen at the command line rather than
hardcoded — so you can compare SB3 PPO against RLtools, or MuJoCo against Isaac Lab, without
touching the training loop.

## Workflow

Train, evaluate, record, and export a walking policy:

```bash
# 1. Train PPO on the simple humanoid, walking task
uv run weir-train agent=simple_humanoid task=walk_forward

# 2. Evaluate the trained policy (mean reward, episode length, forward distance)
uv run weir-eval --checkpoint outputs/2026-08-15/<run>/checkpoint.zip

# 3. Record an mp4 of the policy walking
uv run weir-render --checkpoint outputs/2026-08-15/<run>/checkpoint.zip --output walk.mp4

# 4. Export the policy to a standalone .onnx file
uv run weir-export --checkpoint outputs/2026-08-15/<run>/checkpoint.zip
```

Every run writes a manifest (`checkpoint.meta.json`) next to the checkpoint recording the
resolved configuration and observation/action shapes. `weir-eval`, `weir-render`, and
`weir-export` rebuild the environment from it, so a checkpoint can't be paired with the
wrong model — no flags needed. Flags remain as overrides where useful.

`weir/cli/train.py` imports only the Protocols; Hydra selects which concrete class satisfies each
interface from the config groups under `configs/` (`agent/`, `task/`, `sim/`, `algo/`).

### Swapping pieces at the command line

- `agent=cartpole` / `agent=simple_humanoid` / `agent=humanoid` — different robots, same pipeline
- `task=balance` / `task=standing` / `task=walk_forward` — different objectives (balance is the learnable toy task; `task=survive` gives a constant reward and learns nothing by design)
- `sim=mujoco` — `SimBackend` implementations live in `weir/envs/backends/`
- `algo=ppo` — `AlgorithmPlugin` implementations live in `weir/algo/`
- `algo.checkpoint=<path>` — resume training from an existing checkpoint

## Repository layout

```
weir/
├── cli/            # entry points: train, eval, export, render
├── core/           # protocols, tasks, factory, shared utils
├── envs/
│   ├── backends/   # SimBackend implementations (mujoco)
│   ├── wrappers/   # SimBackend decorators (sim-to-real hardening)
│   └── gym_env.py  # gymnasium adapter
└── algo/           # AlgorithmPlugin implementations (ppo)
configs/            # Hydra config groups: agent/, task/, sim/, algo/
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

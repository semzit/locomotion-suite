# Weir

Train a simulated legged agent to walk using reinforcement learning. A small, self-contained
project: a `SimBackend` and an `AlgorithmPlugin` Protocol, a training loop that never imports a
concrete implementation, Hydra config, and ONNX policy export.

The simulator and the algorithm are both interchangeable, chosen at the command line rather than
hardcoded — so you can compare SB3 PPO against RLtools, or MuJoCo against Isaac Lab, without
touching the training loop.

## Quick demo

See the pipeline in action in about three minutes — no GPU needed:

```bash
# 1. Train PPO on CartPole balance (~3 min on CPU)
uv run weir-train agent=cartpole task=balance train.total_steps=100000

# 2. Point at the newest checkpoint
CHECKPOINT=$(ls -t outputs/*/*/checkpoint.zip | head -1)

# 3. Verify: episode length should reach the 500-step horizon
uv run weir-eval --checkpoint "$CHECKPOINT"

# 4. Render a video of the policy balancing — with kicks
uv run weir-render --checkpoint "$CHECKPOINT" --output cartpole.mp4 --frames 250 \
  --perturb-force 8 --perturb-body 2
```

After ~100k steps the policy balances for the entire episode (`mean_episode_length ≈ 500`).
The rendered mp4 shows the classic cart-pole keeping the pole upright while getting kicked
up to ~6° and recovering each time (body 2 is the pole; drop the `--perturb-*` flags for a
calm video). This is the toy task — the walking policy is the real goal, shown in Workflow
below.

Here is what a trained balancer looks like under kicks:

<p align="center" width="100%">
<video src="https://github.com/user-attachments/assets/edd2b375-55d8-43bb-9167-7b81eaeb9c77" width="80%" controls></video>
</p>

The video is hosted as a GitHub user attachment (uploaded via issue #3); the local source
file lives in `assets/cartpole-balance-demo.mp4` (H.264, 24 KB).

## Workflow

Train, evaluate, record, and export a walking policy:

```bash
# 1. Train PPO on the humanoid, walking task
uv run weir-train agent=humanoid task=walk_forward

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

## Configuration

Everything about a run lives in YAML under `configs/`, composed by Hydra at launch.
`configs/train.yaml` declares the defaults: it picks one file from each group — an agent,
a task, a sim, an algorithm — and Hydra merges them into a single config:

```
configs/
├── train.yaml         # composition + train.seed, train.total_steps
├── agent/             # one robot per file: cartpole.yaml, humanoid.yaml
├── task/              # one objective per file: balance, standing, walk_forward, ...
├── sim/               # one backend per file: mujoco.yaml
└── algo/              # one algorithm per file: ppo.yaml
```

**The full reference — every group, every knob, with examples — is in
[`docs/configuration.md`](docs/configuration.md).**

### Overriding at the command line

Any composed value can be overridden with `key=value` arguments, without editing files:

```bash
# Swap a whole group (picks a different file from that group)
uv run weir-train agent=humanoid task=walk_forward

# Override a nested value inside a group
uv run weir-train task.params.min_height=1.0

# Override a top-level value
uv run weir-train train.total_steps=500000

# Combine as many as you like
uv run weir-train agent=humanoid task=walk_forward algo.n_steps=4096 train.total_steps=1000000
```

Notes:
- Values are typed: numbers parse as floats, `true`/`false` as booleans, `[64, 64]` as lists
- Hydra refuses to *invent* keys — adding a brand-new key needs a `+` prefix (`+task.params.thing=1`)
- `algo.checkpoint=<path>` resumes training from an existing checkpoint (weights only)

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

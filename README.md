# Locomotion Suite

Config-driven, plugin-based orchestration for humanoid reinforcement-learning workflows.

```bash
python -m core.cli validate --manifest configs/experiments/humanoid_walk.example.yaml
python -m core.cli plugins
```

## Plugins

Plugin packages register a callable factory in the `locomotion_suite.plugins` entry-point group.
The entry-point name is the identifier used in stack configuration (for example, `ppo`, `mujoco`,
or `local`). List installed plugins with `locomotion plugins`.

## Development checks

Create the dev environment once:

```bash
uv sync --group dev
```

Then run the checks in the project-native style:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

Or run the full validation stack in one go:

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
```

Once this folder is initialized as a Git repository, enable the local checks with
`uv run pre-commit install`.

## Hydra (prototype)

Hydra integration is available as a prototype entrypoint. It provides `hydra`-style
overrides and config composition while reusing the existing manifest-driven runner.

Install the package in editable mode to get the console script:

```bash
python -m pip install -e .
```

Then run with a manifest override:

```bash
locomotion-hydra --manifest=configs/experiments/humanoid_walk.example.yaml
```

Notes:
- Hydra changes the working directory for runs by default (outputs go under `hydra.run.dir`).
- This is a minimal prototype; if you'd like, I can convert the `configs/` tree to Hydra config groups.

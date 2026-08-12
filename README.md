# Locomotion Suite

Config-driven, plugin-based orchestration for humanoid reinforcement-learning workflows.

```bash
python -m core.cli validate --manifest configs/experiments/humanoid_walk.yaml
```

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

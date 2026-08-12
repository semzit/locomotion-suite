# Locomotio Suite

Config-driven, plugin-based orchestration for humanoid reinforcement-learning workflows.

```bash
python -m core.cli validate --manifest experiments/humanoid_walk.yaml
```

## Development checks

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pyright
.venv/bin/pytest
```

Once this folder is initialized as a Git repository, enable the local checks with
`.venv/bin/pre-commit install`.

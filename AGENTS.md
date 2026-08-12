# Engineering rules

## Approved libraries

Use these libraries for their designated concerns; do not introduce an overlapping library without
recording the reason in the change.

- **Pydantic** (`pydantic`): validate and serialize external data, configuration, and persisted
  models.
- **OmegaConf** (`omegaconf`): load and compose YAML configuration, interpolation, and overrides.
- **Beartype** (`beartype`): runtime guards at dynamic plugin and third-party boundaries only.
- **Tenacity** (`tenacity`): bounded retry and backoff for transient, idempotent external calls.
- **Structlog** (`structlog`): structured application logs.
- **OpenTelemetry** (`opentelemetry-api`, `opentelemetry-sdk`): traces, metrics, and log-context
  propagation.
- **Ruff** (`ruff`): linting, import sorting, and formatting.
- **Pyright** (`pyright`): static type checking.
- **pytest** and **pytest-cov**: tests and coverage reporting.
- **Hypothesis** (`hypothesis`): property-based tests for parsers, merge rules, and stateful logic.
- **pre-commit** (`pre-commit`): local enforcement of the configured quality checks.

## Validation and configuration

- Treat YAML, CLI arguments, environment variables, files, and plugin output as untrusted at
  their boundary. Convert them to typed Pydantic models immediately.
- Prefer Pydantic `BaseModel`, `TypeAdapter`, field validators, and discriminated unions over
  repeated `isinstance` checks in application logic.
- Use OmegaConf for configuration loading, hierarchical merges, interpolation, and dotted-path
  overrides. Keep only project-specific policy (such as manifest include resolution) in `core`.
- Use narrow manual checks only when a value is deliberately dynamic (for example, unknown plugin
  payloads) or before a schema can be selected. Put such checks in one boundary function.
- Make schemas strict by default (`extra="forbid"`); allow extra fields only at documented plugin
  extension points.

## Types and interfaces

- Public functions, protocol methods, and persisted data structures must have explicit types.
- Use `Protocol` for plugin contracts and Pydantic models or dataclasses for data, not ad-hoc
  dictionaries across module boundaries.
- Keep `Any` at integration boundaries and replace it with a concrete type as soon as possible.
- Run `uv run pyright` after changing public types, contracts, or configuration schemas.
- Use Beartype only at dynamic plugin or third-party integration boundaries; static type checking
  remains the primary guard inside application code.

## Reliability and observability

- Retry only transient, idempotent external operations with Tenacity. Set bounded attempts and
  exponential backoff; never retry validation failures or non-idempotent side effects by default.
- Emit structured events through `core.observability.logger`, including stable identifiers such as
  experiment name, plugin name, and run ID. Do not log secrets or full untrusted payloads.
- Wrap meaningful CLI, compute, and simulator operations in OpenTelemetry spans from
  `core.observability.tracer`. Keep exporter and deployment configuration outside core logic.

## Tests and quality gates

- Add focused tests for every behavior change, including invalid input at validation boundaries.
- Use pytest for examples and regression tests. Add Hypothesis when merge rules, parsers, or state
  transitions have broad input spaces.
- Before handing off changes, run:

  ```bash
  uv run ruff check .
  uv run ruff format --check .
  uv run pyright
  uv run pytest
  ```

## Dependency policy

- Prefer one well-maintained library with a clear project fit over duplicate handwritten plumbing.
- Do not add a library for a one-line helper or introduce overlapping tools without a stated reason.
- Pin compatible major-version ranges in `pyproject.toml` and commit the matching `uv.lock` change.
- Keep runtime dependencies separate from developer-only quality tools.

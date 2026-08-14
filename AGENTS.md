# Engineering rules

Lines of code is a liability not an asset

## Approved libraries

Use these libraries for their designated concerns; do not introduce an overlapping library without
recording the reason in the change.

- **PyTorch** (`torch`): the deep-learning framework used to write and debug the RL algorithms
  (PPO, SAC) from scratch, and to run the actor/critic networks during training.
- **Pydantic** (`pydantic`): validate and serialize external data and configuration into typed
  models before use — the structured-validation layer on top of loaded Hydra config.
- **Hydra** (`hydra-core`): composable YAML config groups, command-line overrides, and automatic run
  logging. Hydra depends on OmegaConf, so OmegaConf is already in the stack transitively — do not
  add it separately.
- **ONNX** (`onnx`, `onnxruntime`): export a trained policy to a standalone artifact via
  `torch.onnx.export`, and verify the export runs identically through `onnxruntime` alone, without
  PyTorch present.
- **Ruff** (`ruff`): linting, import sorting, and formatting.
- **Pyright** (`pyright`): static type checking. Particularly useful here: it catches a
  `SimBackend`/`AlgorithmPlugin` implementation that is missing a method at type-check time, before
  you run it.
- **pytest** and **pytest-cov**: tests and coverage reporting.
- **Hypothesis** (`hypothesis`): property-based tests aimed at `SimBackend`/`AlgorithmPlugin`
  conformance (e.g. "for any valid observation, `act()` returns an action within the valid range")
  and the ONNX export round-trip (exported model output ≈ PyTorch output across random inputs).
- **pre-commit** (`pre-commit`): local enforcement of the configured quality checks.
- **Structlog** (`structlog`): *optional*. Reasonable if you want nicer logs than stdlib, but this is
  a single in-process training script, not a service — decide whether it earns its place or whether
  stdlib `logging` plus Hydra's own run logging is enough.

Libraries deliberately *not* used:

- **Beartype**: there are no dynamic plugin or third-party boundaries here — every
  `SimBackend`/`AlgorithmPlugin` implementation is code you wrote, imported directly, and known at
  build time. If a runtime Protocol conformance check is wanted, do it as one small startup check in
  `train.py`, not via a runtime-typing guard library.
- **Tenacity**: Weir makes no external calls, so there is nothing transient to retry. This is a
  holdover from a deferred launcher idea (project plan §8); add it back only if/when that launcher
  is actually built.
- **OpenTelemetry**: traces and metrics propagation exist to debug a multi-process distributed
  system. Weir is one process with no span crossing a boundary — out of scope entirely.

## Validation and configuration

- Treat YAML, CLI arguments, and environment variables as untrusted at their boundary. Convert
  them to typed Pydantic models immediately.
- Prefer Pydantic `BaseModel`, `TypeAdapter`, field validators, and discriminated unions over
  repeated `isinstance` checks in application logic.
- Use Hydra for config composition (config groups under `configs/`), command-line overrides, and
  output-directory management. Let Hydra select which concrete class satisfies each
  `SimBackend`/`AlgorithmPlugin` interface — `train.py` imports only the Protocols and never a
  concrete implementation directly.
- Use narrow manual checks only when a value is deliberately dynamic or before a schema can be
  selected. Put such checks in one boundary function.
- Make schemas strict by default (`extra="forbid"`); allow extra fields only at documented
  extension points.

## Types and interfaces

- Public functions, protocol methods, and persisted data structures must have explicit types.
- Use `Protocol` for the `SimBackend` and `AlgorithmPlugin` contracts, and Pydantic models or
  dataclasses for data, not ad-hoc dictionaries across module boundaries.
- Keep `Any` at integration boundaries and replace it with a concrete type as soon as possible.
- Run `uv run pyright` after changing public types, contracts, or configuration schemas.
- Static type checking is the primary guard inside application code. For runtime Protocol
  conformance, prefer a single startup check written directly in `train.py` over a runtime-typing
  library — every implementation is already known at build time, so there is no dynamic/third-party
  boundary to guard.

## Reliability and observability

- Weir is a single in-process training loop with no external calls, so there are no transient
  failures to retry. Do not add retry/backoff machinery; it isn't needed until the deferred compute
  launcher (plan §8) is actually built.
- Log to stdout/file in a headless-friendly format (no live-progress-bar-only output that breaks
  when run as a batch job), and let Hydra own run-directory logging. Structlog is optional, not
  required.
- Keep device/hardware settings and output paths in config (e.g. `hydra.run.dir`), not hardcoded
  literals — this keeps the script runnable unchanged when a future launcher redirects output.

## Tests and quality gates

- Add focused tests for every behavior change, including invalid input at validation boundaries.
- Use pytest for examples and regression tests. Add Hypothesis when a `SimBackend`/`AlgorithmPlugin`
  conformance property or the ONNX export round-trip has a broad input space worth sampling.
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
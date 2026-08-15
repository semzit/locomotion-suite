from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
from omegaconf import OmegaConf

from weir.core.contracts import Shape

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = ROOT / "weir"
CONFIG_DIR = ROOT / "configs"
MODELS_DIR = PACKAGE_DIR / "models"


def resolve_model_asset(name: str) -> str:
    """Resolve a bare model file name (or relative path) under the models dir."""
    return str((MODELS_DIR / name).resolve())


def resolve_model_path(path: str) -> str:
    """Resolve a config model path to an absolute path against the repo root."""
    model_path = Path(path)
    if model_path.is_absolute():
        return str(model_path)
    return str((ROOT / model_path).resolve())


def sample_action(
    action_shape: Shape,
    rng: np.random.Generator,
    deterministic: bool = False,
) -> np.ndarray:
    """Sample an in-bounds action, or its midpoint when deterministic."""
    dims = tuple(action_shape.dims)
    low = action_shape.low
    high = action_shape.high
    if low is not None and high is not None:
        if deterministic:
            return ((low + high) / 2.0).astype(np.float32)
        return rng.uniform(low, high, size=dims).astype(np.float32)
    if deterministic:
        return np.zeros(dims, dtype=np.float32)
    return rng.uniform(-1.0, 1.0, size=dims).astype(np.float32)


def config_to_dict(section: Any) -> dict[str, Any]:
    """Coerce an OmegaConf section into a resolved plain dict."""
    return cast(dict[str, Any], OmegaConf.to_container(section, resolve=True) or {})


_RESERVED = {
    "name",
    "msg",
    "message",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
}


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Log an INFO event with structured fields, avoiding reserved LogRecord keys."""
    extra = {key: value for key, value in fields.items() if key not in _RESERVED}
    logger.info(event, extra=extra)

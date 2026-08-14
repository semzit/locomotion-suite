from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def resolve_model_path(path: str) -> str:
    """Resolve a config model path to an absolute path against the repo root."""
    model_path = Path(path)
    if model_path.is_absolute():
        return str(model_path)
    return str((ROOT / model_path).resolve())


def config_to_dict(section: Any) -> dict[str, Any]:
    """Coerce an OmegaConf section into a resolved plain dict."""
    return cast(dict[str, Any], OmegaConf.to_container(section, resolve=True) or {})


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Log an INFO event with structured fields, avoiding reserved LogRecord keys."""
    reserved = {
        "name",
        "msg",
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
    extra = {key: value for key, value in fields.items() if key not in reserved}
    logger.info(event, extra=extra)

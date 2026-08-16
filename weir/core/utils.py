from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def resolve_model_path(path: str) -> str:
    """Resolve a config model path to an absolute path against the repo root."""
    model_path = Path(path)
    if model_path.is_absolute():
        return str(model_path)
    return str((ROOT / model_path).resolve())


def rotate_vector(quat: np.ndarray, vector: tuple[float, float, float]) -> np.ndarray:
    """Rotate a 3-vector by a unit quaternion in (w, x, y, z) order.

    Pure numpy (no mujoco import): ``v' = v + 2*w*cross(qv, v) + 2*cross(qv, cross(qv, v))``
    where ``qv = (x, y, z)`` is the vector part of the quaternion.
    """
    w, x, y, z = (float(v) for v in quat[0:4])
    norm = float(np.linalg.norm(quat[0:4]))
    w /= norm
    x /= norm
    y /= norm
    z /= norm
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.asarray(
        [
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ],
        dtype=np.float32,
    )


def config_to_dict(section: Any, *path_keys: str) -> dict[str, Any]:
    """Coerce an OmegaConf section into a resolved plain dict.

    Values under *path_keys are resolved to absolute paths via
    resolve_model_path; missing or empty (e.g. null) values are left as-is.
    """
    resolved = cast(dict[str, Any], OmegaConf.to_container(section, resolve=True) or {})
    for key in path_keys:
        if resolved.get(key):
            resolved[key] = resolve_model_path(str(resolved[key]))
    return resolved


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



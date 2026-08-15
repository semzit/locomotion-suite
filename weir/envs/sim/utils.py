from __future__ import annotations

from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def resolve_model_asset(name: str) -> str:
    """Resolve a bare model file name (or relative path) under the models dir."""
    return str((MODELS_DIR / name).resolve())

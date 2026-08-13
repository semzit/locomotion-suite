from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .utils import CONFIG_MAPPING


class ExperimentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    mode: str = "train"
    seed: int = 0


class ExperimentManifest(BaseModel):
    """Resolved experiment with plugin-specific sections retained as documented extensions."""

    model_config = ConfigDict(extra="allow")
    experiment: ExperimentSettings
    includes: list[str] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)
    benchmarks: str | None = None
    gates: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)


class ConfigDocument(BaseModel):
    """Validated common shape; extra fields are plugin-specific config sections."""

    model_config = ConfigDict(extra="allow")
    includes: list[str] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)


def load_manifest(path: str | Path) -> ExperimentManifest:
    """Load a manifest, recursively merge its includes, and apply its overrides."""
    manifest_path = Path(path).resolve()
    data = _load_config(manifest_path, active_paths=())
    return ExperimentManifest.model_validate(data)


def _load_config(path: Path, active_paths: tuple[Path, ...]) -> dict[str, Any]:
    if path in active_paths:
        cycle = " -> ".join(str(item) for item in (*active_paths, path))
        raise ValueError(f"Manifest include cycle: {cycle}")

    try:
        raw_data = OmegaConf.to_container(OmegaConf.load(path), resolve=False) or {}
    except (OSError, OmegaConfBaseException) as error:
        raise ValueError(f"Could not read manifest file: {path}") from error

    try:
        document = ConfigDocument.model_validate(raw_data)
    except ValidationError as error:
        raise ValueError(f"Invalid manifest file: {path}") from error

    merged = OmegaConf.create()
    for include in document.includes:
        include_path = (path.parent / include).resolve()
        merged = OmegaConf.merge(merged, _load_config(include_path, (*active_paths, path)))

    own_data = document.model_dump(exclude={"includes", "overrides"})
    merged = OmegaConf.merge(merged, own_data)
    OmegaConf.update(merged, "includes", document.includes, merge=False)

    for dotted_path, value in document.overrides.items():
        try:
            OmegaConf.update(merged, dotted_path, value, merge=False)
        except OmegaConfBaseException as error:
            raise ValueError(f"Invalid override path: {dotted_path!r}") from error
    OmegaConf.update(merged, "overrides", document.overrides, merge=False)
    data = OmegaConf.to_container(merged, resolve=True)
    return CONFIG_MAPPING.validate_python(data)

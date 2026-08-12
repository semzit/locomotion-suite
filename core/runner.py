from __future__ import annotations

from .manifest import ExperimentManifest


class ExperimentRunner:
    """Coordinates a resolved experiment; execution arrives with the reference stack."""

    def validate(self, manifest: ExperimentManifest) -> None:
        if manifest.experiment.mode not in {"train", "eval", "export", "search"}:
            raise ValueError(f"Unsupported run mode: {manifest.experiment.mode}")

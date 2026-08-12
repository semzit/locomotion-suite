from pathlib import Path

from core.manifest import load_manifest
from core.runner import ExperimentRunner


def test_reference_manifest_validates() -> None:
    root = Path(__file__).parents[2]
    manifest = load_manifest(root / "experiments" / "humanoid_walk.yaml")
    ExperimentRunner().validate(manifest)
    assert manifest.experiment.name == "humanoid_walk"

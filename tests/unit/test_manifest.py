from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import TypeAdapter

from core.manifest import load_manifest
from core.runner import ExperimentRunner
from core.utils import build_parser


def test_reference_manifest_validates() -> None:
    root = Path(__file__).parents[2]
    manifest = load_manifest(root / "configs" / "experiments" / "humanoid_walk.example.yaml")
    ExperimentRunner().validate(manifest)
    assert manifest.experiment.name == "humanoid_walk"
    assert manifest.model_extra is not None
    assert manifest.model_extra["algorithm"]["plugin"] == "ppo"
    assert manifest.model_extra["algorithm"]["total_steps"] == 5000000
    assert manifest.model_extra["compute"]["plugin"] == "local"


def test_cli_build_parser_supports_validate_and_plugins() -> None:
    parser = build_parser()

    validate_args = parser.parse_args(["validate", "--manifest", "example.yaml"])
    plugins_args = parser.parse_args(["plugins"])

    assert validate_args.command == "validate"
    assert validate_args.manifest == "example.yaml"
    assert plugins_args.command == "plugins"


def test_manifest_rejects_include_cycles(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("includes: [second.yaml]\nexperiment: {name: test}\n", encoding="utf-8")
    second.write_text("includes: [first.yaml]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="include cycle"):
        load_manifest(first)


@given(total_steps=st.integers(min_value=1, max_value=10_000_000))
def test_manifest_applies_generated_overrides(total_steps: int) -> None:
    with TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        included = tmp_path / "algorithm.yaml"
        included.write_text("algorithm: {plugin: ppo, total_steps: 1}\n", encoding="utf-8")
        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(
            "\n".join(
                [
                    "experiment: {name: test}",
                    "includes: [algorithm.yaml]",
                    f"overrides: {{algorithm.total_steps: {total_steps}}}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        manifest = load_manifest(manifest_path)

    extras = manifest.model_extra or {}
    algorithm = TypeAdapter(dict[str, int | str]).validate_python(extras["algorithm"])
    assert algorithm["total_steps"] == total_steps


def test_manifest_rejects_non_mapping_documents(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text("- not\n- a manifest\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid manifest file"):
        load_manifest(path)

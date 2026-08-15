from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from weir.contracts import Shape, TransitionBatch
from weir.export import export_policy_to_onnx, main, verify_export

OBS_DIM = 4
ACTION_DIM = 2


class FakeAlgorithm:
    """Deterministic AlgorithmPlugin double backed by a seeded MLP policy."""

    def __init__(self) -> None:
        self.observation_shape: Shape | None = None
        self.action_shape: Shape | None = None
        self.config: dict[str, Any] | None = None
        self.policy = nn.Sequential(
            nn.Linear(OBS_DIM, 16),
            nn.Tanh(),
            nn.Linear(16, ACTION_DIM),
        )
        torch.manual_seed(0)
        with torch.no_grad():
            for param in self.policy.parameters():
                param.normal_(std=0.1)

    def configure(
        self, observation_shape: Shape, action_shape: Shape, config: dict[str, Any]
    ) -> None:
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.config = config

    def act(self, observations: Any, deterministic: bool = False) -> Any:
        return np.zeros((1, ACTION_DIM), dtype=np.float32)

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        return {"loss": 0.0}

    def save(self, path: Path) -> None:
        Path(path).write_text("fake-checkpoint", encoding="utf-8")

    def load(self, path: Path) -> None:
        assert Path(path).read_text(encoding="utf-8") == "fake-checkpoint"

    def export_policy(self) -> nn.Module:
        return self.policy


def test_export_policy_output_shape() -> None:
    algorithm = FakeAlgorithm()
    policy = algorithm.export_policy()
    for batch in (1, 4):
        actions = policy(torch.randn(batch, OBS_DIM))
        assert actions.shape == (batch, ACTION_DIM)


def test_export_end_to_end(tmp_path: Path) -> None:
    algorithm = FakeAlgorithm()
    checkpoint = tmp_path / "policy.pt"
    algorithm.save(checkpoint)
    onnx_path = tmp_path / "policy.onnx"

    export_policy_to_onnx(algorithm.policy, onnx_path)

    assert onnx_path.exists()
    assert onnx_path.stat().st_size > 100
    for batch in (1, 5):
        assert verify_export(algorithm.policy, onnx_path, batch=batch)


def test_verification_fails_against_different_policy(tmp_path: Path) -> None:
    algorithm = FakeAlgorithm()
    onnx_path = tmp_path / "policy.onnx"
    export_policy_to_onnx(algorithm.policy, onnx_path)
    assert verify_export(algorithm.policy, onnx_path)

    with torch.no_grad():
        for param in algorithm.policy.parameters():
            param.add_(0.5)

    assert not verify_export(algorithm.policy, onnx_path)


def test_cli_exports_and_verifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "policy.pt"
    algorithm = FakeAlgorithm()
    algorithm.save(checkpoint)
    output = tmp_path / "cli_policy.onnx"
    monkeypatch.setattr("weir.export.create_algorithm", lambda _name: FakeAlgorithm())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weir-export",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--obs-dim",
            str(OBS_DIM),
        ],
    )

    assert main() == 0
    assert output.exists()
    assert verify_export(algorithm.policy, output)


def test_cli_returns_nonzero_on_verification_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "policy.pt"
    FakeAlgorithm().save(checkpoint)
    monkeypatch.setattr("weir.export.create_algorithm", lambda _name: FakeAlgorithm())
    monkeypatch.setattr("weir.export.verify_export", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["weir-export", "--checkpoint", str(checkpoint), "--output", str(tmp_path / "policy.onnx")],
    )

    assert main() == 1


def test_cli_returns_nonzero_when_policy_export_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_text("ppo-checkpoint", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["weir-export", "--checkpoint", str(checkpoint), "--output", str(tmp_path / "policy.onnx")],
    )

    assert main() == 1

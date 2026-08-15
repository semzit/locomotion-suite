from pathlib import Path

import numpy as np
import torch

from weir.algo.ppo import PPOAlgorithm
from weir.core.contracts import AlgorithmPlugin, Shape, SimBackend
from weir.envs.backends.mujoco import MuJoCoSim


def _assert_conformance(sim: SimBackend, algo: AlgorithmPlugin) -> None:
    """Type-check only: forces pyright to verify structural conformance."""
    _ = sim
    _ = algo


def test_implementations_conform_to_protocols() -> None:
    _assert_conformance(MuJoCoSim(), PPOAlgorithm())


def test_mujoco_sim_exposes_protocol_methods() -> None:
    required = {"load", "reset", "step", "observation_shape", "action_shape", "close"}
    assert not required.difference(MuJoCoSim.__dict__)


def test_ppo_algorithm_exposes_protocol_methods() -> None:
    required = {"configure", "learn", "act", "save", "load", "export_policy"}
    assert not required.difference(PPOAlgorithm.__dict__)


def test_shape_defaults() -> None:
    shape = Shape(dims=(4,), dtype="float32")
    assert shape.low is None
    assert shape.high is None


def test_ppo_algorithm_acts_in_shape() -> None:
    algo = PPOAlgorithm()
    algo.configure(
        Shape(dims=(4,), dtype="float32"),
        Shape(dims=(1,), dtype="float32", low=np.array([-1.0]), high=np.array([1.0])),
        {"n_steps": 64, "batch_size": 32},
    )
    action = algo.act(np.zeros(4, dtype=np.float32))
    assert action.shape == (1,)
    assert action.dtype == np.float32


def test_ppo_algorithm_save_load_roundtrip(tmp_path: Path) -> None:
    algo = PPOAlgorithm()
    algo.configure(
        Shape(dims=(4,), dtype="float32"),
        Shape(dims=(1,), dtype="float32", low=np.array([-1.0]), high=np.array([1.0])),
        {"n_steps": 64, "batch_size": 32},
    )
    path = tmp_path / "model.zip"
    algo.save(path)
    assert path.exists()

    loaded = PPOAlgorithm()
    loaded.load(path)
    observation = np.zeros(4, dtype=np.float32)
    assert np.allclose(
        algo.act(observation, deterministic=True),
        loaded.act(observation, deterministic=True),
    )


def test_ppo_algorithm_export_policy(tmp_path: Path) -> None:
    algo = PPOAlgorithm()
    algo.configure(
        Shape(dims=(4,), dtype="float32"),
        Shape(dims=(1,), dtype="float32", low=np.array([-1.0]), high=np.array([1.0])),
        {"n_steps": 64, "batch_size": 32},
    )
    policy = algo.export_policy()
    with torch.no_grad():
        output = policy(torch.zeros(2, 4))
    assert output.shape == (2, 1)

    onnx_path = tmp_path / "policy.onnx"
    torch.onnx.export(policy, (torch.zeros(1, 4),), onnx_path, dynamo=False)
    assert onnx_path.exists()

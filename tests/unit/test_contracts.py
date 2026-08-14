from pathlib import Path

import numpy as np

from weir.algo.ppo import PPOAlgorithm
from weir.contracts import AlgorithmPlugin, Shape, SimBackend, TransitionBatch
from weir.envs.mujoco import MuJoCoSim


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
    required = {"configure", "act", "update", "save", "load", "export_policy"}
    assert not required.difference(PPOAlgorithm.__dict__)


def test_shape_defaults() -> None:
    shape = Shape(dims=(4,), dtype="float32")
    assert shape.low is None
    assert shape.high is None


def test_transition_batch_fields() -> None:
    batch = TransitionBatch([0.0], [0.0], [1.0], [0.0], [False], [False])
    assert batch.rewards == [1.0]
    assert batch.info == {}


def test_ppo_algorithm_acts_in_bounds() -> None:
    low = np.array([-1.0], dtype=np.float32)
    high = np.array([1.0], dtype=np.float32)
    algo = PPOAlgorithm()
    algo.configure(
        Shape(dims=(4,), dtype="float32"),
        Shape(dims=(1,), dtype="float32", low=low, high=high),
        {"lr": 1e-3},
    )
    action = algo.act(np.zeros(4, dtype=np.float32))
    assert action.shape == (1,)
    assert low[0] <= action[0] <= high[0]
    zeros = algo.act(np.zeros(4, dtype=np.float32), deterministic=True)
    assert np.array_equal(zeros, np.zeros(1, dtype=np.float32))


def test_ppo_algorithm_stub_roundtrip(tmp_path: Path) -> None:
    algo = PPOAlgorithm()
    algo.configure(Shape((4,), "float32"), Shape((1,), "float32"), {"lr": 1e-3})
    metrics = algo.update(TransitionBatch([], [], [], [], [], []))
    assert "loss" in metrics
    path = tmp_path / "ckpt.bin"
    algo.save(path)
    assert path.exists()
    algo.load(path)

from pathlib import Path

from weir.algo.ppo import PPOAlgorithm
from weir.contracts import AlgorithmPlugin, SimBackend, TransitionBatch
from weir.envs.mujoco import MuJoCoSim


def _assert_conformance(sim: SimBackend, algo: AlgorithmPlugin) -> None:
    """Type-check only: forces pyright to verify structural conformance."""
    _ = sim
    _ = algo


def test_stub_implementations_conform_to_protocols() -> None:
    _assert_conformance(MuJoCoSim(), PPOAlgorithm())


def test_mujoco_sim_exposes_protocol_methods() -> None:
    required = {"load", "reset", "step", "observation_shape", "action_shape", "close"}
    assert not required.difference(MuJoCoSim.__dict__)


def test_ppo_algorithm_exposes_protocol_methods() -> None:
    required = {"configure", "act", "update", "save", "load", "export_policy"}
    assert not required.difference(PPOAlgorithm.__dict__)


def test_transition_batch_fields() -> None:
    batch = TransitionBatch(
        observations=[0.0],
        actions=[0.0],
        rewards=[1.0],
        next_observations=[0.0],
        terminated=[False],
        truncated=[False],
    )
    assert batch.rewards == [1.0]
    assert batch.info == {}


def test_mujoco_sim_runs_stub_loop() -> None:
    sim = MuJoCoSim()
    sim.load({"name": "humanoid"}, {"dt": 0.02})
    obs = sim.reset(batch_size=1)
    step = sim.step([])
    assert obs["batch_size"] == 1
    assert "reward" in step
    assert "done" in step
    assert sim.observation_shape()["dtype"] == "float32"
    assert sim.action_shape()["shape"] == [0]


def test_ppo_algorithm_runs_stub_roundtrip(tmp_path: Path) -> None:
    algo = PPOAlgorithm()
    algo.configure({"shape": [0]}, {"shape": [0]}, {"lr": 1e-3})
    assert algo.act([]) == []
    metrics = algo.update(TransitionBatch([], [], [], [], [], []))
    assert "loss" in metrics
    path = tmp_path / "ckpt.bin"
    algo.save(path)
    assert path.exists()
    algo.load(path)

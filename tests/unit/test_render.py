from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from weir.cli.render import main as render_main
from weir.cli.render import render_episode
from weir.core.factory import create_algorithm
from weir.envs.backends.mujoco import MuJoCoSim
from weir.envs.utils import MODELS_DIR

CART_POLE = MODELS_DIR / "cartpole.xml"


def make_sim() -> MuJoCoSim:
    sim = MuJoCoSim()
    sim.load(
        {"name": "cartpole", "model": str(CART_POLE)},
        {"task": {"name": "survive", "params": {}}, "time_limit": 5.0},
    )
    return sim


def rendering_available() -> bool:
    sim = make_sim()
    try:
        sim.render_frame(width=8, height=8)
    except RuntimeError:
        return False
    finally:
        sim.close()
    return True


if not rendering_available():
    pytest.skip("offscreen rendering unavailable in this environment", allow_module_level=True)


def test_render_frame_returns_rgb_uint8() -> None:
    sim = make_sim()
    try:
        frame = sim.render_frame(width=160, height=120)
    finally:
        sim.close()
    assert frame.shape == (120, 160, 3)
    assert frame.dtype == np.uint8


def test_render_frame_resizes_renderer() -> None:
    sim = make_sim()
    try:
        small = sim.render_frame(width=64, height=48)
        large = sim.render_frame(width=160, height=120)
    finally:
        sim.close()
    assert small.shape == (48, 64, 3)
    assert large.shape == (120, 160, 3)


def test_render_episode_writes_mp4(tmp_path: Path) -> None:
    sim = make_sim()
    algo = create_algorithm("ppo")
    algo.configure(sim.observation_shape(), sim.action_shape(), {})
    output = tmp_path / "episode.mp4"
    result = render_episode(
        sim,
        algo,
        output,
        frames_to_capture=10,
        width=160,
        height=120,
        fps=15,
    )
    sim.close()
    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0
    assert output.read_bytes()[4:8] == b"ftyp"


def test_render_episode_stops_at_termination(tmp_path: Path) -> None:
    sim = make_sim()
    algo = create_algorithm("ppo")
    algo.configure(sim.observation_shape(), sim.action_shape(), {})
    output = tmp_path / "short.mp4"
    render_episode(
        sim,
        algo,
        output,
        frames_to_capture=1000,
        width=160,
        height=120,
        fps=15,
        seed=0,
    )
    sim.close()
    assert output.exists()


def test_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "cli.mp4"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weir-render",
            "--model",
            str(CART_POLE),
            "--task",
            "survive",
            "--time-limit",
            "5.0",
            "--output",
            str(output),
            "--frames",
            "10",
            "--width",
            "160",
            "--height",
            "120",
            "--fps",
            "15",
        ],
    )
    assert render_main() == 0
    assert output.exists()
    assert output.stat().st_size > 0


def test_cli_renders_legacy_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"fake-checkpoint")
    output = tmp_path / "ckpt.mp4"

    class FakeAlgorithm:
        def load(self, path: Path) -> None:
            assert Path(path).read_bytes() == b"fake-checkpoint"

        def act(self, observation: object, deterministic: bool = False) -> np.ndarray:
            return np.zeros(1, dtype=np.float32)

    monkeypatch.setattr("weir.cli.render.create_algorithm", lambda _name: FakeAlgorithm())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weir-render",
            "--model",
            str(CART_POLE),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--frames",
            "5",
            "--width",
            "160",
            "--height",
            "120",
            "--fps",
            "15",
        ],
    )
    assert render_main() == 0
    assert output.exists()


def test_cli_plays_back_manifest_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSim(MuJoCoSim):
        def reset(self, seed: int | None = None) -> np.ndarray:
            return np.zeros(4, dtype=np.float32)

        def step(self, action: object) -> object:
            step = type(
                "Step",
                (),
                {
                    "observation": np.zeros(4, dtype=np.float32),
                    "terminated": True,
                    "truncated": False,
                },
            )
            return step()

        def render_frame(self, width: int = 0, height: int = 0) -> np.ndarray:
            return np.zeros((height, width, 3), dtype=np.uint8)

        def close(self) -> None: ...

    class FakeAlgorithm:
        def act(self, observation: object, deterministic: bool = False) -> np.ndarray:
            return np.zeros(1, dtype=np.float32)

    class FakeRun:
        config = {"agent": {"name": "x"}}

        def sim(self) -> FakeSim:
            return FakeSim()

        def validate(self, sim: object) -> None: ...

        def algorithm(self) -> FakeAlgorithm:
            return FakeAlgorithm()

        @classmethod
        def open(cls, _checkpoint: object) -> FakeRun:
            return cls()

    monkeypatch.setattr("weir.cli.render.Run", FakeRun)
    output = tmp_path / "playback.mp4"
    monkeypatch.setattr(
        sys,
        "argv",
        ["weir-render", "--checkpoint", str(tmp_path / "checkpoint.zip"), "--output", str(output)],
    )
    assert render_main() == 0
    assert output.exists()


def test_cli_rejects_env_flags_with_manifest_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "checkpoint.meta.json"
    manifest.write_text('{"agent": {"name": "x"}}', encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "weir-render",
            "--checkpoint",
            str(tmp_path / "checkpoint.zip"),
            "--task-param",
            "x=1",
        ],
    )
    assert render_main() == 1


def test_cli_rejects_malformed_task_param(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["weir-render", "--model", str(CART_POLE), "--task-param", "nq"],
    )
    assert render_main() == 1


def test_cli_rejects_unknown_task_param(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["weir-render", "--model", str(CART_POLE), "--task-param", "x=1"],
    )
    assert render_main() == 1

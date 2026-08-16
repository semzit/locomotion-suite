from pathlib import Path

import mujoco
import pytest

from weir.envs.utils import MODELS_DIR


def _model_path(name: str) -> Path:
    path = MODELS_DIR / name
    assert path.exists(), f"missing model asset: {path}"
    return path


@pytest.mark.parametrize(
    "xml_path",
    [
        _model_path("cartpole.xml"),
        _model_path("menagerie/berkeley_humanoid/berkeley_humanoid.xml"),
        _model_path("no_ctrlrange.xml"),
    ],
    ids=["cartpole", "berkeley_humanoid", "no_ctrlrange"],
)
def test_model_loads_and_steps(xml_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert data.time > 0.0
    assert model.nu > 0
    assert model.nq > 0


def test_cartpole_has_single_actuator() -> None:
    model = mujoco.MjModel.from_xml_path(str(_model_path("cartpole.xml")))
    assert model.nu == 1


def test_berkeley_humanoid_morphology() -> None:
    model = mujoco.MjModel.from_xml_path(
        str(_model_path("menagerie/berkeley_humanoid/berkeley_humanoid.xml"))
    )
    assert model.nq == 19
    assert model.nu == 12


def test_berkeley_humanoid_uses_position_actuators() -> None:
    model = mujoco.MjModel.from_xml_path(
        str(_model_path("menagerie/berkeley_humanoid/berkeley_humanoid.xml"))
    )
    for i in range(model.nu):
        assert model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT

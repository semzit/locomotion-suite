from pathlib import Path

import mujoco
import pytest

MODELS_DIR = Path(__file__).parents[2] / "weir" / "models"


def _model_path(name: str) -> Path:
    path = MODELS_DIR / name
    assert path.exists(), f"missing model asset: {path}"
    return path


@pytest.mark.parametrize(
    "xml_path",
    [
        _model_path("cartpole.xml"),
        _model_path("simple_humanoid.xml"),
        _model_path("menagerie/berkeley_humanoid/berkeley_humanoid.xml"),
    ],
    ids=["cartpole", "simple_humanoid", "berkeley_humanoid"],
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


def test_simple_humanoid_has_two_legs() -> None:
    model = mujoco.MjModel.from_xml_path(str(_model_path("simple_humanoid.xml")))
    actuator_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
    }
    assert {"hip_l", "knee_l", "ankle_l", "hip_r", "knee_r", "ankle_r"} <= actuator_names


def test_berkeley_humanoid_uses_position_actuators() -> None:
    model = mujoco.MjModel.from_xml_path(
        str(_model_path("menagerie/berkeley_humanoid/berkeley_humanoid.xml"))
    )
    for i in range(model.nu):
        assert model.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT

import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from weir.algo.utils import sample_action
from weir.contracts import Shape
from weir.envs.utils import MODELS_DIR, resolve_model_asset
from weir.utils import CONFIG_DIR, ROOT, config_to_dict, log_event, resolve_model_path


def test_root_aligns_with_repo() -> None:
    assert (ROOT / "weir" / "contracts.py").exists()


def test_subfolder_paths_exist() -> None:
    assert CONFIG_DIR.is_dir()
    assert (CONFIG_DIR / "train.yaml").exists()
    assert MODELS_DIR.is_dir()
    assert (MODELS_DIR / "cartpole.xml").exists()


def test_resolve_model_path_absolute_stays() -> None:
    absolute = str(MODELS_DIR / "cartpole.xml")
    assert resolve_model_path(absolute) == absolute


def test_resolve_model_path_relative_joins_root() -> None:
    resolved = Path(resolve_model_path("weir/models/cartpole.xml"))
    assert resolved.is_absolute()
    assert resolved == (ROOT / "weir" / "models" / "cartpole.xml").resolve()


def test_resolve_model_asset_joins_models_dir() -> None:
    resolved = Path(resolve_model_asset("cartpole.xml"))
    assert resolved == (MODELS_DIR / "cartpole.xml").resolve()


def test_resolve_model_asset_nested_path() -> None:
    resolved = Path(resolve_model_asset("menagerie/berkeley_humanoid/berkeley_humanoid.xml"))
    expected = MODELS_DIR / "menagerie" / "berkeley_humanoid" / "berkeley_humanoid.xml"
    assert resolved == expected.resolve()


def test_config_to_dict_resolves_section() -> None:
    cfg = DictConfig({"plugin": "mujoco", "time_limit": 10.0})
    result = config_to_dict(cfg)
    assert result == {"plugin": "mujoco", "time_limit": 10.0}


def test_log_event_attaches_fields(caplog) -> None:
    logger = logging.getLogger("weir.utils.test")
    with caplog.at_level("INFO", logger="weir.utils.test"):
        log_event(logger, "test.event", sim="mujoco", steps=5)

    assert len(caplog.records) == 1
    assert caplog.records[0].message == "test.event"
    assert caplog.records[0].sim == "mujoco"
    assert caplog.records[0].steps == 5


def test_sample_action_in_bounds() -> None:
    low = np.array([-1.0, 0.0], dtype=np.float32)
    high = np.array([1.0, 2.0], dtype=np.float32)
    shape = Shape(dims=(2,), dtype="float32", low=low, high=high)
    rng = np.random.default_rng(0)

    action = sample_action(shape, rng, deterministic=False)
    assert action.shape == (2,)
    assert np.all(action >= low)
    assert np.all(action <= high)

    midpoint = sample_action(shape, rng, deterministic=True)
    assert np.allclose(midpoint, (low + high) / 2.0)


def test_sample_action_without_bounds() -> None:
    shape = Shape(dims=(1,), dtype="float32")
    rng = np.random.default_rng(0)

    action = sample_action(shape, rng)
    assert action.shape == (1,)
    assert -1.0 <= action[0] <= 1.0

    zero = sample_action(shape, rng, deterministic=True)
    assert np.array_equal(zero, np.zeros(1, dtype=np.float32))

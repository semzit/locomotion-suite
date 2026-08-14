import logging
from pathlib import Path

from omegaconf import DictConfig

from weir.utils import ROOT, config_to_dict, log_event, resolve_model_path


def test_rootaligns_with_repo() -> None:
    assert (ROOT / "weir" / "contracts.py").exists()


def test_resolve_model_path_absolute_stays() -> None:
    absolute = str(ROOT / "weir" / "models" / "cartpole.xml")
    assert resolve_model_path(absolute) == absolute


def test_resolve_model_path_relative_joins_root() -> None:
    resolved = Path(resolve_model_path("weir/models/cartpole.xml"))
    assert resolved.is_absolute()
    assert resolved == (ROOT / "weir" / "models" / "cartpole.xml").resolve()


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

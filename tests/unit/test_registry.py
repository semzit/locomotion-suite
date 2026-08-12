import pytest
from beartype.roar import BeartypeCallHintParamViolation

from core.registry import PluginRegistry


def test_registry_creates_registered_plugin() -> None:
    registry = PluginRegistry()
    registry.register("example", lambda value: {"value": value})
    assert registry.create("example", value=1) == {"value": 1}


def test_registry_rejects_duplicate_name() -> None:
    registry = PluginRegistry()
    registry.register("example", object)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("example", object)


def test_registry_rejects_invalid_plugin_name_type() -> None:
    with pytest.raises(BeartypeCallHintParamViolation):
        PluginRegistry().register(1, object)  # type: ignore[arg-type]

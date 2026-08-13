import pytest
from beartype.roar import BeartypeCallHintParamViolation

from core.registry import PLUGIN_ENTRY_POINT_GROUP, PluginDiscoveryError, PluginRegistry


class FakeEntryPoint:
    def __init__(self, name: str, value: str, factory: object) -> None:
        self.name = name
        self.value = value
        self._factory = factory

    def load(self) -> object:
        return self._factory


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


def test_registry_discovers_entry_point_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [
        FakeEntryPoint("second", "plugins.second:factory", lambda: "second"),
        FakeEntryPoint("first", "plugins.first:factory", lambda: "first"),
    ]

    def entry_points(*, group: str) -> list[FakeEntryPoint]:
        assert group == PLUGIN_ENTRY_POINT_GROUP
        return entries

    monkeypatch.setattr("core.registry.metadata.entry_points", entry_points)
    registry = PluginRegistry()

    assert registry.discover() == ("first", "second")
    assert registry.create("first") == "first"


def test_registry_reports_unloadable_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = FakeEntryPoint("broken", "plugins.broken:factory", object())
    monkeypatch.setattr(
        "core.registry.metadata.entry_points",
        lambda *, group: [entry],  # noqa: ARG005
    )

    with pytest.raises(PluginDiscoveryError, match="not callable"):
        PluginRegistry().discover()

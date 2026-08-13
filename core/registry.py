from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
from typing import cast

from beartype import beartype

from plugins.algorithms.ppo import PPOAlgorithm
from plugins.compute.local import LocalComputeBackend
from plugins.sim.mujoco import MuJoCoSim

from .observability import logger, tracer
from .utils import PLUGIN_ENTRY_POINT_GROUP

PluginFactory = Callable[..., object]


class PluginDiscoveryError(RuntimeError):
    """Raised when an installed plugin cannot be loaded safely."""


class PluginRegistry:
    """Registry of plugin factories, including installed package entry points."""

    def __init__(self) -> None:
        self._factories: dict[str, PluginFactory] = {}
        for name, factory in {
            "local": LocalComputeBackend,
            "mujoco": MuJoCoSim,
            "ppo": PPOAlgorithm,
        }.items():
            self.register(name, cast(PluginFactory, factory))

    @beartype
    def register(self, name: str, factory: PluginFactory) -> None:
        if name in self._factories:
            raise ValueError(f"Plugin already registered: {name}")
        self._factories[name] = factory

    def discover(self, group: str = PLUGIN_ENTRY_POINT_GROUP) -> tuple[str, ...]:
        """Load factories advertised by installed distributions in an entry-point group."""
        with tracer.start_as_current_span("plugins.discover") as span:
            span.set_attribute("plugin.group", group)
            entry_points = metadata.entry_points(group=group)
            for entry_point in sorted(entry_points, key=lambda item: item.name):
                try:
                    factory = entry_point.load()
                except Exception as error:
                    raise PluginDiscoveryError(
                        f"Could not load plugin {entry_point.name!r} from {entry_point.value!r}"
                    ) from error
                if not callable(factory):
                    raise PluginDiscoveryError(
                        f"Plugin {entry_point.name!r} from {entry_point.value!r} is not callable"
                    )
                try:
                    self.register(entry_point.name, cast(PluginFactory, factory))
                except ValueError as error:
                    raise PluginDiscoveryError(
                        f"Duplicate plugin name discovered: {entry_point.name!r}"
                    ) from error

        names = self.names()
        logger.info("plugins.discovered", group=group, plugin_count=len(names), plugins=names)
        return names

    def create(self, name: str, **kwargs: object) -> object:
        try:
            return self._factories[name](**kwargs)
        except KeyError as error:
            raise KeyError(f"Unknown plugin: {name}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

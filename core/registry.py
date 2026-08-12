from __future__ import annotations

from collections.abc import Callable
from typing import Any


class PluginRegistry:
    """In-process registry; entry-point discovery is a later Phase 0 increment."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        if name in self._factories:
            raise ValueError(f"Plugin already registered: {name}")
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> Any:
        try:
            return self._factories[name](**kwargs)
        except KeyError as error:
            raise KeyError(f"Unknown plugin: {name}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

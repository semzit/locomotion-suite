from __future__ import annotations

from typing import Any


class LocalComputeBackend:
    """Reference local compute backend used for the minimal stack."""

    def launch(self, job: Any) -> str:
        return f"local-job:{job}"

    def status(self, handle: Any) -> str:
        return "running" if handle else "idle"

    def cancel(self, handle: Any) -> None:
        _ = handle

    def artifacts(self, handle: Any) -> list[str]:
        return [str(handle)] if handle else []

from __future__ import annotations

import numpy as np

from weir.contracts import Shape


def sample_action(
    action_shape: Shape,
    rng: np.random.Generator,
    deterministic: bool = False,
) -> np.ndarray:
    """Sample an in-bounds action, or its midpoint when deterministic."""
    dims = tuple(action_shape.dims)
    low = action_shape.low
    high = action_shape.high
    if low is not None and high is not None:
        if deterministic:
            return ((low + high) / 2.0).astype(np.float32)
        return rng.uniform(low, high, size=dims).astype(np.float32)
    if deterministic:
        return np.zeros(dims, dtype=np.float32)
    return rng.uniform(-1.0, 1.0, size=dims).astype(np.float32)

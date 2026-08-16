from __future__ import annotations

import numpy as np


def rotate_vector(quat: np.ndarray, vector: tuple[float, float, float]) -> np.ndarray:
    """
    Rotate a 3-vector by a unit quaternion in (w, x, y, z) order.
    """
    w, x, y, z = (float(v) for v in quat[0:4])
    norm = float(np.linalg.norm(quat[0:4]))
    w /= norm
    x /= norm
    y /= norm
    z /= norm
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.asarray(
        [
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ],
        dtype=np.float32,
    )

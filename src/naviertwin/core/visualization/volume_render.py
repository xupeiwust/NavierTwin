"""Volume ray-marcher (CPU) — front-to-back compositing.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.visualization.volume_render import ray_march
    >>> vol = np.zeros((20, 20, 20))
    >>> vol[8:12, 8:12, 8:12] = 1.0
    >>> img = ray_march(vol, n_steps=20, axis=2)
    >>> img.shape
    (20, 20)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ray_march(
    volume: NDArray[np.float64],
    *,
    n_steps: int = 32, axis: int = 2,
    alpha: float = 0.1,
) -> NDArray[np.float64]:
    """Front-to-back along `axis`. Returns 2D projection."""
    V = np.moveaxis(np.asarray(volume), axis, -1)  # (H, W, D)
    H, W, D = V.shape
    img = np.zeros((H, W))
    trans = np.ones((H, W))
    step = max(D // n_steps, 1)
    for k in range(0, D, step):
        sample = V[:, :, k]
        a = alpha * sample
        img += trans * a * sample
        trans *= (1 - a)
    return img


__all__ = ["ray_march"]

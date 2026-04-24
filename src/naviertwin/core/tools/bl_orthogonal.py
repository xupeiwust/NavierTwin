"""Boundary layer orthogonal grid — wall normal extrusion + geometric stretching.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.tools.bl_orthogonal import bl_grid
    >>> wall = np.array([[0., 0], [1., 0], [2., 0]])
    >>> normals = np.tile(np.array([0., 1.]), (3, 1))
    >>> grid = bl_grid(wall, normals, n_layers=5, first=0.01, growth=1.2)
    >>> grid.shape
    (3, 5, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bl_grid(
    wall_pts: NDArray[np.float64],
    wall_normals: NDArray[np.float64],
    *,
    n_layers: int = 10,
    first: float = 1e-3,
    growth: float = 1.2,
) -> NDArray[np.float64]:
    """기하급수 두께로 wall-normal extrusion. 반환: (M, n_layers, dim)."""
    wp = np.asarray(wall_pts, dtype=np.float64)
    wn = np.asarray(wall_normals, dtype=np.float64)
    # cumulative thickness
    h = np.array([first * growth ** k for k in range(n_layers)])
    y = np.cumsum(h)
    # broadcast
    grid = wp[:, None, :] + wn[:, None, :] * y[None, :, None]
    return grid


__all__ = ["bl_grid"]

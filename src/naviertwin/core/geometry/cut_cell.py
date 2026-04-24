"""Cut-cell Cartesian — fluid volume fraction from SDF (1D/2D).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.geometry.cut_cell import cut_cell_fraction_2d
    >>> phi = np.array([[-1, -1, 1], [-1, 0.5, 1], [1, 1, 1]], dtype=float)
    >>> frac = cut_cell_fraction_2d(phi)
    >>> frac.shape
    (2, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cut_cell_fraction_2d(phi: NDArray[np.float64]) -> NDArray[np.float64]:
    """For each cell (i, j), interpolate from corner phi values.

    fraction = mean(corners < 0).  Approximate.
    """
    phi = np.asarray(phi, dtype=np.float64)
    nx, ny = phi.shape
    f = np.zeros((nx - 1, ny - 1))
    for i in range(nx - 1):
        for j in range(ny - 1):
            corners = phi[i:i + 2, j:j + 2].ravel()
            f[i, j] = (corners < 0).mean()
    return f


__all__ = ["cut_cell_fraction_2d"]

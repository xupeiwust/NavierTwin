"""PIC (particle-in-cell) deposition — linear (CIC) charge to grid 1D.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.meshless.pic_deposit import deposit_cic_1d
    >>> x = np.array([0.5, 1.7])
    >>> rho = deposit_cic_1d(x, np.array([1.0, 1.0]), n_grid=4, dx=1.0)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def deposit_cic_1d(
    x: NDArray[np.float64],
    weights: NDArray[np.float64],
    *,
    n_grid: int,
    dx: float = 1.0,
    x0: float = 0.0,
) -> NDArray[np.float64]:
    """Cloud-in-cell linear deposition."""
    rho = np.zeros(n_grid)
    pos = (np.asarray(x) - x0) / dx
    i = np.floor(pos).astype(int)
    f = pos - i
    for k in range(len(x)):
        if 0 <= i[k] < n_grid:
            rho[i[k]] += float(weights[k]) * (1 - f[k])
        if 0 <= i[k] + 1 < n_grid:
            rho[i[k] + 1] += float(weights[k]) * f[k]
    return rho


__all__ = ["deposit_cic_1d"]

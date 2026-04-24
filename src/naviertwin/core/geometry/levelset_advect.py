"""Level-set advection — φ_t + u·∇φ = 0, upwind 1D.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.geometry.levelset_advect import advect_step
    >>> phi = np.linspace(-1, 1, 11)
    >>> u = np.ones(11)
    >>> phi2 = advect_step(phi, u, dt=0.05, dx=0.2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def advect_step(
    phi: NDArray[np.float64],
    u: NDArray[np.float64],
    *,
    dt: float, dx: float,
) -> NDArray[np.float64]:
    phi = np.asarray(phi, dtype=np.float64).copy()
    u = np.asarray(u, dtype=np.float64)
    n = len(phi)
    new = phi.copy()
    for i in range(1, n - 1):
        if u[i] >= 0:
            new[i] = phi[i] - dt / dx * u[i] * (phi[i] - phi[i - 1])
        else:
            new[i] = phi[i] - dt / dx * u[i] * (phi[i + 1] - phi[i])
    return new


__all__ = ["advect_step"]

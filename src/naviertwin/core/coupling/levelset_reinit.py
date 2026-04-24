"""Level-set reinitialization (1D, Sussman-Smereka 1994).

φ_τ + sign(φ_0)(|∇φ| - 1) = 0  → 부호 거리 함수.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.coupling.levelset_reinit import reinit_1d
    >>> phi = np.array([2., 1., 0.5, -0.5, -1., -2.])
    >>> phi2 = reinit_1d(phi, dx=1.0, n_iter=20)
    >>> phi2.shape
    (6,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def reinit_1d(
    phi: NDArray[np.float64], *, dx: float = 1.0, n_iter: int = 30,
) -> NDArray[np.float64]:
    p = np.asarray(phi, dtype=np.float64).copy()
    p0 = p.copy()
    s0 = np.sign(p0)
    dt = 0.3 * dx
    for _ in range(n_iter):
        # |grad phi| via central, with godunov-like sign-aware:
        dxp = (np.roll(p, -1) - p) / dx
        dxm = (p - np.roll(p, 1)) / dx
        dxp[-1] = 0
        dxm[0] = 0
        gp = np.where(s0 > 0,
                      np.maximum(np.maximum(dxm, 0) ** 2, np.minimum(dxp, 0) ** 2),
                      np.maximum(np.maximum(dxp, 0) ** 2, np.minimum(dxm, 0) ** 2))
        grad = np.sqrt(gp)
        p = p - dt * s0 * (grad - 1.0)
    return p


__all__ = ["reinit_1d"]

"""Lattice-Boltzmann D2Q9 — BGK collision + streaming (single step).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.meshless.lbm_d2q9 import lbm_step
    >>> f = np.ones((9, 5, 5)) * (1.0 / 9.0)
    >>> f2 = lbm_step(f, omega=1.0)
    >>> f2.shape
    (9, 5, 5)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# D2Q9 velocities and weights
_E = np.array([
    [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
    [1, 1], [-1, 1], [-1, -1], [1, -1],
])
_W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)


def equilibrium(rho: NDArray, u: NDArray) -> NDArray:
    """rho: (X,Y); u: (2,X,Y) → f_eq: (9,X,Y)."""
    feq = np.zeros((9, *rho.shape))
    u_sq = u[0] ** 2 + u[1] ** 2
    for k in range(9):
        eu = _E[k, 0] * u[0] + _E[k, 1] * u[1]
        feq[k] = _W[k] * rho * (1 + 3 * eu + 4.5 * eu ** 2 - 1.5 * u_sq)
    return feq


def lbm_step(f: NDArray, *, omega: float = 1.0) -> NDArray:
    """BGK collision + streaming (periodic)."""
    rho = f.sum(axis=0)
    u = np.zeros((2, *rho.shape))
    for k in range(9):
        u[0] += _E[k, 0] * f[k]
        u[1] += _E[k, 1] * f[k]
    u /= np.maximum(rho, 1e-30)
    feq = equilibrium(rho, u)
    f = f - omega * (f - feq)
    # streaming
    f_new = np.zeros_like(f)
    for k in range(9):
        f_new[k] = np.roll(f[k], shift=(_E[k, 0], _E[k, 1]), axis=(0, 1))
    return f_new


__all__ = ["equilibrium", "lbm_step"]

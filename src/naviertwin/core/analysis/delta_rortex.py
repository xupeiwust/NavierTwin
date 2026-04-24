"""Δ-criterion (Chong 1990) and Rortex (Liu 2018).

Δ = (Q/3)³ + (R/2)², Δ > 0 → 복소 conjugate eigenvalues → vortex.
Rortex = 2 |λ_ci| (대략 swirl strength 의 2배).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.delta_rortex import delta_criterion
    >>> grad = np.zeros((1, 3, 3))
    >>> grad[0] = [[0, -1, 0], [1, 0, 0], [0, 0, 0]]
    >>> delta_criterion(grad)[0] > 0
    np.True_
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _q_r(grad: NDArray[np.float64]) -> tuple[NDArray, NDArray]:
    """invariants: Q = -(1/2) tr(A²), R = -det(A) (incompressible: tr(A) = 0)."""
    g = np.asarray(grad, dtype=np.float64)
    A2 = np.einsum("...ij,...jk->...ik", g, g)
    Q = -0.5 * np.einsum("...ii->...", A2)
    R = -np.linalg.det(g)
    return Q, R


def delta_criterion(grad_u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Δ = (Q/3)³ + (R/2)²."""
    Q, R = _q_r(grad_u)
    return (Q / 3.0) ** 3 + (R / 2.0) ** 2


def rortex_field(grad_u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rortex magnitude ≈ 2 λ_ci."""
    g = np.asarray(grad_u, dtype=np.float64)
    flat = g.reshape(-1, 3, 3)
    out = np.zeros(flat.shape[0])
    for i in range(flat.shape[0]):
        ev = np.linalg.eigvals(flat[i])
        out[i] = 2.0 * float(np.max(np.abs(ev.imag)))
    return out.reshape(g.shape[:-2])


__all__ = ["delta_criterion", "rortex_field"]

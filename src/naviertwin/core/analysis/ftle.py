"""FTLE (Finite-Time Lyapunov Exponent) — 2D Lagrangian coherent structures 근사.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.ftle import compute_ftle_2d
    >>> def vf(p): return np.array([-p[1], p[0]])  # rotation
    >>> ftle = compute_ftle_2d(vf, x=np.linspace(-1,1,10), y=np.linspace(-1,1,10), T=1.0, dt=0.01)
    >>> ftle.shape
    (10, 10)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def _advect(
    vf: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    pts: NDArray[np.float64], T: float, dt: float,
) -> NDArray[np.float64]:
    """RK4 로 pts (N, 2) 를 [0, T] 에 대해 advection."""
    n = int(np.ceil(T / dt))
    dt = T / n
    P = pts.copy()
    for _ in range(n):
        k1 = np.array([vf(p) for p in P])
        k2 = np.array([vf(p) for p in (P + 0.5 * dt * k1)])
        k3 = np.array([vf(p) for p in (P + 0.5 * dt * k2)])
        k4 = np.array([vf(p) for p in (P + dt * k3)])
        P = P + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return P


def compute_ftle_2d(
    vf: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x: NDArray[np.float64], y: NDArray[np.float64],
    T: float = 1.0, dt: float = 0.01,
    *, eps: float = 1e-4,
) -> NDArray[np.float64]:
    """grid 각 점에서 FTLE = (1/(2T)) log λ_max(Cauchy-Green)."""
    nx = x.size
    ny = y.size
    FT = np.zeros((ny, nx))
    X, Y = np.meshgrid(x, y, indexing="xy")
    for i in range(ny):
        for j in range(nx):
            p = np.array([X[i, j], Y[i, j]])
            stencil = np.array([
                p + [eps, 0],
                p - [eps, 0],
                p + [0, eps],
                p - [0, eps],
            ])
            adv = _advect(vf, stencil, T, dt)
            dxdX = (adv[0] - adv[1]) / (2 * eps)
            dxdY = (adv[2] - adv[3]) / (2 * eps)
            F = np.stack([dxdX, dxdY], axis=1)  # (2, 2)
            C = F.T @ F
            lam = np.linalg.eigvalsh(C).max()
            FT[i, j] = 0.5 / abs(T) * np.log(max(lam, 1e-30))
    return FT


__all__ = ["compute_ftle_2d"]

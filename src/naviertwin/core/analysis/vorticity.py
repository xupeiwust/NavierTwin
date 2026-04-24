"""격자 속도장의 vorticity / Q-criterion / λ₂ 계산.

uniform grid 에 한함. 비균일 격자는 PyVista compute_derivative 사용 권장.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.vorticity import vorticity_2d
    >>> u = np.zeros((8, 8)); v = np.zeros((8, 8))
    >>> w = vorticity_2d(u, v, dx=1.0, dy=1.0)
    >>> w.shape
    (8, 8)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def vorticity_2d(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    dx: float = 1.0,
    dy: float = 1.0,
) -> NDArray[np.float64]:
    """ωz = ∂v/∂x − ∂u/∂y. 배열 shape: (ny, nx)."""
    dv_dx = np.gradient(v, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    return dv_dx - du_dy


def q_criterion_2d(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    dx: float = 1.0,
    dy: float = 1.0,
) -> NDArray[np.float64]:
    """Q = -½(S²-Ω²) = ½(|Ω|²-|S|²) 2D 근사."""
    du_dx = np.gradient(u, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    # Symmetric S and antisymmetric Ω
    S11 = du_dx
    S22 = dv_dy
    S12 = 0.5 * (du_dy + dv_dx)
    O12 = 0.5 * (dv_dx - du_dy)  # single independent component
    s2 = S11 ** 2 + S22 ** 2 + 2 * S12 ** 2
    o2 = 2 * O12 ** 2
    return 0.5 * (o2 - s2)


def vorticity_3d(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    w: NDArray[np.float64],
    dx: float = 1.0, dy: float = 1.0, dz: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """(ωx, ωy, ωz). 배열 shape: (nz, ny, nx)."""
    du_dy = np.gradient(u, dy, axis=1)
    du_dz = np.gradient(u, dz, axis=0)
    dv_dx = np.gradient(v, dx, axis=2)
    dv_dz = np.gradient(v, dz, axis=0)
    dw_dx = np.gradient(w, dx, axis=2)
    dw_dy = np.gradient(w, dy, axis=1)
    wx = dw_dy - dv_dz
    wy = du_dz - dw_dx
    wz = dv_dx - du_dy
    return wx, wy, wz


__all__ = ["vorticity_2d", "q_criterion_2d", "vorticity_3d"]

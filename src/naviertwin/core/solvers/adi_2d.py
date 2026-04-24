"""ADI (Alternating Direction Implicit) — 2D heat equation.

u_t = α (u_xx + u_yy), Dirichlet 0.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.solvers.adi_2d import adi_step
    >>> u = np.zeros((9, 9)); u[4, 4] = 1.0
    >>> u1 = adi_step(u, dt=0.01, dx=0.1, dy=0.1, alpha=1.0)
    >>> u1.shape
    (9, 9)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _solve_tridiag(a: NDArray, b: NDArray, c: NDArray, d: NDArray) -> NDArray:
    """Thomas algorithm; a sub, b diag, c sup, d rhs (vectorized over rows: shape (m, n))."""
    n = b.shape[-1]
    cp = c.copy().astype(np.float64)
    dp = d.copy().astype(np.float64)
    bp = b.copy().astype(np.float64)
    cp[..., 0] = c[..., 0] / bp[..., 0]
    dp[..., 0] = d[..., 0] / bp[..., 0]
    for i in range(1, n):
        m = bp[..., i] - a[..., i] * cp[..., i - 1]
        cp[..., i] = c[..., i] / m
        dp[..., i] = (d[..., i] - a[..., i] * dp[..., i - 1]) / m
    x = np.zeros_like(dp)
    x[..., -1] = dp[..., -1]
    for i in range(n - 2, -1, -1):
        x[..., i] = dp[..., i] - cp[..., i] * x[..., i + 1]
    return x


def adi_step(
    u: NDArray[np.float64], *, dt: float, dx: float, dy: float, alpha: float = 1.0,
) -> NDArray[np.float64]:
    """Peaceman-Rachford ADI 한 스텝 (Dirichlet 0)."""
    u = np.asarray(u, dtype=np.float64).copy()
    rx = alpha * dt / (2.0 * dx * dx)
    ry = alpha * dt / (2.0 * dy * dy)
    nx, ny = u.shape

    # Step 1: implicit x, explicit y. solve per interior row j=1..ny-2.
    inner = u[1:-1, 1:-1]  # shape (nx-2, ny-2)
    rhs1 = inner + ry * (u[1:-1, 2:] - 2 * inner + u[1:-1, :-2])
    # tridiag in x for each j (column-wise solve along axis 0)
    m = nx - 2
    a = np.full((ny - 2, m), -rx)
    b = np.full((ny - 2, m), 1.0 + 2 * rx)
    c = np.full((ny - 2, m), -rx)
    a[..., 0] = 0.0
    c[..., -1] = 0.0
    d = rhs1.T  # (ny-2, nx-2)
    u_half_inner = _solve_tridiag(a, b, c, d).T  # (nx-2, ny-2)
    u_half = np.zeros_like(u)
    u_half[1:-1, 1:-1] = u_half_inner

    # Step 2: implicit y, explicit x.
    rhs2 = u_half_inner + rx * (
        u_half[2:, 1:-1] - 2 * u_half_inner + u_half[:-2, 1:-1]
    )
    n2 = ny - 2
    a2 = np.full((nx - 2, n2), -ry)
    b2 = np.full((nx - 2, n2), 1.0 + 2 * ry)
    c2 = np.full((nx - 2, n2), -ry)
    a2[..., 0] = 0.0
    c2[..., -1] = 0.0
    u_new_inner = _solve_tridiag(a2, b2, c2, rhs2)
    u_new = np.zeros_like(u)
    u_new[1:-1, 1:-1] = u_new_inner
    return u_new


__all__ = ["adi_step"]

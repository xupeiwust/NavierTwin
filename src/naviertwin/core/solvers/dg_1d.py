"""1D DG — 리스트 of 셀별 계수 + local Lagrange nodes (GLL).

Advection ∂u/∂t + c ∂u/∂x = 0 (스칼라 상수 c, upwind flux).

Examples:
    >>> from naviertwin.core.solvers.dg_1d import solve_advection_1d_dg
    >>> import numpy as np
    >>> x, t, U = solve_advection_1d_dg(n_cells=8, T=0.2)
    >>> U.shape[0] > 0
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def _lobatto_nodes(p: int) -> NDArray[np.float64]:
    """Gauss-Lobatto-Legendre nodes on [-1, 1]."""
    if p == 1:
        return np.array([-1.0, 1.0])
    from numpy.polynomial.legendre import legder, legroots
    poly = np.zeros(p + 1)
    poly[-1] = 1.0
    dpoly = legder(poly)
    inner = legroots(dpoly)
    return np.sort(np.r_[-1.0, inner, 1.0])


def solve_advection_1d_dg(
    n_cells: int = 16, L: float = 1.0, T: float = 0.2,
    c: float = 1.0,
    u0: Callable | None = None,
    *, p: int = 2, cfl: float = 0.3,
) -> tuple[NDArray, NDArray, NDArray]:
    """Pk DG advection (periodic). Returns (x_all, t, U[n_nodes, n_steps+1])."""
    xi = _lobatto_nodes(p)
    n_local = xi.size
    h = L / n_cells
    # cell centers + local positions
    x_all = np.zeros(n_cells * n_local)
    for j in range(n_cells):
        xc = (j + 0.5) * h
        x_all[j * n_local:(j + 1) * n_local] = xc + 0.5 * h * xi
    # mass/stiffness per cell (reference)
    # Use Lagrange basis at GLL points; mass matrix = diag(w) (lumped)
    # weights via Gauss-Lobatto formulae (trapezoidal-like)
    # 간단화: lumped mass → fixed GL weights approximation
    from numpy.polynomial.legendre import legval
    # compute lumped weights: w_i = 2 / (p(p+1) [P_p(xi_i)]²)
    Pp = np.zeros(p + 1)
    Pp[-1] = 1.0
    w = 2.0 / (p * (p + 1) * legval(xi, Pp) ** 2)

    # differentiation matrix D_ref
    # Lagrange derivative at GLL nodes (Trefethen-style)
    def lagrange_D(x):
        n = x.size
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                num = 1.0
                den = 1.0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    num *= (x[i] - x[k])
                for k in range(n):
                    if k == j:
                        continue
                    den *= (x[j] - x[k])
                D[i, j] = num / den
            D[i, i] = -np.sum([D[i, j] for j in range(n) if j != i])
        return D
    D_ref = lagrange_D(xi)
    # physical D: D_phys = (2/h) D_ref (chain rule)
    D = (2.0 / h) * D_ref

    # init u
    if u0 is None:
        def u0(x):
            return np.sin(2 * np.pi * x / L)
    u = u0(x_all).reshape(n_cells, n_local)

    dt = cfl * h / abs(c) / (2 * p + 1)
    n_steps = int(np.ceil(T / dt))
    dt = T / n_steps
    U = np.zeros((n_cells * n_local, n_steps + 1))
    U[:, 0] = u.reshape(-1)
    t = np.zeros(n_steps + 1)

    for k in range(n_steps):
        # upwind numerical flux at cell boundaries
        u_left = u[:, 0]   # left face of each cell
        u_right = u[:, -1]  # right face of each cell
        # upwind: if c > 0, flux = c * u_right_prev (left neighbor)
        if c > 0:
            flux_left = c * np.roll(u_right, 1)   # from left cell
            flux_right = c * u_right
        else:
            flux_left = c * u_left
            flux_right = c * np.roll(u_left, -1)
        # interior contribution: -c D u (local)
        du = np.zeros_like(u)
        for j in range(n_cells):
            local = u[j]
            du[j] = -c * (D @ local)
            # subtract/add surface integrals
            # boundary terms divided by w_i to account for lumped mass
            du[j, 0] -= (flux_left[j] - c * local[0]) * (2.0 / h) / w[0]
            du[j, -1] += (flux_right[j] - c * local[-1]) * (2.0 / h) / w[-1]
        u = u + dt * du
        U[:, k + 1] = u.reshape(-1)
        t[k + 1] = (k + 1) * dt
    return x_all, t, U


__all__ = ["solve_advection_1d_dg"]

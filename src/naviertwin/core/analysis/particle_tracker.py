"""Lagrangian particle tracker — 속도장에서 RK4 로 입자 궤적 적분.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.particle_tracker import track_particles_2d
    >>> u_field = np.ones((10, 10))  # 균일 속도
    >>> v_field = np.zeros((10, 10))
    >>> seeds = np.array([[0.5, 0.5]])
    >>> trails = track_particles_2d(u_field, v_field, seeds, Lx=1.0, Ly=1.0, dt=0.01, n_steps=10)
    >>> trails.shape
    (1, 11, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _bilinear_sample(
    field: NDArray[np.float64], x: float, y: float, Lx: float, Ly: float,
) -> float:
    ny, nx = field.shape
    # normalize to [0, nx-1] and [0, ny-1]
    fx = np.clip(x / Lx * (nx - 1), 0, nx - 1 - 1e-12)
    fy = np.clip(y / Ly * (ny - 1), 0, ny - 1 - 1e-12)
    ix = int(np.floor(fx)); iy = int(np.floor(fy))  # noqa: E702
    tx = fx - ix; ty = fy - iy  # noqa: E702
    v00 = field[iy, ix]
    v10 = field[iy, ix + 1]
    v01 = field[iy + 1, ix]
    v11 = field[iy + 1, ix + 1]
    return float(
        (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10
        + (1 - tx) * ty * v01 + tx * ty * v11
    )


def track_particles_2d(
    u: NDArray[np.float64], v: NDArray[np.float64],
    seeds: NDArray[np.float64],
    *, Lx: float = 1.0, Ly: float = 1.0,
    dt: float = 0.01, n_steps: int = 100,
) -> NDArray[np.float64]:
    """(n_seeds, n_steps+1, 2) 궤적."""
    seeds = np.asarray(seeds, dtype=np.float64)
    n = seeds.shape[0]
    trails = np.zeros((n, n_steps + 1, 2))
    trails[:, 0, :] = seeds

    def vel(p):
        return np.array([
            _bilinear_sample(u, p[0], p[1], Lx, Ly),
            _bilinear_sample(v, p[0], p[1], Lx, Ly),
        ])

    for i in range(n):
        p = seeds[i].copy()
        for k in range(n_steps):
            k1 = vel(p)
            k2 = vel(p + 0.5 * dt * k1)
            k3 = vel(p + 0.5 * dt * k2)
            k4 = vel(p + dt * k3)
            p = p + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            # clip into domain
            p[0] = np.clip(p[0], 0, Lx)
            p[1] = np.clip(p[1], 0, Ly)
            trails[i, k + 1, :] = p
    return trails


def residence_time(
    trails: NDArray[np.float64], *, box: tuple[float, float, float, float],
    dt: float,
) -> NDArray[np.float64]:
    """particle 별 box 내 머문 시간."""
    xmin, ymin, xmax, ymax = box
    mask = (
        (trails[:, :, 0] >= xmin) & (trails[:, :, 0] <= xmax)
        & (trails[:, :, 1] >= ymin) & (trails[:, :, 1] <= ymax)
    )
    return mask.sum(axis=1) * dt


__all__ = ["track_particles_2d", "residence_time"]

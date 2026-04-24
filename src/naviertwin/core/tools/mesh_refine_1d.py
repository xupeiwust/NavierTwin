"""1D 적응 격자 세분화 — 오차 지표 기반.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.tools.mesh_refine_1d import refine_by_gradient
    >>> x = np.linspace(0, 1, 11)
    >>> f = np.where(x < 0.5, 0.0, 1.0)
    >>> x2, f2 = refine_by_gradient(x, f, threshold=0.1)
    >>> len(x2) > len(x)
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def refine_by_gradient(
    x: NDArray[np.float64], f: NDArray[np.float64],
    *, threshold: float = 0.1, max_passes: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """셀별 |Δf| > threshold 면 중간점 삽입 (선형 보간)."""
    x = np.asarray(x, dtype=np.float64).copy()
    f = np.asarray(f, dtype=np.float64).copy()
    for _ in range(max_passes):
        df = np.abs(np.diff(f))
        mask = df > threshold
        if not np.any(mask):
            break
        new_x = []
        new_f = []
        for i in range(len(x) - 1):
            new_x.append(x[i])
            new_f.append(f[i])
            if mask[i]:
                new_x.append(0.5 * (x[i] + x[i + 1]))
                new_f.append(0.5 * (f[i] + f[i + 1]))
        new_x.append(x[-1])
        new_f.append(f[-1])
        x = np.array(new_x)
        f = np.array(new_f)
    return x, f


def coarsen_by_tolerance(
    x: NDArray[np.float64], f: NDArray[np.float64],
    *, tol: float = 1e-3,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """인접 두 셀의 2차 차분이 작으면 중간점 제거."""
    x = np.asarray(x, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    keep = np.ones(len(x), dtype=bool)
    for i in range(1, len(x) - 1):
        # 2차 차분
        d2 = f[i - 1] - 2 * f[i] + f[i + 1]
        if abs(d2) < tol:
            keep[i] = False
    return x[keep], f[keep]


__all__ = ["refine_by_gradient", "coarsen_by_tolerance"]

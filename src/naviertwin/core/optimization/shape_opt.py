"""Shape optimization — parametric Bezier control points.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.shape_opt import bezier_eval, optimize_bezier
    >>> ctrl = np.array([[0., 0], [0.5, 1], [1., 0]])
    >>> pts = bezier_eval(ctrl, t=np.linspace(0, 1, 5))
    >>> pts.shape
    (5, 2)
"""

from __future__ import annotations

from collections.abc import Callable
from math import comb

import numpy as np
from numpy.typing import NDArray


def bezier_eval(
    ctrl: NDArray[np.float64], t: NDArray[np.float64],
) -> NDArray[np.float64]:
    """De Casteljau Bezier 평가."""
    ctrl = np.asarray(ctrl, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    n = ctrl.shape[0] - 1
    pts = np.zeros((t.size, ctrl.shape[1]))
    for k, tk in enumerate(t):
        pts[k] = sum(
            ctrl[i] * comb(n, i) * (1 - tk) ** (n - i) * tk ** i
            for i in range(n + 1)
        )
    return pts


def optimize_bezier(
    objective: Callable[[NDArray], float],
    ctrl0: NDArray[np.float64],
    *,
    fixed_endpoints: bool = True,
    max_iter: int = 50,
    lr: float = 0.05,
    eps: float = 1e-4,
) -> NDArray[np.float64]:
    """Bezier 제어점에 대해 finite-diff gradient descent."""
    ctrl = np.asarray(ctrl0, dtype=np.float64).copy()
    for _ in range(max_iter):
        g = np.zeros_like(ctrl)
        f0 = objective(ctrl)
        rng_pts = range(1, len(ctrl) - 1) if fixed_endpoints else range(len(ctrl))
        for i in rng_pts:
            for j in range(ctrl.shape[1]):
                ctrl[i, j] += eps
                g[i, j] = (objective(ctrl) - f0) / eps
                ctrl[i, j] -= eps
        ctrl = ctrl - lr * g
    return ctrl


__all__ = ["bezier_eval", "optimize_bezier"]

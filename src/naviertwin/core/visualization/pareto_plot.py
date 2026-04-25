"""Pareto front extraction (2D, minimization).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.visualization.pareto_plot import pareto_front
    >>> obj = np.array([[1, 5], [2, 3], [3, 4], [4, 1]])
    >>> mask = pareto_front(obj)
    >>> mask.tolist()
    [True, True, False, True]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def pareto_front(objectives: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Returns boolean mask of non-dominated points (minimization)."""
    obj = np.asarray(objectives, dtype=np.float64)
    n = obj.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # any j strictly dominates i?
        for j in range(n):
            if i == j:
                continue
            if (obj[j] <= obj[i]).all() and (obj[j] < obj[i]).any():
                keep[i] = False
                break
    return keep


__all__ = ["pareto_front"]

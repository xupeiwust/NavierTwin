"""r-adaptation — node movement toward high-gradient regions (1D).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.amr.r_adapt import r_adapt_1d
    >>> x = np.linspace(0, 1, 11)
    >>> w = np.where(x > 0.5, 5.0, 1.0)
    >>> x2 = r_adapt_1d(x, w, n_iter=20)
    >>> x2.shape
    (11,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def r_adapt_1d(
    x: NDArray[np.float64], weights: NDArray[np.float64], *, n_iter: int = 20,
) -> NDArray[np.float64]:
    """equidistribution: ∫ w dx between nodes equal."""
    x = np.asarray(x, dtype=np.float64).copy()
    w = np.asarray(weights, dtype=np.float64)
    n = len(x)
    for _ in range(n_iter):
        # cumulative weight
        cw = np.concatenate([[0.0], np.cumsum(0.5 * (w[:-1] + w[1:]) * np.diff(x))])
        target = np.linspace(0, cw[-1], n)
        # invert: find x such that cw(x) = target
        x_new = np.interp(target, cw, x)
        x_new[0] = x[0]
        x_new[-1] = x[-1]
        x = x_new
    return x


__all__ = ["r_adapt_1d"]

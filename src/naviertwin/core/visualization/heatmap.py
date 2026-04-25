"""Heatmap text annotation generator (no matplotlib).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.visualization.heatmap import annotate_heatmap
    >>> M = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> ann = annotate_heatmap(M, fmt="{:.1f}")
    >>> ann[(0, 0)]
    '1.0'
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def annotate_heatmap(
    M: NDArray[np.float64], *, fmt: str = "{:.2f}",
) -> dict[tuple[int, int], str]:
    M = np.asarray(M)
    out = {}
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            out[(i, j)] = fmt.format(float(M[i, j]))
    return out


__all__ = ["annotate_heatmap"]

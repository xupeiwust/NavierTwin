"""Marching cubes-lite — 간단 2D version (marching squares) + 3D voxel iso-segments.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.tools.marching_cubes_lite import marching_squares
    >>> f = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
    >>> segs = marching_squares(f, level=0.5)
    >>> len(segs) > 0
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def marching_squares(
    f: NDArray[np.float64], level: float = 0.0,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """2D iso-contour 추출. (i, j) 인덱스 좌표 반환."""
    f = np.asarray(f, dtype=np.float64)
    nx, ny = f.shape
    segments = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            v = [f[i, j], f[i + 1, j], f[i + 1, j + 1], f[i, j + 1]]
            below = [vi < level for vi in v]
            # cell edges: 0:bot, 1:right, 2:top, 3:left
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            edges_pts = []
            for k in range(4):
                a, b = k, (k + 1) % 4
                if below[a] != below[b]:
                    t = (level - v[a]) / (v[b] - v[a] + 1e-30)
                    pa = corners[a]
                    pb = corners[b]
                    p = (pa[0] + t * (pb[0] - pa[0]),
                         pa[1] + t * (pb[1] - pa[1]))
                    edges_pts.append(p)
            if len(edges_pts) == 2:
                segments.append((edges_pts[0], edges_pts[1]))
            elif len(edges_pts) == 4:
                segments.append((edges_pts[0], edges_pts[1]))
                segments.append((edges_pts[2], edges_pts[3]))
    return segments


__all__ = ["marching_squares"]

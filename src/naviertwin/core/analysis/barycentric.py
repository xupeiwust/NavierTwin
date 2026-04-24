"""Barycentric 좌표 + 삼각형 FE 보간.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.barycentric import barycentric_2d
    >>> tri = np.array([[0., 0.], [1., 0.], [0., 1.]])
    >>> bc = barycentric_2d(tri, np.array([0.25, 0.25]))
    >>> np.allclose(bc.sum(), 1.0)
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def barycentric_2d(
    triangle: NDArray[np.float64], p: NDArray[np.float64],
) -> NDArray[np.float64]:
    """삼각형 (3, 2) + 점 (2,) → barycentric coords (3,)."""
    A, B, C = triangle
    v0 = B - A
    v1 = C - A
    v2 = p - A
    d00 = v0 @ v0
    d01 = v0 @ v1
    d11 = v1 @ v1
    d20 = v2 @ v0
    d21 = v2 @ v1
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-20:
        raise ValueError("degenerate triangle")
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w])


def triangle_interp(
    triangle: NDArray[np.float64], values: NDArray[np.float64],
    p: NDArray[np.float64],
) -> float:
    bc = barycentric_2d(triangle, p)
    return float(bc @ np.asarray(values))


def locate_triangle(
    points: NDArray[np.float64], simplices: NDArray[np.int64],
    p: NDArray[np.float64],
) -> int:
    """p 를 포함하는 삼각형 인덱스. 없으면 -1."""
    for i, tri in enumerate(simplices):
        try:
            bc = barycentric_2d(points[tri], p)
        except ValueError:
            continue
        if (bc >= -1e-10).all():
            return i
    return -1


__all__ = ["barycentric_2d", "triangle_interp", "locate_triangle"]

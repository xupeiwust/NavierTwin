"""2D Delaunay triangulation + P1 질량/강성 행렬 (scipy 기반).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.tools.delaunay_2d import triangulate
    >>> pts = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=float)
    >>> tri = triangulate(pts)
    >>> tri["simplices"].shape[1] == 3
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def triangulate(points: NDArray[np.float64]) -> dict:
    try:
        from scipy.spatial import Delaunay
    except ImportError as exc:
        raise RuntimeError("scipy 필요") from exc
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[1] != 2:
        raise ValueError("2D points 필요")
    tri = Delaunay(pts)
    return {"points": pts, "simplices": tri.simplices.astype(np.int64)}


def triangle_areas(
    points: NDArray[np.float64], simplices: NDArray[np.int64],
) -> NDArray[np.float64]:
    p = points[simplices]  # (n_tri, 3, 2)
    v1 = p[:, 1] - p[:, 0]
    v2 = p[:, 2] - p[:, 0]
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])


def lumped_mass_matrix(
    points: NDArray[np.float64], simplices: NDArray[np.int64],
) -> NDArray[np.float64]:
    """노드별 lumped mass = 인접 삼각형 면적 합 / 3."""
    n = points.shape[0]
    areas = triangle_areas(points, simplices)
    m = np.zeros(n)
    for idx, a in zip(simplices, areas):
        m[idx] += a / 3.0
    return m


def p1_stiffness_matrix(
    points: NDArray[np.float64], simplices: NDArray[np.int64],
) -> NDArray[np.float64]:
    """∇φ_i · ∇φ_j 적분 — P1 element. 반환 dense (n, n)."""
    n = points.shape[0]
    K = np.zeros((n, n))
    for tri in simplices:
        xs = points[tri, 0]
        ys = points[tri, 1]
        det = (
            (xs[1] - xs[0]) * (ys[2] - ys[0])
            - (xs[2] - xs[0]) * (ys[1] - ys[0])
        )
        area = 0.5 * abs(det)
        if area < 1e-20:
            continue
        b = np.array([ys[1] - ys[2], ys[2] - ys[0], ys[0] - ys[1]])
        c = np.array([xs[2] - xs[1], xs[0] - xs[2], xs[1] - xs[0]])
        Ke = (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)
        for i in range(3):
            for j in range(3):
                K[tri[i], tri[j]] += Ke[i, j]
    return K


__all__ = [
    "triangulate", "triangle_areas", "lumped_mass_matrix", "p1_stiffness_matrix",
]

"""간단한 공간 색인 — grid bucket (빠른 근접 점 검색).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.spatial_index import GridIndex
    >>> pts = np.random.default_rng(0).standard_normal((1000, 3))
    >>> idx = GridIndex(pts, cell_size=0.5)
    >>> idx.query_radius(np.array([0., 0., 0.]), 0.3).size >= 0
    True
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray


class GridIndex:
    """균일 셀 공간 색인 — O(1) 평균 근접 질의."""

    def __init__(
        self, points: NDArray[np.float64], cell_size: float,
    ) -> None:
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.ndim != 2:
            raise ValueError("points (n, d) 필요")
        self.cell = float(cell_size)
        self._buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        keys = np.floor(self.points / self.cell).astype(np.int64)
        for i, k in enumerate(map(tuple, keys)):
            self._buckets[k].append(i)

    def query_radius(
        self, q: NDArray[np.float64], radius: float,
    ) -> NDArray[np.int64]:
        q = np.asarray(q, dtype=np.float64).ravel()
        d = self.points.shape[1]
        span = int(np.ceil(radius / self.cell))
        kq = np.floor(q / self.cell).astype(np.int64)
        hits: list[int] = []
        # 주변 셀 iterate
        it = np.ndindex(*([2 * span + 1] * d))
        for delta in it:
            key = tuple(kq + np.array(delta) - span)
            bucket = self._buckets.get(key, None)
            if bucket:
                hits.extend(bucket)
        if not hits:
            return np.array([], dtype=np.int64)
        hits_arr = np.asarray(hits, dtype=np.int64)
        dists = np.linalg.norm(self.points[hits_arr] - q, axis=1)
        return hits_arr[dists <= radius]

    def nearest(self, q: NDArray[np.float64]) -> int:
        """가장 가까운 점의 인덱스."""
        # 셀이 비어있을 수도 있으므로 반경 확장
        r = self.cell
        for _ in range(20):
            idx = self.query_radius(q, r)
            if idx.size > 0:
                dists = np.linalg.norm(self.points[idx] - q, axis=1)
                return int(idx[np.argmin(dists)])
            r *= 2
        # fallback
        dists = np.linalg.norm(self.points - q, axis=1)
        return int(np.argmin(dists))


__all__ = ["GridIndex"]

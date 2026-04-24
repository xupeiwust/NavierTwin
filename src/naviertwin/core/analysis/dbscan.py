"""DBSCAN 클러스터링 (scratch) — 밀도 기반 이상치/클러스터 탐지.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.dbscan import dbscan
    >>> pts = np.random.default_rng(0).standard_normal((50, 2))
    >>> labels = dbscan(pts, eps=0.5, min_samples=3)
    >>> labels.shape
    (50,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def dbscan(
    points: NDArray[np.float64], eps: float = 0.5, min_samples: int = 5,
) -> NDArray[np.int64]:
    """간단 DBSCAN. 반환: label (N,), -1 = noise."""
    X = np.asarray(points, dtype=np.float64)
    n = X.shape[0]
    labels = -np.ones(n, dtype=np.int64)

    # pairwise distance (brute)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    neighbors = [np.where(D[i] <= eps)[0].tolist() for i in range(n)]

    cluster = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        if len(neighbors[i]) < min_samples:
            continue  # 아직 noise
        labels[i] = cluster
        seeds = list(neighbors[i])
        while seeds:
            j = seeds.pop()
            if labels[j] == -1:
                labels[j] = cluster
            elif labels[j] != cluster:
                continue
            if len(neighbors[j]) >= min_samples:
                for k in neighbors[j]:
                    if labels[k] == -1:
                        seeds.append(k)
        cluster += 1
    return labels


def n_clusters(labels: NDArray[np.int64]) -> int:
    return int(len(set(labels.tolist()) - {-1}))


__all__ = ["dbscan", "n_clusters"]

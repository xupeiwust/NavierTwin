"""k-means++ (scratch) — 거리 기반 반복.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.kmeans import kmeans
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.standard_normal((50, 2)), rng.standard_normal((50, 2)) + 5])
    >>> centers, labels = kmeans(X, k=2, seed=0)
    >>> centers.shape
    (2, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _init_plusplus(
    X: NDArray[np.float64], k: int, rng: np.random.Generator,
) -> NDArray[np.float64]:
    n = X.shape[0]
    idx0 = int(rng.integers(n))
    centers = [X[idx0]]
    for _ in range(k - 1):
        dists = np.min(
            np.linalg.norm(
                X[:, None, :] - np.asarray(centers)[None, :, :], axis=2,
            ),
            axis=1,
        )
        probs = dists ** 2
        s = probs.sum()
        if s <= 0:
            idx = int(rng.integers(n))
        else:
            idx = int(rng.choice(n, p=probs / s))
        centers.append(X[idx])
    return np.stack(centers, axis=0)


def kmeans(
    X: NDArray[np.float64], k: int = 3,
    *, max_iter: int = 100, tol: float = 1e-6, seed: int | None = 0,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    centers = _init_plusplus(X, k, rng)
    for _ in range(max_iter):
        # assignment
        D = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(D, axis=1)
        # update
        new_centers = centers.copy()
        for j in range(k):
            pts = X[labels == j]
            if pts.size > 0:
                new_centers[j] = pts.mean(axis=0)
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < tol:
            break
    return centers, labels.astype(np.int64)


def inertia(
    X: NDArray[np.float64], centers: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> float:
    s = 0.0
    for j in range(centers.shape[0]):
        pts = X[labels == j]
        if pts.size > 0:
            s += float(np.sum((pts - centers[j]) ** 2))
    return s


__all__ = ["kmeans", "inertia"]

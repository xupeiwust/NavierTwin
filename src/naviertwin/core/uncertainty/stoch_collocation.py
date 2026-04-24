"""Stochastic collocation on sparse grid (Smolyak 1D-product, Clenshaw-Curtis).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.uncertainty.stoch_collocation import (
    ...     clenshaw_curtis_nodes, sparse_grid_2d,
    ... )
    >>> nodes, weights = clenshaw_curtis_nodes(level=3)
    >>> nodes.shape
    (5,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def clenshaw_curtis_nodes(level: int) -> tuple[NDArray, NDArray]:
    """level→ N=2^(level-1)+1 nodes on [-1,1]."""
    if level == 1:
        return np.array([0.0]), np.array([2.0])
    n = 2 ** (level - 1) + 1
    k = np.arange(n)
    x = np.cos(np.pi * k / (n - 1))
    # Clenshaw-Curtis weights
    w = np.zeros(n)
    for i in range(n):
        c = 1.0 if i in (0, n - 1) else 2.0
        s = 0.0
        for j in range(1, (n - 1) // 2 + 1):
            d = 2.0 / (4 * j * j - 1)
            s += d * np.cos(2 * np.pi * j * i / (n - 1))
        w[i] = c / (n - 1) * (1 - s)
    return x, w


def sparse_grid_2d(level: int = 3) -> tuple[NDArray, NDArray]:
    """2D Smolyak (sum_{l1+l2 <= level+1} ⊗) on Clenshaw-Curtis."""
    nodes_all = []
    weights_all = []
    for l1 in range(1, level + 1):
        for l2 in range(1, level + 2 - l1):
            x1, w1 = clenshaw_curtis_nodes(l1)
            x2, w2 = clenshaw_curtis_nodes(l2)
            X1, X2 = np.meshgrid(x1, x2, indexing="ij")
            W1, W2 = np.meshgrid(w1, w2, indexing="ij")
            sign = (-1) ** (level + 1 - (l1 + l2))
            from math import comb
            coef = sign * comb(1, level + 1 - (l1 + l2)) if (level + 1 - (l1 + l2)) <= 1 else 0
            if coef != 0:
                pts = np.column_stack([X1.ravel(), X2.ravel()])
                ws = (W1 * W2).ravel() * coef
                nodes_all.append(pts)
                weights_all.append(ws)
    nodes = np.vstack(nodes_all)
    weights = np.concatenate(weights_all)
    return nodes, weights


__all__ = ["clenshaw_curtis_nodes", "sparse_grid_2d"]

"""Sobol sensitivity — Saltelli matrix-style first-order + total.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.verification.sobol import sobol_indices
    >>> rng = np.random.default_rng(0)
    >>> def model(X): return X[:, 0] + 0.1 * X[:, 1]
    >>> S, ST = sobol_indices(model, n_dim=2, n_samples=2000, rng=rng)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def sobol_indices(
    model: Callable[[NDArray], NDArray], *,
    n_dim: int, n_samples: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray, NDArray]:
    """Returns (S_i, S_T_i)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    A = rng.uniform(0, 1, (n_samples, n_dim))
    B = rng.uniform(0, 1, (n_samples, n_dim))
    fA = model(A)
    fB = model(B)
    var_y = float(np.var(np.concatenate([fA, fB])) + 1e-30)
    S = np.zeros(n_dim)
    ST = np.zeros(n_dim)
    for i in range(n_dim):
        AB = A.copy()
        AB[:, i] = B[:, i]
        fAB = model(AB)
        S[i] = float(np.mean(fB * (fAB - fA))) / var_y
        ST[i] = float(0.5 * np.mean((fA - fAB) ** 2)) / var_y
    return S, ST


__all__ = ["sobol_indices"]

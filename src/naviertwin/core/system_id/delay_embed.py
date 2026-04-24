"""Takens delay embedding + mutual information 지연 추정.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.system_id.delay_embed import delay_embed
    >>> x = np.sin(np.linspace(0, 10, 500))
    >>> Y = delay_embed(x, dim=3, delay=5)
    >>> Y.shape
    (490, 3)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def delay_embed(
    x: NDArray[np.float64], dim: int = 3, delay: int = 1,
) -> NDArray[np.float64]:
    """x (T,) → (T - (dim-1)*delay, dim)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    m = x.size - (dim - 1) * delay
    if m <= 0:
        raise ValueError("너무 짧은 시계열")
    out = np.zeros((m, dim))
    for i in range(dim):
        out[:, i] = x[i * delay: i * delay + m]
    return out


def autocorrelation(x: NDArray[np.float64], max_lag: int = 50) -> NDArray[np.float64]:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x - x.mean()
    n = x.size
    out = np.zeros(max_lag + 1)
    denom = float(x @ x) + 1e-30
    for lag in range(max_lag + 1):
        out[lag] = float(x[: n - lag] @ x[lag:]) / denom
    return out


def first_zero_crossing(corr: NDArray[np.float64]) -> int:
    """autocorr 첫 zero-crossing → 추천 delay."""
    for i in range(1, corr.size):
        if corr[i] <= 0:
            return i
    return corr.size - 1


__all__ = ["delay_embed", "autocorrelation", "first_zero_crossing"]

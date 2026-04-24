"""Shifted POD — Reiss et al. 2018, traveling structure decomposition.

X(x, t) ≈ Σ_k T_{c_k(t)} u_k(x).  여기서는 단일 shift 추정만 (1-frame, c(t) 추정 후 정렬).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.dimensionality_reduction.linear.shifted_pod import (
    ...     estimate_shifts,
    ... )
    >>> x = np.linspace(0, 2*np.pi, 64)
    >>> X = np.array([np.sin(x - 0.1*t) for t in range(20)]).T
    >>> shifts = estimate_shifts(X)
    >>> shifts.shape
    (20,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def estimate_shifts(X: NDArray[np.float64]) -> NDArray[np.int_]:
    """Cross-correlation argmax against first frame → integer pixel shifts."""
    X = np.asarray(X, dtype=np.float64)
    n, m = X.shape
    ref = X[:, 0]
    shifts = np.zeros(m, dtype=int)
    for k in range(m):
        # FFT-based cross-corr
        c = np.real(np.fft.ifft(np.fft.fft(X[:, k]) * np.conj(np.fft.fft(ref))))
        idx = int(np.argmax(c))
        if idx > n // 2:
            idx -= n
        shifts[k] = idx
    return shifts


def shifted_pod(
    X: NDArray[np.float64], rank: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """간단 shifted-POD: shift 추정 → 정렬 → POD."""
    shifts = estimate_shifts(X)
    n, m = X.shape
    X_aligned = np.zeros_like(X)
    for k in range(m):
        X_aligned[:, k] = np.roll(X[:, k], -shifts[k])
    U, _, _ = np.linalg.svd(X_aligned, full_matrices=False)
    return U[:, :rank], shifts


__all__ = ["estimate_shifts", "shifted_pod"]

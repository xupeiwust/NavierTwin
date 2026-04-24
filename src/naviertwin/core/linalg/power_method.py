"""Power iteration / inverse / Rayleigh quotient.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.linalg.power_method import power_iteration
    >>> A = np.diag([5.0, 2.0, 1.0])
    >>> lam, _ = power_iteration(A, n_iter=200)
    >>> abs(lam - 5.0) < 1e-8
    True
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def power_iteration(
    A: NDArray[np.float64], n_iter: int = 200, *,
    x0: NDArray | None = None, tol: float = 1e-10, seed: int | None = 0,
) -> tuple[float, NDArray[np.float64]]:
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n) if x0 is None else np.asarray(x0).ravel()
    x = x / (np.linalg.norm(x) + 1e-30)
    lam_prev = 0.0
    for _ in range(n_iter):
        y = A @ x
        x = y / (np.linalg.norm(y) + 1e-30)
        lam = float(x @ A @ x)
        if abs(lam - lam_prev) < tol * max(1.0, abs(lam)):
            break
        lam_prev = lam
    return lam, x


def inverse_power(
    A: NDArray[np.float64], shift: float = 0.0, n_iter: int = 100,
    *, seed: int | None = 0,
) -> tuple[float, NDArray[np.float64]]:
    """(A - shift I)⁻¹ 에 대한 power iteration → shift 에 가까운 고유값."""
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    M = A - shift * np.eye(n)
    # 정확한 계산을 위해 LU 한 번
    try:
        from scipy.linalg import lu_factor, lu_solve

        lu, piv = lu_factor(M)

        def solver(b: Any) -> Any:
            return lu_solve((lu, piv), b)
    except ImportError:
        def solver(b: Any) -> Any:
            return np.linalg.solve(M, b)

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    x = x / (np.linalg.norm(x) + 1e-30)
    for _ in range(n_iter):
        y = solver(x)
        x = y / (np.linalg.norm(y) + 1e-30)
    lam = float(x @ A @ x)
    return lam, x


def rayleigh_quotient(A: NDArray[np.float64], x: NDArray[np.float64]) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    return float((x @ A @ x) / (x @ x + 1e-30))


__all__ = ["power_iteration", "inverse_power", "rayleigh_quotient"]

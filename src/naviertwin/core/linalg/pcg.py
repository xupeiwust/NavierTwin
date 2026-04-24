"""Preconditioned Conjugate Gradient (Jacobi preconditioner).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.linalg.pcg import pcg
    >>> A = np.array([[4., 1.], [1., 3.]])
    >>> b = np.array([1., 2.])
    >>> x, info = pcg(A, b)
    >>> info["converged"]
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def jacobi_preconditioner(A: NDArray[np.float64]) -> Callable:
    d = np.diag(A)
    if np.any(d == 0):
        raise ValueError("zero diagonal")
    inv_d = 1.0 / d
    return lambda r: inv_d * r


def pcg(
    A: NDArray[np.float64], b: NDArray[np.float64],
    *, M: Callable | None = None,
    max_iter: int | None = None, tol: float = 1e-10,
    x0: NDArray | None = None,
) -> tuple[NDArray[np.float64], dict]:
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    n = b.size
    x = np.zeros(n) if x0 is None else np.asarray(x0).ravel().copy()
    M = M if M is not None else jacobi_preconditioner(A)
    r = b - A @ x
    z = M(r)
    p = z.copy()
    rz = float(r @ z)
    mi = max_iter or 2 * n
    for i in range(mi):
        Ap = A @ p
        alpha = rz / (p @ Ap + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        if np.linalg.norm(r) < tol:
            return x, {"iters": i + 1, "residual": float(np.linalg.norm(r)),
                       "converged": True}
        z = M(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, {"iters": mi, "residual": float(np.linalg.norm(r)), "converged": False}


__all__ = ["pcg", "jacobi_preconditioner"]

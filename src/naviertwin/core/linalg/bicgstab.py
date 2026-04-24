"""BiCGStab — 비대칭 선형시스템 풀이 (van der Vorst 1992).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.linalg.bicgstab import bicgstab
    >>> A = np.array([[4., 1., 0.], [2., 5., 1.], [0., 1., 3.]])
    >>> b = np.array([1., 2., 3.])
    >>> x, info = bicgstab(A, b)
    >>> info["converged"]
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def bicgstab(
    A, b: NDArray[np.float64],
    *, x0: NDArray | None = None, max_iter: int | None = None,
    tol: float = 1e-10, M: Callable | None = None,
) -> tuple[NDArray[np.float64], dict]:
    """A 는 dense ndarray 또는 linear op callable."""
    b = np.asarray(b, dtype=np.float64).ravel()
    n = b.size
    if callable(A):
        A_fn = A
    else:
        A_arr = np.asarray(A, dtype=np.float64)

        def A_fn(v):
            return A_arr @ v

    x = np.zeros(n) if x0 is None else np.asarray(x0).ravel().copy()
    M_fn = M if M is not None else (lambda v: v)
    r = b - A_fn(x)
    r_hat = r.copy()
    rho_prev = alpha = omega = 1.0
    v = np.zeros(n)
    p = np.zeros(n)
    mi = max_iter or 2 * n
    for i in range(mi):
        rho = float(r_hat @ r)
        if abs(rho) < 1e-30:
            break
        beta = (rho / rho_prev) * (alpha / (omega + 1e-30))
        p = r + beta * (p - omega * v)
        y = M_fn(p)
        v = A_fn(y)
        alpha = rho / (r_hat @ v + 1e-30)
        s = r - alpha * v
        if np.linalg.norm(s) < tol:
            x = x + alpha * y
            return x, {"iters": i + 1, "residual": float(np.linalg.norm(b - A_fn(x))),
                       "converged": True}
        z = M_fn(s)
        t = A_fn(z)
        omega = float(t @ s) / (t @ t + 1e-30)
        x = x + alpha * y + omega * z
        r = s - omega * t
        if np.linalg.norm(r) < tol:
            return x, {"iters": i + 1, "residual": float(np.linalg.norm(r)),
                       "converged": True}
        rho_prev = rho
    return x, {"iters": mi, "residual": float(np.linalg.norm(b - A_fn(x))),
               "converged": False}


__all__ = ["bicgstab"]

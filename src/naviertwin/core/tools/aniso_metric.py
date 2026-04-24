"""Anisotropic mesh metric — Hessian-based metric tensor.

M(x) = |H_f(x)| (절대값 고유분해), 길이 측정 ds² = dx M dx.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.tools.aniso_metric import metric_from_hessian
    >>> H = np.eye(2) * np.array([[2.0]])
    >>> H = np.array([np.eye(2)*2])
    >>> M = metric_from_hessian(H)
    >>> M.shape
    (1, 2, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def metric_from_hessian(
    H: NDArray[np.float64], *, h_min: float = 1e-3, h_max: float = 1.0,
) -> NDArray[np.float64]:
    """|H| 의 spectrum 을 [1/h_max², 1/h_min²] 에 clip → metric tensor."""
    H = np.asarray(H, dtype=np.float64)
    flat = H.reshape(-1, H.shape[-2], H.shape[-1])
    out = np.zeros_like(flat)
    lo, hi = 1.0 / (h_max * h_max), 1.0 / (h_min * h_min)
    for i in range(flat.shape[0]):
        Hs = 0.5 * (flat[i] + flat[i].T)
        w, V = np.linalg.eigh(Hs)
        w_abs = np.clip(np.abs(w), lo, hi)
        out[i] = V @ np.diag(w_abs) @ V.T
    return out.reshape(H.shape)


def edge_length_metric(
    M_a: NDArray[np.float64], M_b: NDArray[np.float64],
    a: NDArray[np.float64], b: NDArray[np.float64],
) -> float:
    """metric-induced edge length (mid-point average metric)."""
    M = 0.5 * (M_a + M_b)
    e = b - a
    return float(np.sqrt(e @ M @ e))


__all__ = ["edge_length_metric", "metric_from_hessian"]

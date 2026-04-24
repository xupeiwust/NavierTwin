"""Lambda-2 criterion — 소용돌이 식별.

Λ₂ = 2nd eigenvalue of (S² + Ω²), Λ₂ < 0 → vortex.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.lambda2 import lambda2_2d
    >>> u = np.zeros((32, 32)); v = np.zeros((32, 32))
    >>> L = lambda2_2d(u, v)
    >>> L.shape
    (32, 32)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def lambda2_2d(
    u: NDArray[np.float64], v: NDArray[np.float64],
    dx: float = 1.0, dy: float = 1.0,
) -> NDArray[np.float64]:
    """2D Λ₂ (두 번째 eigenvalue). 음수 → vortex."""
    du_dx = np.gradient(u, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    # velocity gradient J
    # S = (J + Jᵀ)/2, Ω = (J - Jᵀ)/2
    S11 = du_dx
    S22 = dv_dy
    S12 = 0.5 * (du_dy + dv_dx)
    O12 = 0.5 * (dv_dx - du_dy)
    # M = S² + Ω² (2×2)
    M11 = S11 * S11 + S12 * S12 - O12 * O12
    M22 = S22 * S22 + S12 * S12 - O12 * O12
    M12 = S11 * S12 + S12 * S22  # off-diagonal (Ω 항은 off-diagonal 에서 0)
    # 2x2 고유값: (M11+M22)/2 ± sqrt(((M11-M22)/2)² + M12²)
    mid = 0.5 * (M11 + M22)
    rad = np.sqrt(((M11 - M22) / 2) ** 2 + M12 * M12)
    lam2 = mid - rad
    return lam2


__all__ = ["lambda2_2d"]

"""FE² (Feyel) two-scale skeleton — macro element invokes micro RVE.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.multiscale.fe2 import fe2_macro_stress
    >>> def micro(strain): return 200.0 * strain
    >>> fe2_macro_stress(np.array([0.01, 0.02]), micro)
    array([2., 4.])
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


def fe2_macro_stress(
    macro_strain: NDArray[np.float64],
    micro_rve: Callable[[float], float],
) -> NDArray[np.float64]:
    s = np.asarray(macro_strain, dtype=np.float64)
    return np.array([float(micro_rve(float(e))) for e in s])


def fe2_tangent_fd(
    macro_strain: NDArray[np.float64],
    micro_rve: Callable[[float], float],
    *,
    eps: float = 1e-6,
) -> NDArray[np.float64]:
    s = np.asarray(macro_strain, dtype=np.float64)
    out = np.zeros_like(s)
    for i, e in enumerate(s):
        out[i] = (micro_rve(float(e) + eps) - micro_rve(float(e) - eps)) / (2 * eps)
    return out


__all__ = ["fe2_macro_stress", "fe2_tangent_fd"]

"""Fast-marching lite — 1D SDF reinit (Sethian style 단순화).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.geometry.sdf_reinit import fast_march_1d
    >>> phi = np.array([5., 3, 0.0, -3, -5])
    >>> phi2 = fast_march_1d(phi, dx=1.0)
    >>> phi2[2]
    0.0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def fast_march_1d(
    phi: NDArray[np.float64], *, dx: float = 1.0,
) -> NDArray[np.float64]:
    """1D 등거리 격자 → |x - x_0|, x_0 = 가장 가까운 zero-crossing."""
    phi = np.asarray(phi, dtype=np.float64)
    n = len(phi)
    sign = np.sign(phi)
    # detect interfaces (sign change)
    iface_locs = []
    for i in range(n - 1):
        if phi[i] == 0:
            iface_locs.append(i * dx)
        elif phi[i] * phi[i + 1] < 0:
            t = phi[i] / (phi[i] - phi[i + 1])
            iface_locs.append((i + t) * dx)
    if not iface_locs:
        return phi.copy()
    iface_arr = np.asarray(iface_locs)
    xs = np.arange(n) * dx
    d = np.min(np.abs(xs[:, None] - iface_arr[None, :]), axis=1)
    return d * sign


__all__ = ["fast_march_1d"]

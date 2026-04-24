"""PPM — Colella & Woodward 1984, parabolic reconstruction.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.solvers.ppm import ppm_face_values
    >>> u = np.array([1., 2, 3, 4, 5])
    >>> uL, uR = ppm_face_values(u)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ppm_face_values(u: NDArray[np.float64]) -> tuple[float, float]:
    """5-point stencil → u_{i-1/2}, u_{i+1/2} for cell i=2."""
    u = np.asarray(u, dtype=np.float64)
    # 4th-order interp at faces (centered)
    u_face_left = (7 / 12) * (u[1] + u[2]) - (1 / 12) * (u[0] + u[3])
    u_face_right = (7 / 12) * (u[2] + u[3]) - (1 / 12) * (u[1] + u[4])
    return float(u_face_left), float(u_face_right)


def ppm_monotonize(u_im: float, u_i: float, u_ip: float,
                    uL: float, uR: float) -> tuple[float, float]:
    """Colella-Woodward monotonization."""
    if (uR - u_i) * (u_i - uL) <= 0:
        uL = u_i
        uR = u_i
    elif 6 * (uR - uL) * (u_i - 0.5 * (uL + uR)) > (uR - uL) ** 2:
        uL = 3 * u_i - 2 * uR
    elif 6 * (uR - uL) * (u_i - 0.5 * (uL + uR)) < -(uR - uL) ** 2:
        uR = 3 * u_i - 2 * uL
    return uL, uR


__all__ = ["ppm_face_values", "ppm_monotonize"]

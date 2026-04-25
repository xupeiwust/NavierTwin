"""Solar PV cell — single-diode I-V model (Lambert-W lite).

I = I_ph - I_0 (exp(V/(nV_T)) - 1).

Examples:
    >>> from naviertwin.core.applied.solar_pv import iv_curve
    >>> V, I = iv_curve(I_ph=8.0, I_0=1e-9, V_T=0.0257, n=1.0, n_pts=20)
"""

from __future__ import annotations

import numpy as np


def iv_curve(
    *, I_ph: float = 8.0, I_0: float = 1e-9, V_T: float = 0.0257,
    n: float = 1.0, V_max: float = 0.7, n_pts: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    V = np.linspace(0, V_max, n_pts)
    I = I_ph - I_0 * (np.exp(V / (n * V_T)) - 1)  # noqa: E741
    I = np.clip(I, 0, None)
    return V, I


def mppt(V: np.ndarray, I: np.ndarray) -> tuple[float, float, float]:
    P = V * I
    k = int(np.argmax(P))
    return float(V[k]), float(I[k]), float(P[k])


__all__ = ["iv_curve", "mppt"]

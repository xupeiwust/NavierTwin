"""Bingham plastic — τ = τ_y + μ_p γ̇ for τ > τ_y, else 0.

Examples:
    >>> from naviertwin.core.rheology.bingham import bingham_stress
    >>> bingham_stress(gamma_dot=2.0, tau_y=1.0, mu_p=0.5)
    2.0
"""

from __future__ import annotations


def bingham_stress(*, gamma_dot: float, tau_y: float, mu_p: float) -> float:
    g = float(gamma_dot)
    if g == 0:
        return 0.0
    sign = 1.0 if g > 0 else -1.0
    return sign * (tau_y + mu_p * abs(g))


def bingham_apparent_viscosity(*, gamma_dot: float, tau_y: float, mu_p: float,
                                eps: float = 1e-6) -> float:
    g = max(abs(float(gamma_dot)), eps)
    return float(tau_y / g + mu_p)


__all__ = ["bingham_apparent_viscosity", "bingham_stress"]

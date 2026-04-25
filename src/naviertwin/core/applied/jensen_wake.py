"""Jensen wake (1983) — single-turbine deficit, linear expansion.

V_wake/V0 = 1 - 2a / (1 + k x/R)².

Examples:
    >>> from naviertwin.core.applied.jensen_wake import wake_velocity
    >>> wake_velocity(V0=10, x=200, R=40, a=0.3, k=0.04)
"""

from __future__ import annotations


def wake_velocity(*, V0: float, x: float, R: float, a: float = 0.3,
                    k: float = 0.04) -> float:
    if x <= 0:
        return float(V0)
    deficit = 2.0 * a / (1.0 + k * x / R) ** 2
    return float(V0 * (1.0 - deficit))


def farm_velocity(*, V0: float, distances: list[float], R: float,
                    a: float = 0.3, k: float = 0.04) -> list[float]:
    """Series of turbines downstream; each sees previous wake."""
    out = [V0]
    for d in distances:
        out.append(wake_velocity(V0=out[-1], x=d, R=R, a=a, k=k))
    return out


__all__ = ["farm_velocity", "wake_velocity"]

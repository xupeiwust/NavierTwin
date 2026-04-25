"""Greenhouse energy balance — Q_solar - U A (T_in - T_out) = m c_p dT/dt.

Examples:
    >>> from naviertwin.core.applied.greenhouse import temperature_step
    >>> T = 25
    >>> for _ in range(60): T = temperature_step(T, T_out=10, Q_solar=500, U=5, A=10, m=200, cp=1000, dt=60)
"""

from __future__ import annotations


def temperature_step(
    T_in: float, *, T_out: float, Q_solar: float,
    U: float, A: float, m: float, cp: float, dt: float = 1.0,
) -> float:
    dTdt = (Q_solar - U * A * (T_in - T_out)) / (m * cp)
    return T_in + dt * dTdt


__all__ = ["temperature_step"]

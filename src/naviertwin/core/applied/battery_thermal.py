"""Battery thermal — lumped capacitance: m c_p dT/dt = Q_gen - h A (T - T_amb).

Examples:
    >>> from naviertwin.core.applied.battery_thermal import temperature_step
    >>> T = 25.0
    >>> for _ in range(100):
    ...     T = temperature_step(T, T_amb=20.0, Q_gen=10.0, h=5.0, A=0.1, m=0.5, cp=900, dt=1.0)
"""

from __future__ import annotations


def temperature_step(
    T: float, *, T_amb: float, Q_gen: float, h: float, A: float,
    m: float, cp: float, dt: float = 1.0,
) -> float:
    dTdt = (Q_gen - h * A * (T - T_amb)) / (m * cp)
    return T + dt * dTdt


def steady_temperature(*, T_amb: float, Q_gen: float, h: float, A: float) -> float:
    return T_amb + Q_gen / (h * A)


__all__ = ["steady_temperature", "temperature_step"]

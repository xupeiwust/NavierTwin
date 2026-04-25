"""Round 456 — battery thermal."""

from __future__ import annotations


class TestBattery:
    def test_steady(self) -> None:
        from naviertwin.core.applied.battery_thermal import steady_temperature

        T_ss = steady_temperature(T_amb=25.0, Q_gen=10.0, h=5.0, A=0.1)
        # T_ss = 25 + 10/0.5 = 45
        assert T_ss == 45.0

    def test_step_approaches_steady(self) -> None:
        from naviertwin.core.applied.battery_thermal import (
            steady_temperature,
            temperature_step,
        )

        T = 25.0
        for _ in range(2000):
            T = temperature_step(
                T, T_amb=25, Q_gen=10, h=5, A=0.1, m=0.5, cp=900, dt=1.0,
            )
        T_ss = steady_temperature(T_amb=25, Q_gen=10, h=5, A=0.1)
        assert abs(T - T_ss) < 0.5

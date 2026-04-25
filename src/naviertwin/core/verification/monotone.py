"""Monotone convergence test — successive errors strictly decreasing.

Examples:
    >>> from naviertwin.core.verification.monotone import is_monotone_decreasing
    >>> is_monotone_decreasing([1.0, 0.5, 0.25])
    True
"""

from __future__ import annotations

from collections.abc import Sequence


def is_monotone_decreasing(errs: Sequence[float], *, atol: float = 0.0) -> bool:
    e = list(errs)
    return all(e[i + 1] <= e[i] + atol for i in range(len(e) - 1))


def convergence_ratio(errs: Sequence[float]) -> list[float]:
    e = list(errs)
    out = []
    for i in range(len(e) - 1):
        out.append(e[i + 1] / max(abs(e[i]), 1e-30))
    return out


__all__ = ["convergence_ratio", "is_monotone_decreasing"]

"""Arrhenius rate law: k = A T^β exp(-E_a / (R T)).

Examples:
    >>> from naviertwin.core.reaction.arrhenius import arrhenius_k
    >>> arrhenius_k(A=1e10, T=1000, Ea=50000) > 0
    True
"""

from __future__ import annotations

import numpy as np

R_GAS = 8.314


def arrhenius_k(*, A: float, T: float, Ea: float, beta: float = 0.0) -> float:
    return float(A * T ** beta * np.exp(-Ea / (R_GAS * T)))


def reaction_rate(*, k: float, concentrations: list[float], orders: list[float]) -> float:
    rate = k
    for c, n in zip(concentrations, orders, strict=True):
        rate *= max(c, 0) ** n
    return float(rate)


__all__ = ["R_GAS", "arrhenius_k", "reaction_rate"]

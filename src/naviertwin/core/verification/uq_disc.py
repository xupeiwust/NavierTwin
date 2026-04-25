"""Discretization uncertainty propagator — combine GCI across QoIs.

Examples:
    >>> from naviertwin.core.verification.uq_disc import combined_disc_uncertainty
    >>> combined_disc_uncertainty([0.01, 0.02, 0.005])
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def combined_disc_uncertainty(uncs: Sequence[float]) -> float:
    """RSS of per-QoI discretization uncertainties."""
    return float(math.sqrt(sum(u * u for u in uncs)))


__all__ = ["combined_disc_uncertainty"]

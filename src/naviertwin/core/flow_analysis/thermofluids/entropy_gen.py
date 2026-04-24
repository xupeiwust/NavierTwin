"""엔트로피 생성율 계산 (heat + viscous).

Bejan 의 entropy generation rate:
    s_gen = (k / T²) ∇T · ∇T + (μ / T) Φ

여기서 Φ = 2·(e_ij e_ij) - (2/3)(div u)² — viscous dissipation.

2D 필드에 대한 간단 구현 (균일 격자).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.flow_analysis.thermofluids.entropy_gen import (
    ...     entropy_generation_2d,
    ... )
    >>> u = np.ones((20, 20))
    >>> v = np.zeros((20, 20))
    >>> T = 300.0 + np.random.default_rng(0).standard_normal((20, 20))
    >>> s = entropy_generation_2d(u, v, T, dx=0.1, dy=0.1, mu=1e-3, k=0.026)
    >>> s.shape
    (20, 20)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def entropy_generation_2d(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    T: NDArray[np.float64],
    dx: float = 1.0,
    dy: float = 1.0,
    mu: float = 1.8e-5,
    k: float = 0.026,
) -> NDArray[np.float64]:
    """2D 엔트로피 생성률 필드 (격자 정렬).

    Args:
        u, v: 속도 성분 (ny, nx).
        T: 온도 (ny, nx) — Kelvin.
        dx, dy: 격자 간격.
        mu: 동점성계수 [Pa·s].
        k: 열전도율 [W/(m·K)].

    Returns:
        s_gen (ny, nx).
    """
    for arr in (u, v, T):
        if arr.ndim != 2 or arr.shape != u.shape:
            raise ValueError("u, v, T 는 같은 2D shape 이어야 합니다")
    if np.any(T <= 0):
        raise ValueError("T 는 양수(Kelvin) 여야 합니다")

    dTdy, dTdx = np.gradient(T, dy, dx)
    dudy, dudx = np.gradient(u, dy, dx)
    dvdy, dvdx = np.gradient(v, dy, dx)

    # strain rate tensor e_ij
    e11 = dudx
    e22 = dvdy
    e12 = 0.5 * (dudy + dvdx)
    div = dudx + dvdy
    phi = 2.0 * (e11 ** 2 + e22 ** 2 + 2 * e12 ** 2) - (2.0 / 3.0) * div ** 2

    s_thermal = (k / T ** 2) * (dTdx ** 2 + dTdy ** 2)
    s_viscous = (mu / T) * phi

    logger.debug(
        "entropy_gen: thermal mean=%.3g, viscous mean=%.3g",
        float(s_thermal.mean()), float(s_viscous.mean()),
    )
    return s_thermal + s_viscous


__all__ = ["entropy_generation_2d"]

"""Hyperparameter sweep — grid + random.

Examples:
    >>> from naviertwin.utils.workflow.sweep import grid_sweep
    >>> list(grid_sweep({'lr': [0.1, 0.01], 'bs': [32]}))
    [{'lr': 0.1, 'bs': 32}, {'lr': 0.01, 'bs': 32}]
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product
from typing import Any

import numpy as np


def grid_sweep(space: dict[str, list[Any]]) -> Iterator[dict[str, Any]]:
    keys = list(space.keys())
    for combo in product(*[space[k] for k in keys]):
        yield dict(zip(keys, combo, strict=True))


def random_sweep(
    space: dict[str, tuple[float, float]], n: int, *, seed: int = 0,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append({k: float(rng.uniform(lo, hi)) for k, (lo, hi) in space.items()})
    return out


__all__ = ["grid_sweep", "random_sweep"]

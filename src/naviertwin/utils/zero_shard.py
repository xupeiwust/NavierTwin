"""ZeRO-style optimizer state shard (toy) — partition param tensors among ranks.

Examples:
    >>> import numpy as np
    >>> from naviertwin.utils.zero_shard import shard_state
    >>> state = {'p1': np.ones(10), 'p2': np.zeros(20)}
    >>> shards = shard_state(state, n_ranks=2)
    >>> len(shards)
    2
"""

from __future__ import annotations

from typing import Any

import numpy as np


def shard_state(
    state: dict[str, Any], *, n_ranks: int = 2,
) -> list[dict[str, Any]]:
    """각 param tensor 를 n_ranks 등분 (마지막 rank 가 잔여)."""
    shards: list[dict[str, Any]] = [{} for _ in range(n_ranks)]
    for k, v in state.items():
        v = np.asarray(v)
        chunks = np.array_split(v, n_ranks, axis=0)
        for r, c in enumerate(chunks):
            shards[r][k] = c
    return shards


def gather_state(shards: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    keys = shards[0].keys()
    out = {}
    for k in keys:
        out[k] = np.concatenate([s[k] for s in shards], axis=0)
    return out


__all__ = ["gather_state", "shard_state"]

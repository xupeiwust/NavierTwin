"""Edge collapse — 가장 짧은 edge 부터 점진적 합치기 (mesh decimation lite).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.tools.edge_collapse import edge_collapse_once
    >>> v = np.array([[0., 0], [0.1, 0], [1., 0], [0.5, 1]])
    >>> tri = np.array([[0, 1, 3], [1, 2, 3]])
    >>> v2, tri2 = edge_collapse_once(v, tri)
    >>> len(v2) <= len(v)
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def edge_collapse_once(
    verts: NDArray[np.float64], triangles: NDArray[np.int_],
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """가장 짧은 edge 를 midpoint 로 합치기 (1회)."""
    v = np.asarray(verts, dtype=np.float64)
    tri = np.asarray(triangles)
    edges = set()
    for t in tri:
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            edges.add(tuple(sorted((int(t[a]), int(t[b])))))
    edge_list = list(edges)
    lens = [np.linalg.norm(v[a] - v[b]) for a, b in edge_list]
    idx = int(np.argmin(lens))
    a, b = edge_list[idx]
    # collapse b → a, midpoint
    v_new = v.copy()
    v_new[a] = 0.5 * (v[a] + v[b])
    # remap b to a
    remap = np.arange(len(v))
    remap[b] = a
    tri_new = remap[tri]
    # drop degenerate triangles (any two equal)
    keep = np.array([
        len({int(t[0]), int(t[1]), int(t[2])}) == 3 for t in tri_new
    ])
    tri_new = tri_new[keep]
    # remove unused vertex b
    used = np.zeros(len(v_new), dtype=bool)
    used[tri_new.ravel()] = True
    keep_v = np.where(used)[0]
    new_idx = -np.ones(len(v_new), dtype=int)
    new_idx[keep_v] = np.arange(len(keep_v))
    return v_new[keep_v], new_idx[tri_new]


__all__ = ["edge_collapse_once"]

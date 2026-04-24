"""Zarr chunked array reader — zarr optional dependency.

Examples:
    >>> from naviertwin.core.cfd_reader.zarr_reader import has_zarr
    >>> isinstance(has_zarr(), bool)
    True
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def has_zarr() -> bool:
    try:
        import zarr  # noqa: F401
    except ImportError:
        return False
    return True


def read_zarr(path: str | Path) -> dict[str, Any]:
    """zarr group → dict[name, ndarray]."""
    if not has_zarr():
        raise ImportError("zarr not installed; pip install zarr")
    import numpy as np
    import zarr
    g = zarr.open(str(path), mode="r")
    return {k: np.asarray(g[k][:]) for k in g.array_keys()}


def write_zarr(path: str | Path, data: dict[str, Any]) -> None:
    if not has_zarr():
        raise ImportError("zarr not installed; pip install zarr")
    import numpy as np
    import zarr
    g = zarr.open(str(path), mode="w")
    for k, v in data.items():
        g.create_dataset(k, data=np.asarray(v))


__all__ = ["has_zarr", "read_zarr", "write_zarr"]

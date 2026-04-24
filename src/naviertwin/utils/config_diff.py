"""Nested dict diff — 두 config 간 변경사항 비교.

Examples:
    >>> from naviertwin.utils.config_diff import diff_configs
    >>> a = {"lr": 1e-3, "model": {"layers": 3}}
    >>> b = {"lr": 5e-4, "model": {"layers": 3, "dropout": 0.1}}
    >>> d = diff_configs(a, b)
    >>> "lr" in d["changed"]
    True
"""

from __future__ import annotations

from typing import Any


def diff_configs(a: dict, b: dict, path: str = "") -> dict[str, dict[str, Any]]:
    changed: dict[str, tuple[Any, Any]] = {}
    added: dict[str, Any] = {}
    removed: dict[str, Any] = {}
    for k in a.keys() | b.keys():
        p = f"{path}.{k}" if path else k
        if k in a and k not in b:
            removed[p] = a[k]
        elif k not in a and k in b:
            added[p] = b[k]
        else:
            va, vb = a[k], b[k]
            if isinstance(va, dict) and isinstance(vb, dict):
                sub = diff_configs(va, vb, p)
                changed.update(sub["changed"])
                added.update(sub["added"])
                removed.update(sub["removed"])
            elif va != vb:
                changed[p] = (va, vb)
    return {"changed": changed, "added": added, "removed": removed}


def format_diff(d: dict[str, dict]) -> str:
    lines: list[str] = []
    for p, (a, b) in d["changed"].items():
        lines.append(f"~ {p}: {a!r} -> {b!r}")
    for p, v in d["added"].items():
        lines.append(f"+ {p}: {v!r}")
    for p, v in d["removed"].items():
        lines.append(f"- {p}: {v!r}")
    return "\n".join(lines)


__all__ = ["diff_configs", "format_diff"]

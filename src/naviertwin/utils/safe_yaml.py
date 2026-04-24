"""YAML safe load/dump — PyYAML 있으면 사용, 없으면 stdlib INI-lite parser 로 fallback.

Examples:
    >>> from naviertwin.utils.safe_yaml import safe_yaml_load
    >>> safe_yaml_load("a: 1\\nb: 2\\n")
    {'a': 1, 'b': 2}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _parse_scalar(s: str) -> Any:
    s = s.strip()
    if s == "" or s in ("null", "~", "None"):
        return None
    if s.lower() in ("true", "yes", "on"):
        return True
    if s.lower() in ("false", "no", "off"):
        return False
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s.strip('"').strip("'")


def _fallback_load(text: str) -> dict[str, Any]:
    """매우 단순한 YAML: top-level key: value 만."""
    out: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = _parse_scalar(v)
    return out


def safe_yaml_load(text: str) -> Any:
    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:
        return _fallback_load(text)


def safe_yaml_load_file(path: str | Path) -> Any:
    return safe_yaml_load(Path(path).read_text(encoding="utf-8"))


def safe_yaml_dump(data: Any) -> str:
    try:
        import yaml

        return yaml.safe_dump(data, allow_unicode=True)
    except ImportError:
        # naive dumper
        if not isinstance(data, dict):
            return repr(data)
        lines = []
        for k, v in data.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines) + "\n"


__all__ = ["safe_yaml_load", "safe_yaml_load_file", "safe_yaml_dump"]

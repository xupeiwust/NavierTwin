"""Artifact zipper + manifest.

Examples:
    >>> import tempfile, pathlib
    >>> from naviertwin.utils.workflow.artifact_zip import zip_artifacts
    >>> with tempfile.TemporaryDirectory() as d:
    ...     p = pathlib.Path(d) / 'f.txt'
    ...     p.write_text('hi')
    ...     out = zip_artifacts([p], pathlib.Path(d) / 'a.zip')
"""

from __future__ import annotations

import json
import zipfile
from hashlib import sha256
from pathlib import Path
from typing import Any


def zip_artifacts(
    files: list[str | Path], out_path: str | Path,
    *,
    manifest_name: str = "MANIFEST.json",
    extra_entries: dict[str, str | bytes] | None = None,
) -> Path:
    """Zip artifacts and write an integrity manifest.

    Manifest entries contain:
    - ``name``: archived filename
    - ``bytes``: file size in bytes
    - ``sha256``: SHA-256 of archived bytes
    """
    out = Path(out_path)
    manifest: list[dict[str, Any]] = []
    seen: set[str] = set()
    with zipfile.ZipFile(out, "w") as zf:
        for f in files:
            fp = Path(f)
            if fp.name == manifest_name or fp.name in seen:
                raise ValueError(f"duplicate or reserved archive entry: {fp.name}")
            seen.add(fp.name)
            data = fp.read_bytes()
            zf.writestr(fp.name, data)
            manifest.append(
                {
                    "name": fp.name,
                    "bytes": len(data),
                    "sha256": sha256(data).hexdigest(),
                }
            )
        for name in sorted(extra_entries or {}):
            if name == manifest_name or name in seen:
                raise ValueError(f"duplicate or reserved archive entry: {name}")
            seen.add(name)
            payload = (extra_entries or {})[name]
            data = payload.encode("utf-8") if isinstance(payload, str) else payload
            zf.writestr(name, data)
            manifest.append(
                {
                    "name": name,
                    "bytes": len(data),
                    "sha256": sha256(data).hexdigest(),
                }
            )
        zf.writestr(manifest_name, json.dumps(manifest, indent=2, sort_keys=True))
    return out


def read_manifest(
    zip_path: str | Path,
    *,
    manifest_name: str = "MANIFEST.json",
) -> list[dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read(manifest_name))


__all__ = ["read_manifest", "zip_artifacts"]

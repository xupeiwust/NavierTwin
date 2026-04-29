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
    *, manifest_name: str = "MANIFEST.json",
) -> Path:
    """Zip artifacts and write an integrity manifest.

    Manifest entries contain:
    - ``name``: archived filename
    - ``bytes``: file size in bytes
    - ``sha256``: SHA-256 of archived bytes
    """
    out = Path(out_path)
    manifest: list[dict[str, Any]] = []
    with zipfile.ZipFile(out, "w") as zf:
        for f in files:
            fp = Path(f)
            data = fp.read_bytes()
            zf.writestr(fp.name, data)
            manifest.append(
                {
                    "name": fp.name,
                    "bytes": len(data),
                    "sha256": sha256(data).hexdigest(),
                }
            )
        zf.writestr(manifest_name, json.dumps(manifest, indent=2))
    return out


def read_manifest(
    zip_path: str | Path,
    *,
    manifest_name: str = "MANIFEST.json",
) -> list[dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read(manifest_name))


__all__ = ["read_manifest", "zip_artifacts"]

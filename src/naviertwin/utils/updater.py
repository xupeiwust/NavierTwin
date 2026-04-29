"""Offline update metadata contract for release channels."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from naviertwin import __version__

SUPPORTED_CHANNELS = {"stable", "beta", "nightly"}
WINDOWS_INSTALLER_NAME = "NavierTwinSetup.exe"


@dataclass(frozen=True)
class ReleaseMetadata:
    """Validated release metadata loaded from a GitHub Releases mirror."""

    version: str
    channel: str
    url: str
    sha256: str
    notes: str = ""


@dataclass(frozen=True)
class UpdateCheckResult:
    """Deterministic update-check result for CLI and GUI callers."""

    current_version: str
    latest_version: str
    channel: str
    update_available: bool
    url: str
    sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _version_key(version: str) -> tuple[int, ...]:
    """Convert a numeric release version into a comparable tuple."""
    parts = version.strip().split(".")
    if not parts or not all(part.isdigit() for part in parts):
        raise ValueError(f"numeric dotted version required: {version!r}")
    return tuple(int(part) for part in parts)


def is_newer_version(candidate: str, current: str = __version__) -> bool:
    """Return whether ``candidate`` is newer than ``current``."""
    left = _version_key(candidate)
    right = _version_key(current)
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) > right + (0,) * (width - len(right))


def _validate_release_url(url: str, version: str) -> None:
    """Validate that release metadata points at the expected versioned artifact."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("release metadata url must be an https URL")

    expected_tag = f"/releases/download/v{version}/"
    if expected_tag not in parsed.path:
        raise ValueError(f"release metadata url tag must match version v{version}")

    filename = PurePosixPath(parsed.path).name
    if filename != WINDOWS_INSTALLER_NAME:
        raise ValueError(f"release metadata url must point to {WINDOWS_INSTALLER_NAME}")


def load_release_metadata(path: Path) -> ReleaseMetadata:
    """Load and validate local release metadata JSON.

    Expected schema:
        ``{"version": "4.2.59", "channel": "stable", "url": "...", "sha256": "..."}``
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("release metadata must be a JSON object")

    version = str(data.get("version", "")).strip()
    channel = str(data.get("channel", "")).strip()
    url = str(data.get("url", "")).strip()
    sha256 = str(data.get("sha256", "")).strip().lower()
    notes = str(data.get("notes", "")).strip()

    if channel not in SUPPORTED_CHANNELS:
        raise ValueError(f"unsupported release channel: {channel!r}")
    if not version:
        raise ValueError("release metadata requires version")
    _version_key(version)
    if not url:
        raise ValueError("release metadata requires url")
    _validate_release_url(url, version)
    if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError("release metadata requires a 64-character sha256 hex digest")

    return ReleaseMetadata(
        version=version,
        channel=channel,
        url=url,
        sha256=sha256,
        notes=notes,
    )


def check_for_update(
    metadata_path: Path,
    *,
    current_version: str = __version__,
    channel: str = "stable",
) -> UpdateCheckResult:
    """Evaluate a local release metadata file against the current version."""
    metadata = load_release_metadata(metadata_path)
    if channel not in SUPPORTED_CHANNELS:
        raise ValueError(f"unsupported release channel: {channel!r}")
    if metadata.channel != channel:
        return UpdateCheckResult(
            current_version=current_version,
            latest_version=current_version,
            channel=channel,
            update_available=False,
            url="",
            sha256="",
        )

    return UpdateCheckResult(
        current_version=current_version,
        latest_version=metadata.version,
        channel=channel,
        update_available=is_newer_version(metadata.version, current_version),
        url=metadata.url,
        sha256=metadata.sha256,
    )


__all__ = [
    "ReleaseMetadata",
    "UpdateCheckResult",
    "check_for_update",
    "is_newer_version",
    "load_release_metadata",
]

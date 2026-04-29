"""Fast deterministic smoke checks for Windows installer contracts."""

from __future__ import annotations

import json
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _assert_contains(text: str, needle: str, *, message: str) -> int:
    if needle not in text:
        raise AssertionError(message)
    return 1


def main() -> int:
    root = _repo_root()
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = str(pyproject["project"]["version"])

    spec = (root / "installer" / "naviertwin.spec").read_text(encoding="utf-8")
    iss = (root / "installer" / "naviertwin.iss").read_text(encoding="utf-8")

    checks = 0

    checks += _assert_contains(
        iss,
        f"AppVersion={version}",
        message="Inno AppVersion must match project.version",
    )

    checks += _assert_contains(
        spec,
        "ROOT = Path(SPECPATH).parent.parent",
        message="PyInstaller spec must derive ROOT from SPECPATH",
    )
    checks += _assert_contains(
        spec,
        'name="NavierTwin"',
        message='PyInstaller EXE/COLLECT name must be "NavierTwin"',
    )
    checks += _assert_contains(
        spec,
        'distpath=str(ROOT / "dist")',
        message='PyInstaller distpath must be str(ROOT / "dist")',
    )
    checks += _assert_contains(
        spec,
        "naviertwin/gui/styles/i18n",
        message="PyInstaller spec must include GUI i18n runtime assets",
    )
    checks += _assert_contains(
        spec,
        "naviertwin.gui.panels.postproc_panel",
        message="PyInstaller spec must include postproc panel hidden import",
    )
    checks += _assert_contains(
        spec,
        "naviertwin.core.post_process_facade",
        message="PyInstaller spec must include postproc facade hidden import",
    )

    checks += _assert_contains(
        iss,
        'Source: "..\\dist\\NavierTwin\\*"',
        message="Inno must package dist/NavierTwin output directory",
    )
    checks += _assert_contains(
        iss,
        "NavierTwin.exe",
        message="Inno must reference NavierTwin.exe",
    )
    checks += _assert_contains(
        iss,
        "OutputBaseFilename=NavierTwinSetup",
        message="Inno OutputBaseFilename must be NavierTwinSetup",
    )
    checks += _assert_contains(
        iss,
        "DefaultDirName={autopf}\\NavierTwin",
        message="Inno DefaultDirName must target Program Files/NavierTwin",
    )
    checks += _assert_contains(
        iss,
        "LicenseFile=..\\LICENSE",
        message="Inno must reference repository LICENSE",
    )
    checks += _assert_contains(
        iss,
        "UninstallDisplayIcon={app}\\NavierTwin.exe",
        message="Inno uninstall icon must point to NavierTwin.exe",
    )
    checks += _assert_contains(
        iss,
        "SetupIconFile=",
        message="Inno SetupIconFile key must exist (may be intentionally blank)",
    )
    checks += _assert_contains(
        iss,
        "AppPublisher=",
        message="Inno AppPublisher key must exist",
    )
    checks += _assert_contains(
        iss,
        "AppPublisherURL=",
        message="Inno AppPublisherURL key must exist",
    )

    publisher = next(
        (line.split("=", 1)[1].strip() for line in iss.splitlines() if line.startswith("AppPublisher=")),
        "",
    )
    publisher_url = next(
        (
            line.split("=", 1)[1].strip()
            for line in iss.splitlines()
            if line.startswith("AppPublisherURL=")
        ),
        "",
    )
    if not publisher:
        raise AssertionError("Inno AppPublisher must be non-empty")
    checks += 1
    if not publisher_url:
        raise AssertionError("Inno AppPublisherURL must be non-empty")
    checks += 1

    dist_dir = root / "dist" / "NavierTwin"
    if dist_dir.exists():
        if not (dist_dir / "NavierTwin.exe").exists():
            raise AssertionError("dist/NavierTwin exists but NavierTwin.exe is missing")
        checks += 1

    print(json.dumps({"status": "ok", "checks": checks}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

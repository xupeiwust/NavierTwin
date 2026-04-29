"""Windows installer/PyInstaller packaging metadata regression tests."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


def _project_version() -> str:
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def test_inno_setup_version_matches_pyproject() -> None:
    """Inno Setup AppVersion must track the Python package version."""
    root = Path(__file__).resolve().parents[1]
    iss = (root / "installer" / "naviertwin.iss").read_text(encoding="utf-8")

    assert f"AppVersion={_project_version()}" in iss
    assert "AppVersion=4.2.0" not in iss


def test_inno_setup_matches_pyinstaller_output_layout() -> None:
    """Inno source path and executable names must match PyInstaller output."""
    root = Path(__file__).resolve().parents[1]
    iss = (root / "installer" / "naviertwin.iss").read_text(encoding="utf-8")
    spec = (root / "installer" / "naviertwin.spec").read_text(encoding="utf-8")

    assert 'name="NavierTwin"' in spec
    assert 'distpath=str(ROOT / "dist")' in spec
    assert 'Source: "..\\dist\\NavierTwin\\*"' in iss
    assert "NavierTwin.exe" in iss
    assert "OutputBaseFilename=NavierTwinSetup" in iss
    assert "DefaultDirName={autopf}\\NavierTwin" in iss
    assert "LicenseFile=..\\LICENSE" in iss
    assert "UninstallDisplayIcon={app}\\NavierTwin.exe" in iss
    assert "SetupIconFile=" in iss


def test_pyinstaller_spec_uses_project_root_and_runtime_assets() -> None:
    """PyInstaller spec must resolve repo root and include GUI runtime assets."""
    root = Path(__file__).resolve().parents[1]
    spec = (root / "installer" / "naviertwin.spec").read_text(encoding="utf-8")

    assert "ROOT = Path(SPECPATH).parent.parent" in spec
    assert "gui/styles/i18n" in spec
    assert "naviertwin.gui.panels.postproc_panel" in spec
    assert "naviertwin.core.post_process_facade" in spec


def test_inno_publisher_fields_are_present_and_non_empty() -> None:
    """Inno publisher metadata must be defined for installer identity links."""
    root = Path(__file__).resolve().parents[1]
    iss = (root / "installer" / "naviertwin.iss").read_text(encoding="utf-8")

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
    assert publisher
    assert publisher_url

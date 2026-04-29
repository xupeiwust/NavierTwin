"""Offline updater metadata contract tests."""

from __future__ import annotations

import json

import pytest


def _metadata(
    version: str = "4.2.59",
    channel: str = "stable",
    url: str = "https://github.com/naviertwin/naviertwin/releases/download/v4.2.59/NavierTwinSetup.exe",
) -> dict[str, str]:
    return {
        "version": version,
        "channel": channel,
        "url": url,
        "sha256": "a" * 64,
        "notes": "smoke",
    }


def test_update_metadata_detects_newer_version(tmp_path) -> None:
    from naviertwin.utils.updater import check_for_update

    path = tmp_path / "release.json"
    path.write_text(json.dumps(_metadata()), encoding="utf-8")

    result = check_for_update(path, current_version="4.2.58", channel="stable")

    assert result.update_available is True
    assert result.latest_version == "4.2.59"
    assert result.sha256 == "a" * 64


def test_update_metadata_rejects_invalid_integrity_hash(tmp_path) -> None:
    from naviertwin.utils.updater import load_release_metadata

    payload = _metadata()
    payload["sha256"] = "not-a-hash"
    path = tmp_path / "release.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="sha256"):
        load_release_metadata(path)


def test_update_metadata_rejects_url_version_mismatch(tmp_path) -> None:
    from naviertwin.utils.updater import load_release_metadata

    payload = _metadata(
        version="4.2.60",
        url="https://github.com/naviertwin/naviertwin/releases/download/v4.2.59/NavierTwinSetup.exe",
    )
    path = tmp_path / "release.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="url tag"):
        load_release_metadata(path)


def test_update_metadata_rejects_unexpected_installer_name(tmp_path) -> None:
    from naviertwin.utils.updater import load_release_metadata

    payload = _metadata(
        url="https://github.com/naviertwin/naviertwin/releases/download/v4.2.59/OtherSetup.exe",
    )
    path = tmp_path / "release.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="NavierTwinSetup.exe"):
        load_release_metadata(path)


def test_update_check_channel_mismatch_is_not_update(tmp_path) -> None:
    from naviertwin.utils.updater import check_for_update

    path = tmp_path / "release.json"
    path.write_text(json.dumps(_metadata(channel="beta")), encoding="utf-8")

    result = check_for_update(path, current_version="4.2.58", channel="stable")

    assert result.update_available is False
    assert result.latest_version == "4.2.58"
    assert result.url == ""


@pytest.mark.parametrize("latest", ["4.2.57", "4.2.58"])
def test_update_check_equal_or_older_version_is_not_update(tmp_path, latest) -> None:
    from naviertwin.utils.updater import check_for_update

    path = tmp_path / "release.json"
    path.write_text(
        json.dumps(
            _metadata(
                version=latest,
                url=(
                    "https://github.com/naviertwin/naviertwin/releases/"
                    f"download/v{latest}/NavierTwinSetup.exe"
                ),
            )
        ),
        encoding="utf-8",
    )

    result = check_for_update(path, current_version="4.2.58", channel="stable")

    assert result.update_available is False
    assert result.latest_version == latest
    assert result.url.endswith("/NavierTwinSetup.exe")


def test_update_check_cli_outputs_json(tmp_path, capsys) -> None:
    from naviertwin.main import _build_parser, _run_update_check

    path = tmp_path / "release.json"
    path.write_text(json.dumps(_metadata()), encoding="utf-8")
    args = _build_parser().parse_args(["update-check", "--metadata", str(path)])

    code = _run_update_check(
        metadata=args.metadata,
        channel=args.channel,
        current_version="4.2.58",
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["update_available"] is True
    assert output["channel"] == "stable"


def test_update_check_cli_reports_invalid_metadata_without_traceback(tmp_path, capsys) -> None:
    from naviertwin.main import _run_update_check

    missing = tmp_path / "missing-release.json"
    missing_code = _run_update_check(
        metadata=str(missing),
        channel="stable",
        current_version="4.2.58",
    )
    missing_output = capsys.readouterr()

    assert missing_code == 2
    assert missing_output.out == ""
    assert "update-check error:" in missing_output.err
    assert "Traceback" not in missing_output.err

    invalid = tmp_path / "invalid-release.json"
    payload = _metadata()
    payload["sha256"] = "invalid"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    invalid_code = _run_update_check(
        metadata=str(invalid),
        channel="stable",
        current_version="4.2.58",
    )
    invalid_output = capsys.readouterr()

    assert invalid_code == 2
    assert invalid_output.out == ""
    assert "update-check error:" in invalid_output.err
    assert "Traceback" not in invalid_output.err


def test_release_metadata_example_is_valid() -> None:
    from pathlib import Path

    from naviertwin.utils.updater import check_for_update, load_release_metadata

    root = Path(__file__).resolve().parents[1]
    example = root / "examples" / "release-metadata.example.json"

    metadata = load_release_metadata(example)
    result = check_for_update(example, current_version="4.2.58")

    assert metadata.channel == "stable"
    assert result.update_available is True

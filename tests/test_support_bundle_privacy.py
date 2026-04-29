"""Support bundle privacy hardening tests."""

from __future__ import annotations

import json
import zipfile


def _assert_secret_absent(text: str) -> None:
    assert "secret123" not in text
    assert "Bearer tok.abc.123" not in text
    assert "API_KEY=abcd1234" not in text


def test_support_bundle_redacts_sensitive_values(tmp_path, monkeypatch) -> None:
    from naviertwin.core.validation import dataset_preflight
    from naviertwin.utils import doctor
    from naviertwin.utils.support_bundle import build_support_bundle

    monkeypatch.setenv("NAVIER_TWIN_TEST_TOKEN", "secret123")

    monkeypatch.setattr(
        doctor,
        "build_doctor_report",
        lambda include_optional=False: {
            "status": "ok",
            "version": "x.y.z",
            "environment": {
                "seeded": "NAVIER_TWIN_TEST_TOKEN=secret123",
                "authorization": "Bearer tok.abc.123",
            },
            "checks": [],
            "warnings": ["token=secret123"],
            "errors": [],
        },
    )
    monkeypatch.setattr(
        dataset_preflight,
        "build_dataset_preflight_report",
        lambda path: {
            "status": "ok",
            "path": str(path),
            "checks": [{"name": "ok", "status": "ok", "details": {"api_key": "abcd1234"}}],
            "summary": {"metadata": {"password": "secret123"}},
            "warnings": ["API_KEY=abcd1234"],
            "errors": [],
        },
    )

    outdir = tmp_path / "support"
    metadata = build_support_bundle(
        outdir=outdir,
        preflight=tmp_path / "input_SECRET_TOKEN=secret123.su2",
        zip_bundle=True,
    )

    doctor_text = (outdir / "doctor.json").read_text(encoding="utf-8")
    preflight_text = (outdir / "preflight.json").read_text(encoding="utf-8")
    metadata_text = (outdir / "metadata.json").read_text(encoding="utf-8")
    _assert_secret_absent(doctor_text)
    _assert_secret_absent(preflight_text)
    _assert_secret_absent(metadata_text)

    assert "TOKEN=***REDACTED***" in doctor_text
    assert '"authorization": "***REDACTED***"' in doctor_text

    with zipfile.ZipFile(outdir / "support-bundle.zip") as zf:
        for name in ("doctor.json", "preflight.json", "metadata.json"):
            text = zf.read(name).decode("utf-8")
            _assert_secret_absent(text)

    encoded = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    _assert_secret_absent(encoded)
    assert isinstance(metadata.get("inputs"), dict)
    assert metadata["inputs"]["preflight"] != str(tmp_path / "input_SECRET_TOKEN=secret123.su2")

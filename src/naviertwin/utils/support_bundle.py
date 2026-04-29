"""Support bundle builder for customer diagnostics."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from naviertwin import __version__
from naviertwin.utils.secret_redact import redact_object


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(_json_bytes(payload))


def _artifact_integrity(path: Path, expected_payload: dict[str, Any]) -> dict[str, Any]:
    expected = _json_bytes(expected_payload)
    actual = path.read_bytes()
    if actual != expected:
        raise RuntimeError(f"support bundle artifact verification failed: {path.name}")
    return {
        "bytes": len(actual),
        "sha256": sha256(actual).hexdigest(),
    }


def report_to_json(report: dict[str, Any]) -> str:
    """Serialize support bundle metadata to stable JSON."""
    return json.dumps(report, ensure_ascii=False, sort_keys=True)


def _final_status(statuses: list[str]) -> str:
    if "error" in statuses:
        return "error"
    if "warn" in statuses:
        return "warn"
    return "ok"


def build_support_bundle(
    outdir: str | Path,
    preflight: str | Path | None = None,
    include_optional: bool = False,
    zip_bundle: bool = False,
) -> dict[str, Any]:
    """Build a support bundle with doctor/preflight reports and metadata."""
    from naviertwin.utils.doctor import build_doctor_report
    from naviertwin.utils.doctor import report_to_json as doctor_to_json
    from naviertwin.utils.workflow.artifact_zip import zip_artifacts

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[str] = []
    artifacts: dict[str, dict[str, Any]] = {}
    statuses: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    doctor_report = build_doctor_report(include_optional=include_optional)
    doctor_payload = redact_object(json.loads(doctor_to_json(doctor_report)))
    doctor_path = output_dir / "doctor.json"
    _write_json(doctor_path, doctor_payload)
    written_files.append(doctor_path.name)
    artifacts[doctor_path.name] = _artifact_integrity(doctor_path, doctor_payload)
    statuses.append(str(doctor_payload.get("status", "error")))
    warnings.extend(str(item) for item in doctor_payload.get("warnings", []))
    errors.extend(str(item) for item in doctor_payload.get("errors", []))

    if preflight is not None:
        from naviertwin.core.validation.dataset_preflight import (
            build_dataset_preflight_report,
        )
        from naviertwin.core.validation.dataset_preflight import (
            report_to_json as preflight_to_json,
        )

        preflight_report = build_dataset_preflight_report(preflight)
        preflight_payload = redact_object(json.loads(preflight_to_json(preflight_report)))
        preflight_path = output_dir / "preflight.json"
        _write_json(preflight_path, preflight_payload)
        written_files.append(preflight_path.name)
        artifacts[preflight_path.name] = _artifact_integrity(preflight_path, preflight_payload)
        statuses.append(str(preflight_payload.get("status", "error")))
        warnings.extend(str(item) for item in preflight_payload.get("warnings", []))
        errors.extend(str(item) for item in preflight_payload.get("errors", []))

    files = [*written_files, "metadata.json"]
    metadata: dict[str, Any] = {
        "status": _final_status(statuses),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": __version__,
        "files": files,
        "artifacts": artifacts,
        "inputs": {
            "preflight": None if preflight is None else str(Path(preflight)),
            "include_optional": include_optional,
        },
        "warnings": warnings,
        "errors": errors,
    }

    if zip_bundle:
        zip_path = output_dir / "support-bundle.zip"
        metadata["zip_path"] = str(zip_path)
    metadata = redact_object(metadata)

    metadata_path = output_dir / "metadata.json"
    _write_json(metadata_path, metadata)
    _artifact_integrity(metadata_path, metadata)

    if zip_bundle:
        zip_artifacts([output_dir / name for name in files], zip_path)

    return metadata


__all__ = ["build_support_bundle", "report_to_json"]

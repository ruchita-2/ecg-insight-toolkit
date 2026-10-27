"""Deterministic report persistence."""

import json
import os
import tempfile
from pathlib import Path


def validate_report(report: dict) -> list[str]:
    missing = [
        key for key in ("dataset_version", "split_sha256", "model_id")
        if not report.get(key)
    ]
    if any(row.get("support", -1) < 0 for row in report.get("metrics", [])):
        missing.append("metrics.support")
    return missing


def write_report(path: Path, payload: dict) -> None:
    if "dataset_version" in payload or "split_sha256" in payload:
        invalid = validate_report(payload)
        if invalid:
            raise ValueError(f"report missing required evidence: {', '.join(invalid)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

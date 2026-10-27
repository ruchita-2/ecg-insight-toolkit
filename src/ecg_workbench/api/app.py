"""Artifact-backed HTTP API; run IDs and record IDs are catalog constrained."""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ecg_workbench.api.schemas import WaveformWindow
from ecg_workbench.data.records import ECGRecord, load_record
from ecg_workbench.eval.reports import validate_report

SAFE_ID = re.compile(r"[A-Za-z0-9_-]+\Z")


class RunCatalog:
    def __init__(self, artifact_root: Path, data_root: Path):
        self.artifact_root = artifact_root
        self.data_root = data_root

    def _path(self, run_id: str) -> Path:
        if not SAFE_ID.fullmatch(run_id) or not run_id.startswith("end-to-end-"):
            raise HTTPException(404, "unknown run")
        path = self.artifact_root / f"{run_id}.json"
        if not path.is_file():
            raise HTTPException(404, "unknown run")
        return path

    def report(self, run_id: str) -> dict:
        report = json.loads(self._path(run_id).read_text())
        if (
            report.get("dataset") != "mitdb"
            or not isinstance(report.get("records"), dict)
            or validate_report(report)
        ):
            raise HTTPException(500, "invalid run artifact")
        return report

    def require_record(self, run_id: str, record_id: str) -> ECGRecord:
        report = self.report(run_id)
        if not SAFE_ID.fullmatch(record_id) or record_id not in report["records"]:
            raise HTTPException(404, "unknown record")
        return load_record(self.data_root, record_id)

    def associated_report(self, filename: str) -> dict | None:
        path = self.artifact_root / filename
        return json.loads(path.read_text()) if path.is_file() else None


def create_app(catalog: RunCatalog) -> FastAPI:
    api = FastAPI(title="ECG Beat Analysis Workbench", version="0.1.0")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET"], allow_headers=["*"],
    )

    @api.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/api/runs")
    def list_runs() -> list[dict]:
        return [
            {"run_id": path.stem, "model_id": report.get("model_id"),
             "dataset_version": report.get("dataset_version"),
             "split_sha256": report.get("split_sha256")}
            for path in sorted(catalog.artifact_root.glob("end-to-end-*.json"), reverse=True)
            for report in [catalog.report(path.stem)]
        ]

    @api.get("/api/runs/{run_id}/summary")
    def run_summary(run_id: str) -> dict:
        report = catalog.report(run_id)
        version = run_id.rsplit("-", 1)[-1]
        if not re.fullmatch(r"v[0-9]+", version):
            version = "v1"
        model_version = report["model_id"].rsplit("-", 1)[-1]
        if not re.fullmatch(r"v[0-9]+", model_version):
            model_version = version
        return {
            "run_id": run_id, "dataset": report["dataset"],
            "dataset_version": report["dataset_version"],
            "split_sha256": report["split_sha256"],
            "model_id": report["model_id"], "detector_id": report["detector_id"],
            "pooled": report["pooled"],
            "reliability": report.get("reliability"),
            "records": list(report["records"]),
            "detection": catalog.associated_report(
                f"detection-{'v3' if version == 'v4' else version}.json"
            )
            or catalog.associated_report("detection-v1.json"),
            "oracle_feature": catalog.associated_report(f"oracle-feature-{model_version}.json"),
            "oracle_waveform": catalog.associated_report("oracle-waveform-v1.json"),
            "noise": catalog.associated_report(f"noise-{version}.json"),
        }

    @api.get("/api/runs/{run_id}/records/{record_id}")
    def record_results(run_id: str, record_id: str) -> dict:
        report = catalog.report(run_id)
        if not SAFE_ID.fullmatch(record_id) or record_id not in report["records"]:
            raise HTTPException(404, "unknown record")
        return {
            "run_id": run_id, "record_id": record_id,
            "dataset_version": report["dataset_version"],
            "split_sha256": report["split_sha256"],
            "model_id": report["model_id"],
            **report["records"][record_id],
        }

    @api.get("/api/runs/{run_id}/records/{record_id}/window")
    def waveform_window(
        run_id: str, record_id: str,
        start: float = Query(ge=0), duration: float = Query(gt=0, le=10),
    ) -> WaveformWindow:
        record = catalog.require_record(run_id, record_id)
        left = round(start * record.fs)
        right = min(len(record.signals), round((start + duration) * record.fs))
        if right <= left:
            raise HTTPException(422, "window starts beyond recording")
        return WaveformWindow.from_record(record, left, right)

    return api


app = create_app(RunCatalog(Path("artifacts/runs"), Path("data/mitdb/1.0.0")))

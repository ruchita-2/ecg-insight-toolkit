"""Locked NSTDB paired noise stress benchmark."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import joblib
import numpy as np

from ecg_workbench.data.labels import CLASS_ORDER
from ecg_workbench.data.records import load_record
from ecg_workbench.data.splits import SplitManifest, validate_split
from ecg_workbench.eval.end_to_end import reference_truth
from ecg_workbench.eval.events import match_events
from ecg_workbench.eval.metrics import (
    detection_metrics,
    end_to_end_metrics,
    pooled_detection,
)
from ecg_workbench.models.reliability import TemperatureCalibrator
from ecg_workbench.pipeline import BeatPrediction, analyze
from ecg_workbench.signal.detectors import (
    detect_pan_tompkins,
    detect_pan_tompkins_gap_rescue,
)
from ecg_workbench.signal.quality import QualityScaler

SNR_VALUES = (-6, 0, 6, 12, 18, 24)
NSTDB_REQUIRED_FILES = [
    f"{source}e{suffix}.{extension}"
    for source in ("118", "119")
    for suffix in ("_6", "00", "06", "12", "18", "24")
    for extension in ("hea", "dat", "atr")
]
RECORD_PATTERN = re.compile(r"(118|119)e(_6|00|06|12|18|24)\Z")


def parse_noise_record(record_id: str) -> tuple[str, int]:
    match = RECORD_PATTERN.fullmatch(record_id)
    if match is None:
        raise ValueError(f"unsupported NSTDB record: {record_id}")
    return match.group(1), -6 if match.group(2) == "_6" else int(match.group(2))


def assert_noise_isolation(split: SplitManifest) -> None:
    for source in ("118", "119"):
        for partition in ("train", "validation", "test"):
            if source in getattr(split, partition):
                raise ValueError(f"noise source {source} appears in {partition}")
    validate_split(split)


@dataclass(frozen=True)
class FrozenPipeline:
    classifier: object
    calibrator: TemperatureCalibrator
    quality_scaler: QualityScaler
    abstention_threshold: float
    quality_floor: float
    model_id: str
    feature_version: str = "v1"
    detector_id: str = "pan_tompkins_style-v1"
    quality_version: str = "v1"
    morphology_block_limit: float | None = None
    severe_morphology_limit: float | None = None


def load_frozen_pipeline(model_path: Path, end_to_end_report: dict) -> FrozenPipeline:
    package = joblib.load(model_path)
    if package["metadata"]["split_sha256"] != end_to_end_report["split_sha256"]:
        raise ValueError("frozen model and pipeline split differ")
    scaler = end_to_end_report["quality_scaler"]
    return FrozenPipeline(
        classifier=package["model"],
        calibrator=TemperatureCalibrator(end_to_end_report["calibration"]["temperature"]),
        quality_scaler=QualityScaler(tuple(scaler["median"]), tuple(scaler["scale"])),
        abstention_threshold=end_to_end_report["abstention_threshold"],
        quality_floor=end_to_end_report["quality_floor"],
        model_id=package["metadata"]["model_id"],
        feature_version=package["metadata"].get("feature_version", "v1"),
        detector_id=end_to_end_report["detector_id"],
        quality_version=end_to_end_report.get("quality_version", "v1"),
        morphology_block_limit=end_to_end_report.get("morphology_block_limit"),
        severe_morphology_limit=end_to_end_report.get("severe_morphology_limit"),
    )


def interval_kind(seconds: float) -> str:
    if seconds < 300:
        return "clean"
    return "noisy" if int((seconds - 300) // 120) % 2 == 0 else "clean"


def _score_interval(
    truth: list[tuple[int, str | None]], predictions: list[BeatPrediction],
    fs: float, hours: float, quality_floor: float,
) -> dict:
    matched = match_events(
        np.array([sample for sample, _ in truth]),
        np.array([row.sample for row in predictions]), round(0.15 * fs),
    )
    return {
        "detection": {**detection_metrics(matched, hours, fs), "hours": hours},
        "classification": end_to_end_metrics(truth, predictions, fs),
        "quality_flag_rate": (
            sum(row.signal_quality < quality_floor for row in predictions) / len(predictions)
            if predictions else None
        ),
        "quality_flags": sum(row.signal_quality < quality_floor for row in predictions),
        "abstention_rate": (
            sum(row.abstained for row in predictions) / len(predictions)
            if predictions else None
        ),
    }


def noise_benchmark(root: Path, frozen_pipeline: FrozenPipeline, split_hash: str) -> dict:
    detector = (
        detect_pan_tompkins_gap_rescue if frozen_pipeline.detector_id == "pan_tompkins_gap_rescue-v3"
        else detect_pan_tompkins
    )
    rows = []
    for source in ("118", "119"):
        for snr in SNR_VALUES:
            suffix = "_6" if snr == -6 else f"{snr:02d}"
            record_id = f"{source}e{suffix}"
            record = load_record(root, record_id)
            predictions = analyze(
                record, detector, frozen_pipeline.classifier,
                frozen_pipeline.calibrator, frozen_pipeline.abstention_threshold,
                frozen_pipeline.quality_floor, frozen_pipeline.quality_scaler,
                frozen_pipeline.feature_version,
                frozen_pipeline.quality_version,
                frozen_pipeline.morphology_block_limit,
                frozen_pipeline.severe_morphology_limit,
            )
            truth = reference_truth(record)
            # Score contiguous intervals separately to avoid matching across a noise boundary.
            end_seconds = len(record.signals) / record.fs
            boundaries = [0.0, min(300.0, end_seconds)]
            while boundaries[-1] < end_seconds:
                boundaries.append(min(boundaries[-1] + 120.0, end_seconds))
            for left, right in pairwise(boundaries):
                kind = interval_kind(left + 0.001)
                interval_truth = [
                    item for item in truth if left <= item[0] / record.fs < right
                ]
                interval_predictions = [
                    row for row in predictions if left <= row.sample / record.fs < right
                ]
                rows.append({
                    "source": source, "record_id": record_id, "snr_db": snr,
                    "interval": kind, "start_s": left, "end_s": right,
                    **_score_interval(
                        interval_truth, interval_predictions, record.fs,
                        (right - left) / 3600, frozen_pipeline.quality_floor,
                    ),
                })
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["snr_db"], row["interval"])].append(row)
    summary = []
    for (source, snr, kind), members in sorted(grouped.items()):
        detection = pooled_detection([row["detection"] for row in members])
        predicted = sum(row["classification"]["predictions"] for row in members)
        abstained = sum(row["classification"]["abstentions"] for row in members)
        quality_flags = sum(row["quality_flags"] for row in members)
        classes = {}
        for label in CLASS_ORDER:
            support = sum(row["classification"]["classes"][label]["support"] for row in members)
            correct = sum(row["classification"]["classes"][label]["correct"] for row in members)
            predicted_class = sum(
                row["classification"]["classes"][label]["matched_predictions"]
                for row in members
            )
            classes[label] = {
                "support": support, "correct": correct,
                "matched_predictions": predicted_class,
                "recall": correct / support if support else None,
                "matched_precision": correct / predicted_class if predicted_class else None,
            }
        summary.append({
            "source": source, "snr_db": snr, "interval": kind,
            "detection": detection,
            "classes": classes,
            "beat_support": sum(
                row["classification"]["matched_beats"] + row["classification"]["missed_beats"]
                for row in members
            ),
            "abstention_rate": abstained / predicted if predicted else None,
            "quality_flag_rate": quality_flags / predicted if predicted else None,
            "predictions": predicted,
        })
    return {
        "dataset": "nstdb", "dataset_version": "1.0.0",
        "split_sha256": split_hash, "model_id": frozen_pipeline.model_id,
        "detector_id": frozen_pipeline.detector_id,
        "quality_version": frozen_pipeline.quality_version,
        "snr_values_db": list(SNR_VALUES),
        "source_records": ["118", "119"],
        "noise_schedule": "first 5 min clean, then alternating 2 min noisy and clean",
        "intervals": rows, "summary": summary,
        "independent_source_count": 2,
    }

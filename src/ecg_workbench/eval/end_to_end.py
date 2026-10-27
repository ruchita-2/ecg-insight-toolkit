"""Validation-frozen, continuous-recording benchmark and review rows."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from ecg_workbench.data.labels import CLASS_ORDER, group_beat, is_beat_annotation
from ecg_workbench.data.records import ECGRecord, load_record
from ecg_workbench.eval.events import match_events
from ecg_workbench.eval.metrics import end_to_end_metrics
from ecg_workbench.models.reliability import (
    TemperatureCalibrator,
    choose_abstention_threshold,
    fit_temperature,
    reliability_evidence,
)
from ecg_workbench.pipeline import BeatPrediction, analyze, heart_rate_series
from ecg_workbench.signal.detectors import (
    detect_local,
    detect_pan_tompkins,
    detect_pan_tompkins_gap_rescue,
)
from ecg_workbench.signal.features import (
    extract_features,
    extract_features_v2,
    extract_features_v3,
)
from ecg_workbench.signal.filters import preprocess
from ecg_workbench.signal.quality import (
    QualityScaler,
    estimate_quality,
    fit_quality_scaler,
    morphology_block_scores,
    quality_indicators,
)


def reference_truth(record: ECGRecord) -> list[tuple[int, str | None]]:
    return [
        (mark.sample, group_beat(mark.symbol))
        for mark in record.annotations if is_beat_annotation(mark.symbol)
    ]


def review_rows(record: ECGRecord, predictions: list[BeatPrediction]) -> list[dict]:
    truth = [mark for mark in record.annotations if is_beat_annotation(mark.symbol)]
    matches = match_events(
        np.array([mark.sample for mark in truth]),
        np.array([row.sample for row in predictions]), round(0.15 * record.fs),
    )
    rows = []
    for ri, pi, offset in matches.matched:
        reference = truth[ri]
        prediction = predictions[pi]
        label = group_beat(reference.symbol)
        kind = (
            "out_of_scope" if label is None else
            "abstained" if prediction.abstained else
            "correct" if prediction.group == label else "wrong_class"
        )
        rows.append({
            **prediction.to_dict(record.fs), "kind": kind,
            "reference_sample": reference.sample,
            "reference_symbol": reference.symbol,
            "reference_group": label, "timing_error_ms": offset / record.fs * 1000,
        })
    for ri in matches.unmatched_reference:
        reference = truth[ri]
        rows.append({
            "sample": reference.sample, "seconds": reference.sample / record.fs,
            "kind": "missed", "reference_sample": reference.sample,
            "reference_symbol": reference.symbol,
            "reference_group": group_beat(reference.symbol),
            "group": None, "score": None, "signal_quality": None,
            "abstained": False, "timing_error_ms": None,
            "abstention_reason": None,
        })
    for pi in matches.unmatched_predicted:
        prediction = predictions[pi]
        rows.append({
            **prediction.to_dict(record.fs), "kind": "extra",
            "reference_sample": None, "reference_symbol": None,
            "reference_group": None, "timing_error_ms": None,
        })
    return sorted(rows, key=lambda row: (row["sample"], row["kind"]))


def fit_reliability(
    root: Path, partitions: dict[str, list[str]], classifier,
    feature_version: str = "v1",
    detector=detect_pan_tompkins,
    quality_version: str = "v1",
) -> tuple[
    TemperatureCalibrator, float, float, QualityScaler, dict, float | None, float | None,
]:
    # Quality scaling uses development signals only, without labels.
    indicators = []
    morphology_blocks = []
    for record_id in partitions["train"]:
        record = load_record(root, record_id)
        raw = record.signals[:, 0]
        filtered = preprocess(raw, record.fs)
        all_peaks = detector(filtered, record.fs)
        peaks = all_peaks[::20]
        alternate = detect_local(filtered, record.fs)
        indicators.append(quality_indicators(
            raw, filtered, peaks, record.fs, alternate,
            dc_offset_correction=quality_version == "v3",
        ))
        if quality_version == "v3" and len(all_peaks):
            scores = morphology_block_scores(filtered, all_peaks, record.fs)
            blocks = all_peaks // round(30 * record.fs)
            _, first = np.unique(blocks, return_index=True)
            morphology_blocks.extend(scores[first].tolist())
    scaler = fit_quality_scaler(np.vstack(indicators))

    probabilities = []
    labels = []
    qualities = []
    for record_id in partitions["validation"]:
        record = load_record(root, record_id)
        selected = [
            (mark.sample, label) for mark in record.annotations
            if (label := group_beat(mark.symbol)) is not None
        ]
        samples = np.array([sample for sample, _ in selected], dtype=int)
        y = np.array([CLASS_ORDER.index(label) for _, label in selected], dtype=int)
        raw = record.signals[:, 0]
        filtered = preprocess(raw, record.fs)
        if feature_version in ("v2", "v3"):
            all_beats = [mark for mark in record.annotations if is_beat_annotation(mark.symbol)]
            all_samples = np.array([mark.sample for mark in all_beats], dtype=int)
            keep = np.array([group_beat(mark.symbol) is not None for mark in all_beats])
            extractor = extract_features_v3 if feature_version == "v3" else extract_features_v2
            X = extractor(filtered, all_samples, record.fs)[keep]
        elif feature_version == "v1":
            X = extract_features(filtered, samples, record.fs)
        else:
            raise ValueError("unsupported feature version")
        probabilities.append(classifier.predict_proba(X))
        labels.append(y)
        qualities.append(estimate_quality(
            raw, filtered, samples, record.fs, scaler,
            dc_offset_correction=quality_version == "v3",
        ))
    p = np.vstack(probabilities)
    y = np.concatenate(labels)
    calibrator = fit_temperature(p, y)
    calibrated = calibrator.transform(p)
    abstention_threshold = choose_abstention_threshold(calibrated, y)
    quality_floor = float(np.percentile(np.concatenate(qualities), 10))
    evidence = reliability_evidence(calibrated.max(axis=1), calibrated.argmax(axis=1) == y)
    morphology_limit = (
        float(np.percentile(morphology_blocks, 97))
        if quality_version == "v3" else None
    )
    severe_morphology_limit = (
        float(np.percentile(morphology_blocks, 99))
        if quality_version == "v3" else None
    )
    return (
        calibrator, abstention_threshold, quality_floor, scaler, evidence,
        morphology_limit, severe_morphology_limit,
    )


def pooled_end_to_end(records: dict[str, dict]) -> dict:
    values = [row["metrics"] for row in records.values()]
    pooled = {
        key: sum(row[key] for row in values)
        for key in (
            "matched_beats", "missed_beats", "extra_detections",
            "excluded_reference_beats", "abstentions", "predictions",
        )
    }
    pooled["abstention_rate"] = pooled["abstentions"] / pooled["predictions"] if pooled["predictions"] else None
    pooled["classes"] = {}
    for label in CLASS_ORDER:
        support = sum(row["classes"][label]["support"] for row in values)
        correct = sum(row["classes"][label]["correct"] for row in values)
        predicted = sum(row["classes"][label]["matched_predictions"] for row in values)
        pooled["classes"][label] = {
            "support": support, "correct": correct, "matched_predictions": predicted,
            "recall": correct / support if support else None,
            "matched_precision": correct / predicted if predicted else None,
        }
    return pooled


def benchmark_end_to_end(
    root: Path, partitions: dict[str, list[str]], split_sha256: str,
    model_path: Path,
    pipeline_version: str = "v1",
) -> dict:
    if pipeline_version not in ("v1", "v3"):
        raise ValueError("unsupported pipeline version")
    package = joblib.load(model_path)
    classifier = package["model"]
    feature_version = package["metadata"].get("feature_version", "v1")
    if package["metadata"]["split_sha256"] != split_sha256:
        raise ValueError("model and split identifiers differ")
    detector = detect_pan_tompkins_gap_rescue if pipeline_version == "v3" else detect_pan_tompkins
    quality_version = "v3" if pipeline_version == "v3" else "v1"
    (
        calibrator, threshold, quality_floor, scaler, validation_reliability,
        morphology_limit, severe_morphology_limit,
    ) = fit_reliability(
        root, partitions, classifier, feature_version, detector, quality_version,
    )
    records = {}
    for record_id in partitions["test"]:
        record = load_record(root, record_id)
        predictions = analyze(
            record, detector, classifier, calibrator,
            threshold, quality_floor, scaler, feature_version, quality_version,
            morphology_limit, severe_morphology_limit,
        )
        records[record_id] = {
            "metrics": end_to_end_metrics(reference_truth(record), predictions, record.fs),
            "rows": review_rows(record, predictions),
            "heart_rate": heart_rate_series(predictions, record.fs),
            "sample_rate_hz": record.fs,
            "lead_names": list(record.lead_names),
        }
    matched_rows = [
        row for record in records.values() for row in record["rows"]
        if row["reference_group"] in CLASS_ORDER and row["score"] is not None
    ]
    test_reliability = reliability_evidence(
        np.array([row["score"] for row in matched_rows]),
        np.array([row["group"] == row["reference_group"] for row in matched_rows]),
    )
    return {
        "dataset": "mitdb", "dataset_version": "1.0.0",
        "split_sha256": split_sha256, "evaluation": "continuous_end_to_end",
        "model_id": package["metadata"]["model_id"],
        "feature_version": feature_version,
        "detector_id": (
            "pan_tompkins_gap_rescue-v3" if pipeline_version == "v3" else
            "pan_tompkins_style-v1"
        ),
        "quality_version": quality_version,
        "calibration": {"temperature": calibrator.temperature, "fit_partition": "validation"},
        "abstention_threshold": threshold,
        "quality_floor": quality_floor,
        "morphology_block_limit": morphology_limit,
        "severe_morphology_limit": severe_morphology_limit,
        "quality_scaler": {"median": scaler.median, "scale": scaler.scale, "fit_partition": "train"},
        "reliability": {
            "validation_reference_locations": validation_reliability,
            "test_matched_detections": test_reliability,
        },
        "records": records,
        "pooled": pooled_end_to_end(records),
    }

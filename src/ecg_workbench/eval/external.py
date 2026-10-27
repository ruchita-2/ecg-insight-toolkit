"""Frozen reference-location evaluation on the separate SVDB corpus."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import wfdb

from ecg_workbench.data.labels import group_beat, is_beat_annotation
from ecg_workbench.data.manifest import verify_dataset
from ecg_workbench.eval.classification import score_classes
from ecg_workbench.signal.features import extract_features_v2, extract_features_v3
from ecg_workbench.signal.filters import preprocess


def benchmark_svdb(
    root: Path, model_paths: dict[str, Path], split_sha256: str,
) -> dict:
    record_file = root / "RECORDS"
    if not record_file.is_file():
        raise FileNotFoundError(f"missing SVDB record list: {record_file}")
    record_ids = [line.strip() for line in record_file.read_text().splitlines() if line.strip()]
    if len(record_ids) != 78 or len(set(record_ids)) != 78:
        raise ValueError("SVDB benchmark requires all 78 unique recordings")
    required = ["RECORDS"] + [
        f"{record_id}.{suffix}"
        for record_id in record_ids for suffix in ("hea", "dat", "atr")
    ]
    problems = verify_dataset(root, required)
    if problems:
        raise ValueError(f"SVDB checksum verification failed: {problems[:5]}")

    packages = {}
    for name, path in model_paths.items():
        package = joblib.load(path)
        metadata = package["metadata"]
        if metadata["split_sha256"] != split_sha256:
            raise ValueError("model and MIT-BIH split identifiers differ")
        if metadata["feature_version"] not in ("v2", "v3"):
            raise ValueError("SVDB comparison requires v2 or v3 feature models")
        packages[name] = package

    records = {name: {} for name in packages}
    all_truth = {name: [] for name in packages}
    all_predicted = {name: [] for name in packages}
    excluded_counts = {}
    terminal_annotations = {}
    for record_id in record_ids:
        base = str(root / record_id)
        record = wfdb.rdrecord(base)
        marks = wfdb.rdann(base, "atr")
        if record.fs != 128 or record.sig_name[0] != "ECG1" or record.p_signal is None:
            raise ValueError(f"unexpected SVDB recording configuration: {record_id}")
        if any(sample < 0 or sample > record.sig_len for sample in marks.sample):
            raise ValueError(f"annotation beyond SVDB recording: {record_id}")
        terminal_annotations[record_id] = int(np.sum(marks.sample == record.sig_len))
        beat_marks = [
            (int(sample), group_beat(symbol))
            for sample, symbol in zip(marks.sample, marks.symbol, strict=True)
            if sample < record.sig_len and is_beat_annotation(symbol)
        ]
        samples = np.array([sample for sample, _ in beat_marks], dtype=int)
        keep = np.array([label is not None for _, label in beat_marks], dtype=bool)
        truth = np.array([label for _, label in beat_marks if label is not None], dtype="U1")
        excluded_counts[record_id] = int((~keep).sum())
        filtered = preprocess(record.p_signal[:, 0], record.fs)
        for name, package in packages.items():
            extractor = (
                extract_features_v3 if package["metadata"]["feature_version"] == "v3"
                else extract_features_v2
            )
            X = extractor(filtered, samples, record.fs)[keep]
            predicted = package["model"].predict(X)
            records[name][record_id] = score_classes(truth, predicted)
            all_truth[name].append(truth)
            all_predicted[name].append(predicted)
    return {
        "dataset": "svdb", "dataset_version": "1.0.0",
        "evaluation": "frozen_reference_locations_external",
        "split_sha256": split_sha256,
        "model_id": "frozen-v2-v3-comparison",
        "record_ids": record_ids,
        "lead": "ECG1", "sample_rate_hz": 128,
        "verified_files": len(required),
        "excluded_beat_counts": excluded_counts,
        "terminal_annotations_excluded": terminal_annotations,
        "models": {
            name: {
                "model_id": package["metadata"]["model_id"],
                "feature_version": package["metadata"]["feature_version"],
                "pooled": score_classes(
                    np.concatenate(all_truth[name]), np.concatenate(all_predicted[name]),
                ),
                "records": records[name],
            }
            for name, package in packages.items()
        },
    }

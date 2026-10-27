"""Fair waveform-model comparison on the frozen reference-location split."""

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample

from ecg_workbench.data.labels import CLASS_ORDER, group_beat, is_beat_annotation
from ecg_workbench.data.records import load_record
from ecg_workbench.eval.classification import score_classes
from ecg_workbench.models.waveform_model import (
    fit_window_normalizer,
    predict_waveform,
    train_waveform_model,
)
from ecg_workbench.signal.features import extract_windows
from ecg_workbench.signal.filters import preprocess


def reference_windows(root: Path, record_id: str) -> tuple[np.ndarray, np.ndarray, int]:
    record = load_record(root, record_id)
    selected = [
        (mark.sample, label)
        for mark in record.annotations
        if (label := group_beat(mark.symbol)) is not None
    ]
    samples = np.array([sample for sample, _ in selected], dtype=int)
    labels = np.array([CLASS_ORDER.index(label) for _, label in selected], dtype=int)
    filtered = preprocess(record.signals[:, 0], record.fs)
    windows = extract_windows(filtered, samples, record.fs)
    if windows.shape[1] != 234:
        windows = resample(windows, 234, axis=1)
    excluded = sum(
        is_beat_annotation(mark.symbol) and group_beat(mark.symbol) is None
        for mark in record.annotations
    )
    return windows.astype(np.float32), labels, excluded


def benchmark_waveform(
    root: Path, partitions: dict[str, list[str]], split_sha256: str,
    model_dir: Path, seed: int = 20261025,
) -> dict:
    datasets = {}
    per_record = {}
    excluded = Counter()
    for partition in ("train", "validation", "test"):
        chunks, labels = [], []
        for record_id in partitions[partition]:
            windows, y, count = reference_windows(root, record_id)
            per_record[record_id] = (windows, y)
            chunks.append(windows)
            labels.append(y)
            excluded[partition] += count
        datasets[partition] = (np.vstack(chunks), np.concatenate(labels))
    train_x, train_y = datasets["train"]
    stats = fit_window_normalizer(train_x)
    val_x, val_y = datasets["validation"]
    test_x, test_y = datasets["test"]
    model = train_waveform_model(
        stats.transform(train_x), train_y,
        stats.transform(val_x), val_y, seed,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_id": "waveform-v1", "dataset_version": "1.0.0",
        "split_sha256": split_sha256, "seed": seed,
        "mean": stats.mean, "std": stats.std,
        "window_samples": 234, "class_order": CLASS_ORDER,
        "lead": 0, "filter": "Butterworth 0.5-40 Hz, offline",
        "validation_macro_f1": model.validation_macro_f1,
    }
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, model_dir / "waveform-v1.pt")
    pred = predict_waveform(model, stats.transform(test_x))
    record_scores = {
        record_id: score_classes(
            np.array([CLASS_ORDER[i] for i in y]),
            np.array([CLASS_ORDER[i] for i in predict_waveform(model, stats.transform(windows))]),
        )
        for record_id in partitions["test"]
        for windows, y in [per_record[record_id]]
    }
    return {
        "dataset": "mitdb", "dataset_version": "1.0.0",
        "split_sha256": split_sha256, "evaluation": "reference_locations",
        "model_id": "waveform-v1", "validation_macro_f1": model.validation_macro_f1,
        "test": score_classes(
            np.array([CLASS_ORDER[i] for i in test_y]),
            np.array([CLASS_ORDER[i] for i in pred]),
        ),
        "test_records": record_scores,
        "excluded_beat_counts": dict(excluded), "record_ids": partitions,
    }

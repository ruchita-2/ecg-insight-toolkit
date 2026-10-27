"""Reference-location classifier benchmark."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from ecg_workbench.data.labels import CLASS_ORDER, group_beat, is_beat_annotation
from ecg_workbench.data.records import load_record
from ecg_workbench.models.feature_model import (
    train_feature_models,
    train_feature_models_v2,
    train_feature_models_v3,
)
from ecg_workbench.signal.features import (
    FEATURE_NAMES,
    FEATURE_NAMES_V2,
    FEATURE_NAMES_V3,
    extract_features,
    extract_features_v2,
    extract_features_v3,
)
from ecg_workbench.signal.filters import preprocess


def reference_features(
    root: Path, record_id: str, *, all_beats_for_timing: bool = False,
    context_features: bool = False,
    feature_version: str | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    if feature_version is not None:
        if feature_version not in ("v1", "v2", "v3"):
            raise ValueError("unsupported feature version")
        all_beats_for_timing = feature_version in ("v2", "v3")
        context_features = feature_version in ("v2", "v3")
    record = load_record(root, record_id)
    marks = [(mark.sample, group_beat(mark.symbol)) for mark in record.annotations]
    selected = [(sample, label) for sample, label in marks if label is not None]
    labels = np.array([label for _, label in selected], dtype="U1")
    excluded = sum(
        is_beat_annotation(mark.symbol) and group_beat(mark.symbol) is None
        for mark in record.annotations
    )
    if all_beats_for_timing:
        beat_marks = [
            (mark.sample, group_beat(mark.symbol)) for mark in record.annotations
            if is_beat_annotation(mark.symbol)
        ]
        samples = np.array([sample for sample, _ in beat_marks], dtype=int)
        keep = np.array([label is not None for _, label in beat_marks], dtype=bool)
    else:
        samples = np.array([sample for sample, _ in selected], dtype=int)
        keep = None
    feature_extractor = (
        extract_features_v3 if feature_version == "v3" else
        extract_features_v2 if context_features else extract_features
    )
    X = feature_extractor(preprocess(record.signals[:, 0], record.fs), samples, record.fs)
    if keep is not None:
        X = X[keep]
    return X, labels, excluded


def score_classes(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    detail = classification_report(
        y_true, y_pred, labels=list(CLASS_ORDER),
        output_dict=True, zero_division=0,
    )
    return {
        "classes": {
            label: {
                key: detail[label][key]
                for key in ("precision", "recall", "f1-score", "support")
            }
            for label in CLASS_ORDER
        },
        "macro_f1": detail["macro avg"]["f1-score"],
        "accuracy": detail["accuracy"],
        "support": len(y_true),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(CLASS_ORDER)
        ).tolist(),
        "class_order": list(CLASS_ORDER),
    }


def benchmark_oracle(
    root: Path, partitions: dict[str, list[str]], split_sha256: str,
    model_dir: Path, seed: int = 20261025, version: str = "v1",
) -> dict:
    if version not in ("v1", "v2", "v3"):
        raise ValueError("unsupported classifier version")
    datasets = {}
    record_data = {}
    train_record_ids = []
    excluded = Counter()
    for partition in ("train", "validation", "test"):
        blocks = []
        labels = []
        for record_id in partitions[partition]:
            X, y, count = reference_features(
                root, record_id, feature_version=version,
            )
            record_data[record_id] = (X, y)
            blocks.append(X)
            labels.append(y)
            if partition == "train":
                train_record_ids.extend([record_id] * len(y))
            excluded[partition] += count
        datasets[partition] = (np.vstack(blocks), np.concatenate(labels))
    X_train, y_train = datasets["train"]
    if version == "v3":
        models = train_feature_models_v3(X_train, y_train, np.asarray(train_record_ids), seed)
    elif version == "v2":
        models = train_feature_models_v2(X_train, y_train, np.asarray(train_record_ids), seed)
    else:
        models = train_feature_models(X_train, y_train, seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    comparisons = {}
    for name, model in models.items():
        metadata = {
            "model_id": f"{name}-{version}", "dataset_version": "1.0.0",
            "split_sha256": split_sha256, "seed": seed,
            "feature_names": (
                FEATURE_NAMES_V3 if version == "v3" else
                FEATURE_NAMES_V2 if version == "v2" else FEATURE_NAMES
            ),
            "feature_version": version, "class_order": CLASS_ORDER,
            "lead": 0, "filter": "Butterworth 0.5-40 Hz, offline",
            "timing_reference": "all_annotated_beats" if version in ("v2", "v3") else "in_scope_beats",
        }
        joblib.dump({"model": model, "metadata": metadata}, model_dir / f"{name}-{version}.joblib")
        X_val, y_val = datasets["validation"]
        comparisons[name] = score_classes(y_val, model.predict(X_val))
    selected = max(comparisons, key=lambda name: comparisons[name]["macro_f1"])
    test_model = models[selected]
    X_test, y_test = datasets["test"]
    record_scores = {
        record_id: score_classes(y, test_model.predict(X))
        for record_id in partitions["test"]
        for X, y in [record_data[record_id]]
    }
    return {
        "dataset": "mitdb", "dataset_version": "1.0.0",
        "split_sha256": split_sha256, "evaluation": "reference_locations",
        "model_id": f"{selected}-{version}", "model_selection": "validation_macro_f1",
        "timing_reference": "all_annotated_beats" if version in ("v2", "v3") else "in_scope_beats",
        "validation_comparison": comparisons,
        **({"validation_records": {
            name: {
                record_id: score_classes(
                    record_data[record_id][1], model.predict(record_data[record_id][0]),
                )
                for record_id in partitions["validation"]
            }
            for name, model in models.items()
        }} if version == "v3" else {}),
        "test": score_classes(y_test, test_model.predict(X_test)),
        "test_records": record_scores,
        "excluded_beat_counts": dict(excluded),
        "record_ids": partitions,
        "feature_names": (
            FEATURE_NAMES_V3 if version == "v3" else
            FEATURE_NAMES_V2 if version == "v2" else FEATURE_NAMES
        ),
    }

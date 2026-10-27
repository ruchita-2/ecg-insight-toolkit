"""Count-based, pooled ECG evaluation metrics."""

import numpy as np

from ecg_workbench.data.labels import CLASS_ORDER
from ecg_workbench.eval.events import EventMatches, match_events


def detection_metrics(matches: EventMatches, hours: float, fs: float) -> dict:
    if hours <= 0 or fs <= 0:
        raise ValueError("duration and sample rate must be positive")
    tp = len(matches.matched)
    fn = len(matches.unmatched_reference)
    fp = len(matches.unmatched_predicted)
    errors_ms = np.array([abs(row[2]) * 1000 / fs for row in matches.matched])
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "positive_predictivity": tp / (tp + fp) if tp + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "false_detections_per_hour": fp / hours,
        "median_absolute_error_ms": float(np.median(errors_ms)) if tp else None,
        "p95_absolute_error_ms": float(np.percentile(errors_ms, 95)) if tp else None,
    }


def pooled_detection(rows: list[dict]) -> dict:
    tp = sum(row["tp"] for row in rows)
    fp = sum(row["fp"] for row in rows)
    fn = sum(row["fn"] for row in rows)
    hours = sum(row["hours"] for row in rows)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "positive_predictivity": tp / (tp + fp) if tp + fp else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "false_detections_per_hour": fp / hours if hours else None,
        "hours": hours,
    }


def end_to_end_metrics(truth: list[tuple[int, str | None]], predictions: list, fs: float) -> dict:
    matches = match_events(
        np.array([sample for sample, _ in truth]),
        np.array([row.sample for row in predictions]), round(0.15 * fs),
    )
    paired = [(truth[ri][1], predictions[pi]) for ri, pi, _ in matches.matched]
    classes = {}
    for label in CLASS_ORDER:
        support = sum(group == label for _, group in truth)
        correct = sum(group == label and row.group == label and not row.abstained for group, row in paired)
        predicted = sum(
            row.group == label and not row.abstained and group in CLASS_ORDER
            for group, row in paired
        )
        classes[label] = {
            "support": support, "correct": correct, "matched_predictions": predicted,
            "recall": correct / support if support else None,
            "matched_precision": correct / predicted if predicted else None,
        }
    return {
        "classes": classes,
        "matched_beats": len(matches.matched),
        "missed_beats": len(matches.unmatched_reference),
        "extra_detections": len(matches.unmatched_predicted),
        "excluded_reference_beats": sum(group is None for _, group in truth),
        "abstentions": sum(row.abstained for row in predictions),
        "predictions": len(predictions),
        "abstention_rate": sum(row.abstained for row in predictions) / len(predictions) if predictions else None,
    }

"""Validation-fitted temperature calibration and abstention selection."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class TemperatureCalibrator:
    temperature: float

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        logits = np.log(np.clip(probabilities, 1e-9, 1)) / self.temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> TemperatureCalibrator:
    if probabilities.ndim != 2 or probabilities.shape[1] != 3 or len(probabilities) != len(labels):
        raise ValueError("calibration requires aligned three-class probabilities")
    def loss(temperature: float) -> float:
        p = TemperatureCalibrator(temperature).transform(probabilities)
        return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-9, 1)).mean())
    fitted = minimize_scalar(loss, bounds=(0.5, 5.0), method="bounded")
    return TemperatureCalibrator(float(fitted.x))


def choose_abstention_threshold(
    probabilities: np.ndarray, labels: np.ndarray, min_coverage: float = 0.8,
) -> float:
    scores = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    candidates = np.linspace(0.35, 0.9, 56)
    best = (float("-inf"), 0.35)
    for threshold in candidates:
        accepted = scores >= threshold
        if np.mean(accepted) < min_coverage or not np.any(accepted):
            continue
        objective = float(np.mean(correct[accepted])) - 0.05 * (1 - np.mean(accepted))
        if objective > best[0]:
            best = (objective, float(threshold))
    return best[1]


def reliability_evidence(scores: np.ndarray, correct: np.ndarray) -> dict:
    """Counted calibration bins and accuracy as coverage falls."""
    scores = np.asarray(scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if scores.ndim != 1 or correct.shape != scores.shape or np.any((scores < 0) | (scores > 1)):
        raise ValueError("reliability scores and outcomes must align in [0, 1]")
    bins = []
    bucket = np.minimum((scores * 10).astype(int), 9)
    for index in range(10):
        left, right = index / 10, (index + 1) / 10
        selected = bucket == index
        count = int(selected.sum())
        bins.append({
            "lower": float(left), "upper": float(right), "support": count,
            "mean_score": float(scores[selected].mean()) if count else None,
            "accuracy": float(correct[selected].mean()) if count else None,
        })
    coverage = []
    for threshold in np.linspace(0, 1, 11):
        accepted = scores >= threshold
        count = int(accepted.sum())
        coverage.append({
            "threshold": float(threshold), "support": count,
            "coverage": count / len(scores) if len(scores) else None,
            "accuracy": float(correct[accepted].mean()) if count else None,
        })
    return {"support": len(scores), "calibration_bins": bins, "accuracy_coverage": coverage}

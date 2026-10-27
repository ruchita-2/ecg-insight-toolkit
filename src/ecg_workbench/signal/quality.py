"""Transparent ECG signal-quality indicators; higher means cleaner."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ecg_workbench.signal.features import extract_windows


@dataclass(frozen=True)
class QualityScaler:
    median: tuple[float, ...]
    scale: tuple[float, ...]

    def transform(self, indicators: np.ndarray) -> np.ndarray:
        center = np.asarray(self.median)
        scale = np.asarray(self.scale)
        severity = np.maximum(0, (indicators - center) / scale)
        return np.exp(-0.18 * np.mean(severity, axis=1))


def quality_indicators(
    raw: np.ndarray, filtered: np.ndarray, peaks: np.ndarray, fs: float,
    alternate_peaks: np.ndarray | None = None,
    center_baseline: bool = False,
    dc_offset_correction: bool = False,
) -> np.ndarray:
    """Baseline drift, high-frequency residue, flatline, clipping, disagreement."""
    raw = np.asarray(raw, dtype=float)
    filtered = np.asarray(filtered, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    if raw.shape != filtered.shape or raw.ndim != 1 or fs <= 0:
        raise ValueError("raw and filtered ECG must align")
    if np.any(peaks < 0) or np.any(peaks >= len(raw)):
        raise ValueError("quality peak outside signal")
    if len(raw) < round(2 * fs):
        raise ValueError("quality estimate needs at least two seconds")
    baseline = sosfiltfilt(butter(2, 0.5, fs=fs, output="sos"), raw)
    radius = max(1, round(fs))
    rows = []
    for peak in peaks:
        left, right = max(0, peak - radius), min(len(raw), peak + radius)
        x, clean = raw[left:right], filtered[left:right]
        raw_power = float(np.mean(clean * clean)) + 1e-9
        baseline_window = baseline[left:right]
        drift = float(np.mean(baseline_window ** 2) / raw_power)
        centered = baseline_window - np.median(baseline_window)
        centered_drift = float(np.mean(centered ** 2) / raw_power)
        if center_baseline or (dc_offset_correction and drift > 20 and centered_drift < 0.1):
            drift = centered_drift
        residue = x - clean
        high_frequency = float(np.mean(np.diff(residue) ** 2) / raw_power) if len(x) > 1 else 0.0
        flat = float(np.mean(np.abs(np.diff(x)) < 1e-5)) if len(x) > 1 else 0.0
        clip = float(np.mean((x == np.max(x)) | (x == np.min(x))))
        disagreement = 0.0
        if alternate_peaks is not None:
            disagreement = float(
                not np.any(np.abs(alternate_peaks - peak) <= round(0.15 * fs))
            )
        rows.append((drift, high_frequency, flat, clip, disagreement))
    return np.asarray(rows, dtype=float).reshape(-1, 5)


def fit_quality_scaler(indicators: np.ndarray) -> QualityScaler:
    if indicators.ndim != 2 or indicators.shape[1] != 5 or len(indicators) == 0:
        raise ValueError("quality scaler needs five training indicators")
    median = np.median(indicators, axis=0)
    scale = np.maximum(np.percentile(indicators, 90, axis=0) - median, 1e-4)
    return QualityScaler(tuple(median.tolist()), tuple(scale.tolist()))


def estimate_quality(
    raw: np.ndarray, filtered: np.ndarray, peaks: np.ndarray, fs: float,
    scaler: QualityScaler | None = None,
    alternate_peaks: np.ndarray | None = None,
    center_baseline: bool = False,
    dc_offset_correction: bool = False,
) -> np.ndarray:
    indicators = quality_indicators(
        raw, filtered, peaks, fs, alternate_peaks, center_baseline,
        dc_offset_correction,
    )
    if scaler is None:
        # A fixed neutral fallback is useful for a small interactive demo.
        scaler = QualityScaler((0.5, 0.1, 0.0, 0.01, 0.0), (0.5, 0.2, 0.1, 0.05, 1.0))
    return scaler.transform(indicators)


def cross_lead_quality_flags(
    primary_peaks: np.ndarray, secondary_peaks: np.ndarray, fs: float,
    block_seconds: float = 30.0, min_agreement: float = 0.8,
    min_secondary_coverage: float = 0.9,
) -> np.ndarray:
    """Flag blocks with poor lead agreement when the second lead is usable."""
    primary = np.asarray(primary_peaks, dtype=int)
    secondary = np.asarray(secondary_peaks, dtype=int)
    if (primary.ndim != 1 or secondary.ndim != 1 or fs <= 0 or block_seconds <= 0 or
            not 0 <= min_agreement <= 1 or not 0 <= min_secondary_coverage <= 1 or
            np.any(np.diff(primary) <= 0) or np.any(np.diff(secondary) <= 0)):
        raise ValueError("lead peak sequences and quality thresholds must be valid")
    flags = np.zeros(len(primary), dtype=bool)
    if (not len(primary) or not len(secondary) or
            len(secondary) / len(primary) < min_secondary_coverage):
        return flags
    insert = np.searchsorted(secondary, primary)
    left = np.abs(primary - secondary[np.clip(insert - 1, 0, len(secondary) - 1)])
    right = np.abs(primary - secondary[np.clip(insert, 0, len(secondary) - 1)])
    agreed = np.minimum(left, right) <= round(0.15 * fs)
    blocks = primary // round(block_seconds * fs)
    for block in np.unique(blocks):
        in_block = blocks == block
        flags[in_block] = np.mean(agreed[in_block]) < min_agreement
    return flags


def morphology_block_scores(
    filtered: np.ndarray, peaks: np.ndarray, fs: float,
    block_seconds: float = 30.0,
) -> np.ndarray:
    """Return the median QRS shape mismatch in each beat's local block."""
    beats = np.asarray(peaks, dtype=int)
    if fs <= 0 or block_seconds <= 0 or beats.ndim != 1 or np.any(np.diff(beats) <= 0):
        raise ValueError("morphology requires sorted beats and positive timing")
    if not len(beats):
        return np.empty(0, dtype=float)
    windows = extract_windows(filtered, beats, fs)
    center = round(0.25 * fs)
    radius = round(0.085 * fs)
    shapes = windows[:, center - radius : center + radius + 1]
    shapes = shapes - shapes.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(shapes * shapes, axis=1, keepdims=True))
    shapes = shapes / np.maximum(rms, 1e-3)
    mismatch = np.zeros(len(beats), dtype=float)
    for index in range(1, len(beats)):
        prior = shapes[max(0, index - 20) : index]
        mismatch[index] = float(np.min(np.mean((prior - shapes[index]) ** 2, axis=1)))
    blocks = beats // round(block_seconds * fs)
    scores = np.empty(len(beats), dtype=float)
    for block in np.unique(blocks):
        in_block = blocks == block
        scores[in_block] = np.median(mismatch[in_block])
    return scores

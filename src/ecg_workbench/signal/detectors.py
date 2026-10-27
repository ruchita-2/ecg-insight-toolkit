"""Two deliberately transparent beat detector baselines."""

from itertools import pairwise

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


def _validate(signal: np.ndarray, fs: float) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    if fs <= 80 or values.ndim != 1 or len(values) < round(2 * fs):
        raise ValueError("detector requires at least two seconds of ECG")
    if not np.all(np.isfinite(values)):
        raise ValueError("ECG contains non-finite samples")
    return values


def _pick_refractory(candidates: np.ndarray, strength: np.ndarray, distance: int) -> np.ndarray:
    """Keep the strongest candidate within each refractory neighborhood."""
    order = sorted(candidates, key=lambda i: (-float(strength[i]), int(i)))
    chosen: list[int] = []
    for candidate in order:
        if all(abs(int(candidate) - prior) >= distance for prior in chosen):
            chosen.append(int(candidate))
    return np.asarray(sorted(chosen), dtype=int)


def detect_local(signal: np.ndarray, fs: float) -> np.ndarray:
    """Adaptive local-maximum baseline on a filtered ECG."""
    values = _validate(signal, fs)
    # The first and last second are omitted because zero-phase filtering is unstable there.
    margin = round(fs)
    robust_scale = np.median(np.abs(values - np.median(values))) * 1.4826
    prominence = max(0.12, 0.8 * robust_scale)
    candidates, _ = find_peaks(
        values, prominence=prominence, distance=round(0.2 * fs)
    )
    keep = (candidates >= margin) & (candidates < len(values) - margin)
    return _pick_refractory(
        candidates[keep], values, round(0.2 * fs)
    )


def _pan_integrated(values: np.ndarray, fs: float) -> np.ndarray:
    band = sosfiltfilt(
        butter(3, (5, 15), btype="bandpass", fs=fs, output="sos"), values
    )
    derivative = np.gradient(band)
    energy = derivative * derivative
    window = max(1, round(0.15 * fs))
    return np.convolve(energy, np.ones(window) / window, mode="same")


def _pan_aligned(
    values: np.ndarray, integrated: np.ndarray, threshold: float | np.ndarray, fs: float,
) -> np.ndarray:
    candidates, _ = find_peaks(
        integrated, height=threshold, distance=round(0.2 * fs)
    )
    margin = round(fs)
    aligned: list[int] = []
    for candidate in candidates:
        if candidate < margin or candidate >= len(values) - margin:
            continue
        radius = round(0.1 * fs)
        left, right = max(0, candidate - radius), min(len(values), candidate + radius + 1)
        # Some leads have negative R waves; choose the largest absolute excursion.
        aligned.append(left + int(np.argmax(np.abs(values[left:right]))))
    return _pick_refractory(np.unique(aligned), np.abs(values), round(0.2 * fs))


def detect_pan_tompkins(signal: np.ndarray, fs: float) -> np.ndarray:
    """Offline Pan–Tompkins-style detector with one record-wide threshold."""
    values = _validate(signal, fs)
    integrated = _pan_integrated(values, fs)
    # A robust global threshold remains fixed for the record and avoids adapting
    # to held-out annotation labels.
    threshold = np.median(integrated) + 0.22 * (
        np.percentile(integrated, 95) - np.median(integrated)
    )
    return _pan_aligned(values, integrated, threshold, fs)


def detect_pan_tompkins_local(signal: np.ndarray, fs: float) -> np.ndarray:
    """Apply the same detector with a threshold recalculated every 30 seconds."""
    values = _validate(signal, fs)
    integrated = _pan_integrated(values, fs)
    threshold = np.empty(len(integrated), dtype=float)
    block_samples = round(30 * fs)
    for left in range(0, len(integrated), block_samples):
        right = min(left + block_samples, len(integrated))
        block = integrated[left:right]
        median = np.median(block)
        threshold[left:right] = median + 0.22 * (np.percentile(block, 95) - median)
    return _pan_aligned(values, integrated, threshold, fs)


def detect_pan_tompkins_gap_rescue(signal: np.ndarray, fs: float) -> np.ndarray:
    """Use local thresholds only inside gaps longer than one minute."""
    values = _validate(signal, fs)
    primary = detect_pan_tompkins(values, fs)
    boundaries = np.r_[0, primary, len(values)]
    gaps = [
        (left, right) for left, right in pairwise(boundaries)
        if right - left > round(60 * fs)
    ]
    if not gaps:
        return primary
    local = detect_pan_tompkins_local(values, fs)
    rescued = [
        int(peak) for peak in local
        if any(left < peak < right for left, right in gaps)
    ]
    combined = np.unique(np.r_[primary, rescued])
    return _pick_refractory(combined, np.abs(values), round(0.2 * fs))

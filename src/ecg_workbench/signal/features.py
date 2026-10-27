"""Fixed-width, past-only beat morphology and timing features."""

import numpy as np

FEATURE_NAMES = (
    "mean", "std", "rms", "range", "r_amplitude", "qrs_energy",
    "qrs_width_s", "left_area", "right_area", "slope_before",
    "slope_after", "spectral_0_5", "spectral_5_15", "spectral_15_40",
    "previous_rr_s", "past_three_rr_s", "past_heart_rate_bpm",
)
FEATURE_NAMES_V2 = FEATURE_NAMES + (
    "rr_ratio_recent", "rr_ratio_long", "r_amplitude_ratio_long",
)
FEATURE_NAMES_V3 = FEATURE_NAMES_V2 + (
    "following_rr_s", "following_rr_to_previous",
    "following_rr_ratio_recent", "qrs_mismatch_prior8",
)


def extract_windows(signal: np.ndarray, beat_samples: np.ndarray, fs: float) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    beats = np.asarray(beat_samples, dtype=int)
    if values.ndim != 1 or beats.ndim != 1 or fs <= 0:
        raise ValueError("signal and beat positions must be 1-D with positive fs")
    if np.any(beats < 0) or np.any(beats >= len(values)):
        raise ValueError("beat outside signal")
    before, after = round(0.25 * fs), round(0.4 * fs)
    if len(beats) == 0:
        return np.empty((0, before + after), dtype=float)
    offsets = np.arange(-before, after)
    indices = np.clip(beats[:, None] + offsets, 0, len(values) - 1)
    return values[indices]


def extract_features(signal: np.ndarray, beat_samples: np.ndarray, fs: float) -> np.ndarray:
    beats = np.asarray(beat_samples, dtype=int)
    windows = extract_windows(signal, beats, fs)
    if len(beats) == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=float)
    center = round(0.25 * fs)
    qrs_radius = max(1, round(0.07 * fs))
    qrs = windows[:, center - qrs_radius : center + qrs_radius + 1]
    # Preserve amplitude polarity, but width is measured relative to absolute peak.
    peak_abs = np.max(np.abs(qrs), axis=1)
    width = np.sum(np.abs(qrs) > 0.5 * peak_abs[:, None], axis=1) / fs
    centered = windows - windows.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2 / windows.shape[1] ** 2
    freqs = np.fft.rfftfreq(windows.shape[1], 1 / fs)
    bands = [
        power[:, (freqs >= lo) & (freqs < hi)].sum(axis=1)
        for lo, hi in ((0, 5), (5, 15), (15, 40))
    ]
    # The first beat has no prior interval; use a fixed, documented neutral value.
    previous = np.full(len(beats), 1.0)
    past_three = np.full(len(beats), 1.0)
    for index in range(1, len(beats)):
        previous[index] = max(0.2, (beats[index] - beats[index - 1]) / fs)
        past_three[index] = np.mean(previous[max(1, index - 2) : index + 1])
    hr = 60 / np.maximum(past_three, 0.2)
    features = np.column_stack((
        windows.mean(axis=1), windows.std(axis=1),
        np.sqrt(np.mean(windows**2, axis=1)),
        np.ptp(windows, axis=1), windows[:, center],
        np.mean(qrs**2, axis=1), width,
        np.mean(np.abs(windows[:, :center]), axis=1),
        np.mean(np.abs(windows[:, center:]), axis=1),
        windows[:, center] - windows[:, center - qrs_radius],
        windows[:, center + qrs_radius] - windows[:, center],
        *bands, previous, past_three, hr,
    ))
    if not np.all(np.isfinite(features)):
        raise ValueError("non-finite beat feature")
    return features


def extract_features_v2(signal: np.ndarray, beat_samples: np.ndarray, fs: float) -> np.ndarray:
    """Base measurements plus past-only, within-record timing and amplitude ratios."""
    beats = np.asarray(beat_samples, dtype=int)
    base = extract_features(signal, beats, fs)
    if len(beats) == 0:
        return np.empty((0, len(FEATURE_NAMES_V2)), dtype=float)
    rr = base[:, FEATURE_NAMES.index("previous_rr_s")]
    amplitude = np.abs(base[:, FEATURE_NAMES.index("r_amplitude")])
    recent_ratio = np.ones(len(beats))
    long_ratio = np.ones(len(beats))
    amplitude_ratio = np.ones(len(beats))
    for index in range(1, len(beats)):
        prior_rr = rr[max(1, index - 100) : index]
        if len(prior_rr):
            recent_ratio[index] = rr[index] / max(np.median(prior_rr[-5:]), 0.2)
            long_ratio[index] = rr[index] / max(np.median(prior_rr), 0.2)
        prior_amplitude = amplitude[max(0, index - 100) : index]
        amplitude_ratio[index] = amplitude[index] / max(np.median(prior_amplitude), 1e-3)
    return np.column_stack((base, recent_ratio, long_ratio, amplitude_ratio))


def extract_features_v3(signal: np.ndarray, beat_samples: np.ndarray, fs: float) -> np.ndarray:
    """V2 context plus offline following-RR and prior-QRS similarity evidence."""
    beats = np.asarray(beat_samples, dtype=int)
    base = extract_features_v2(signal, beats, fs)
    if not len(beats):
        return np.empty((0, len(FEATURE_NAMES_V3)), dtype=float)
    previous = base[:, FEATURE_NAMES.index("previous_rr_s")]
    following = np.ones(len(beats), dtype=float)
    following[:-1] = np.maximum(0.2, np.diff(beats) / fs)
    following_to_previous = following / np.maximum(previous, 0.2)
    following_recent = np.ones(len(beats), dtype=float)
    for index in range(1, len(beats)):
        prior = previous[max(1, index - 5) : index]
        if len(prior):
            following_recent[index] = following[index] / max(np.median(prior), 0.2)

    windows = extract_windows(signal, beats, fs)
    center = round(0.25 * fs)
    radius = max(1, round(0.085 * fs))
    shapes = windows[:, center - radius : center + radius + 1]
    shapes = shapes - shapes.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(shapes * shapes, axis=1, keepdims=True))
    shapes = shapes / np.maximum(rms, 1e-3)
    mismatch = np.zeros(len(beats), dtype=float)
    for index in range(1, len(beats)):
        typical = np.median(shapes[max(0, index - 8) : index], axis=0)
        mismatch[index] = float(np.mean((shapes[index] - typical) ** 2))
    result = np.column_stack((
        base, following, following_to_previous, following_recent, mismatch,
    ))
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite beat feature")
    return result

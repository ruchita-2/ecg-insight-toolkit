"""Fixed offline ECG filters."""

import numpy as np
from scipy.signal import butter, sosfiltfilt


def preprocess(signal: np.ndarray, fs: float) -> np.ndarray:
    """Bandpass a full ECG record at 0.5–40 Hz, without phase delay."""
    samples = np.asarray(signal, dtype=float)
    if fs <= 80 or samples.ndim != 1 or len(samples) < int(fs * 2):
        raise ValueError("ECG needs at least two seconds and fs > 80 Hz")
    if not np.all(np.isfinite(samples)):
        raise ValueError("ECG contains non-finite samples")
    sos = butter(4, (0.5, 40.0), btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, samples)

import numpy as np
import pytest

from ecg_workbench.signal.detectors import (
    detect_local,
    detect_pan_tompkins,
    detect_pan_tompkins_gap_rescue,
    detect_pan_tompkins_local,
)
from ecg_workbench.signal.filters import preprocess


@pytest.mark.parametrize("detector", [
    detect_local, detect_pan_tompkins, detect_pan_tompkins_local,
    detect_pan_tompkins_gap_rescue,
])
def test_detector_keeps_well_spaced_peaks(detector):
    x = np.arange(1800)
    signal = sum(np.exp(-0.5 * ((x - p) / 3) ** 2) for p in (360, 390, 720))
    peaks = detector(signal, 360)
    assert len(peaks) == 2
    assert np.all(np.diff(peaks) >= 72)


def test_short_signal_is_rejected():
    with pytest.raises(ValueError, match="two seconds"):
        preprocess(np.zeros(100), 360)


def test_local_threshold_recovers_low_amplitude_segment():
    fs = 360
    samples = np.arange(60 * fs)
    signal = np.zeros_like(samples, dtype=float)
    for second in range(2, 58):
        center = second * fs
        amplitude = 1.0 if second < 30 else 8.0
        signal += amplitude * np.exp(-0.5 * ((samples - center) / 3) ** 2)
    global_peaks = detect_pan_tompkins(signal, fs)
    local_peaks = detect_pan_tompkins_local(signal, fs)
    assert sum(local_peaks < 30 * fs) >= 25
    assert sum(local_peaks < 30 * fs) > sum(global_peaks < 30 * fs)


def test_gap_rescue_only_changes_long_silent_stretches():
    fs = 360
    samples = np.arange(120 * fs)
    signal = np.zeros_like(samples, dtype=float)
    for second in range(2, 118):
        center = second * fs
        amplitude = 1.0 if second < 70 else 8.0
        signal += amplitude * np.exp(-0.5 * ((samples - center) / 3) ** 2)
    global_peaks = detect_pan_tompkins(signal, fs)
    rescued = detect_pan_tompkins_gap_rescue(signal, fs)
    assert sum(rescued < 70 * fs) > sum(global_peaks < 70 * fs)
    assert sum(rescued >= 70 * fs) == sum(global_peaks >= 70 * fs)

import numpy as np

from ecg_workbench.signal.quality import (
    cross_lead_quality_flags,
    estimate_quality,
    fit_quality_scaler,
    morphology_block_scores,
    quality_indicators,
)


def test_quality_is_bounded_and_train_scaler_is_separate():
    x = np.sin(np.arange(720) / 12)
    peaks = np.array([180, 540])
    indicators = quality_indicators(x, x, peaks, 360)
    scaler = fit_quality_scaler(indicators)
    scores = estimate_quality(x, x, peaks, 360, scaler)
    assert np.all((0 <= scores) & (scores <= 1))


def test_centered_baseline_does_not_treat_constant_offset_as_drift():
    fs = 360
    samples = np.arange(4 * fs)
    filtered = np.sin(samples / 12)
    raw = filtered + 10
    peaks = np.array([fs, 2 * fs, 3 * fs])
    old = quality_indicators(raw, filtered, peaks, fs)
    centered = quality_indicators(raw, filtered, peaks, fs, center_baseline=True)
    corrected = quality_indicators(raw, filtered, peaks, fs, dc_offset_correction=True)
    assert np.median(centered[:, 0]) < np.median(old[:, 0]) / 100
    np.testing.assert_allclose(corrected[:, 0], centered[:, 0])


def test_cross_lead_flags_low_agreement_blocks_with_coverage_fallback():
    fs = 360
    primary = np.arange(1, 61) * fs
    secondary = primary.copy()
    secondary[30:] += round(0.3 * fs)
    flags = cross_lead_quality_flags(primary, secondary, fs)
    assert not np.any(flags[:29])
    assert np.all(flags[30:59])
    assert not np.any(cross_lead_quality_flags(primary, primary[:30], fs))


def test_morphology_uses_block_median_not_one_unusual_beat():
    fs = 360
    samples = np.arange(60 * fs)
    peaks = np.arange(1, 60) * fs
    signal = np.zeros(len(samples), dtype=float)
    for peak in peaks:
        signal += np.exp(-0.5 * ((samples - peak) / 5) ** 2)
    signal -= 2 * np.exp(-0.5 * ((samples - 15 * fs) / 8) ** 2)
    rng = np.random.default_rng(42)
    signal[30 * fs:] += rng.normal(0, 0.5, len(signal) - 30 * fs)
    scores = morphology_block_scores(signal, peaks, fs)
    assert scores[14] == scores[0]
    assert scores[35] > scores[14]

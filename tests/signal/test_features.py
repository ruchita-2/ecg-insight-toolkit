import numpy as np

from ecg_workbench.signal.features import (
    FEATURE_NAMES,
    FEATURE_NAMES_V2,
    FEATURE_NAMES_V3,
    extract_features,
    extract_features_v2,
    extract_features_v3,
)


def test_features_ignore_future_rr():
    signal = np.sin(np.arange(1800) * 0.1)
    before = extract_features(signal, np.array([360, 720, 1080]), 360)
    after = extract_features(signal, np.array([360, 720, 1500]), 360)
    np.testing.assert_allclose(before[1], after[1])
    assert before.shape[1] == len(FEATURE_NAMES)
    assert np.isfinite(before).all()


def test_v2_context_uses_only_past_beats():
    signal = np.sin(np.arange(1800) * 0.1)
    before = extract_features_v2(signal, np.array([360, 720, 1080]), 360)
    after = extract_features_v2(signal, np.array([360, 720, 1500]), 360)
    np.testing.assert_allclose(before[1], after[1])
    assert before.shape[1] == len(FEATURE_NAMES_V2)
    assert np.isfinite(before).all()


def test_v3_offline_context_uses_following_interval():
    signal = np.sin(np.arange(1800) * 0.1)
    before = extract_features_v3(signal, np.array([360, 720, 1080]), 360)
    after = extract_features_v3(signal, np.array([360, 720, 1500]), 360)
    next_rr = FEATURE_NAMES_V3.index("following_rr_s")
    assert before[1, next_rr] == 1.0
    assert after[1, next_rr] == 780 / 360
    assert before.shape[1] == len(FEATURE_NAMES_V3)
    assert np.isfinite(before).all()


def test_v3_morphology_compares_with_prior_beats():
    signal = np.zeros(1800)
    for peak in (360, 720, 1080):
        signal[peak - 2 : peak + 3] = [0.1, 0.5, 1.0, 0.5, 0.1]
    signal[1440 - 2 : 1440 + 3] = [-0.1, -0.5, -1.0, -0.5, -0.1]
    result = extract_features_v3(signal, np.array([360, 720, 1080, 1440]), 360)
    mismatch = FEATURE_NAMES_V3.index("qrs_mismatch_prior8")
    assert result[0, mismatch] == 0.0
    assert result[1, mismatch] < 1e-6
    assert result[3, mismatch] > 1.0

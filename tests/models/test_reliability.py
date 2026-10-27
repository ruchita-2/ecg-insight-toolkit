import numpy as np

from ecg_workbench.models.reliability import fit_temperature, reliability_evidence


def test_calibration_returns_normalized_probabilities():
    p = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    calibrated = fit_temperature(p, np.array([0, 1, 2])).transform(p)
    np.testing.assert_allclose(calibrated.sum(axis=1), 1)


def test_reliability_evidence_keeps_support_and_coverage():
    evidence = reliability_evidence(np.array([0.6, 0.9]), np.array([False, True]))
    assert evidence["support"] == 2
    assert sum(row["support"] for row in evidence["calibration_bins"]) == 2
    assert evidence["accuracy_coverage"][0]["coverage"] == 1

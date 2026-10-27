import numpy as np

from ecg_workbench.data.records import Annotation, ECGRecord
from ecg_workbench.eval import classification
from ecg_workbench.signal.features import FEATURE_NAMES, FEATURE_NAMES_V3


def test_excluded_beat_contributes_to_v2_previous_rr(monkeypatch):
    record = ECGRecord(
        "sample", 100.0, np.zeros((400, 2)), ("lead0", "lead1"),
        (
            Annotation(100, "N", ""),
            Annotation(150, "F", ""),
            Annotation(200, "A", ""),
        ),
    )
    monkeypatch.setattr(classification, "load_record", lambda root, record_id: record)
    monkeypatch.setattr(classification, "preprocess", lambda signal, fs: signal)
    X, labels, excluded = classification.reference_features(
        None, "sample", all_beats_for_timing=True,
    )
    previous_rr = FEATURE_NAMES.index("previous_rr_s")
    assert labels.tolist() == ["N", "S"]
    assert excluded == 1
    assert X[1, previous_rr] == 0.5


def test_v3_reference_features_use_following_beat_even_if_excluded(monkeypatch):
    record = ECGRecord(
        "sample", 100.0, np.zeros((500, 2)), ("lead0", "lead1"),
        (
            Annotation(100, "N", ""),
            Annotation(200, "A", ""),
            Annotation(250, "F", ""),
        ),
    )
    monkeypatch.setattr(classification, "load_record", lambda root, record_id: record)
    monkeypatch.setattr(classification, "preprocess", lambda signal, fs: signal)
    X, labels, excluded = classification.reference_features(
        None, "sample", feature_version="v3",
    )
    assert labels.tolist() == ["N", "S"]
    assert excluded == 1
    assert X.shape == (2, len(FEATURE_NAMES_V3))
    assert X[1, FEATURE_NAMES_V3.index("following_rr_s")] == 0.5

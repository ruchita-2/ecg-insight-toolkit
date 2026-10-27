import numpy as np
import pytest

from ecg_workbench.models.feature_model import (
    record_capped_class_weights,
    train_feature_models,
)


def test_training_rejects_unknown_label():
    with pytest.raises(ValueError, match="N/S/V"):
        train_feature_models(np.ones((3, 5)), np.array(["N", "V", "F"]), 20261025)


def test_record_cap_limits_one_record_without_erasing_class_balance():
    labels = np.array(["N"] * 250 + ["V"] * 250 + ["S"] * 200 + ["S"] * 50)
    records = np.array(["n"] * 250 + ["v"] * 250 + ["large"] * 200 + ["small"] * 50)
    weights = record_capped_class_weights(labels, records, cap=100)
    assert np.isclose(weights[labels == "S"].sum(), weights[labels == "N"].sum())
    assert np.isclose(weights[records == "large"].sum() / weights[labels == "S"].sum(), 2 / 3)

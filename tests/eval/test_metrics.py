import pytest

from ecg_workbench.eval.events import EventMatches
from ecg_workbench.eval.metrics import detection_metrics, pooled_detection


def test_detection_metrics_count_misses_and_extras():
    matches = EventMatches(((0, 0, 2), (1, 1, -1)), (2,), (2,))
    result = detection_metrics(matches, hours=0.5, fs=360)
    assert result["sensitivity"] == pytest.approx(2 / 3)
    assert result["positive_predictivity"] == pytest.approx(2 / 3)
    assert result["false_detections_per_hour"] == 2


def test_pooling_uses_counts():
    pooled = pooled_detection([
        {"tp": 100, "fp": 0, "fn": 0, "hours": 1},
        {"tp": 0, "fp": 1, "fn": 1, "hours": 1},
    ])
    assert pooled["sensitivity"] == pytest.approx(100 / 101)

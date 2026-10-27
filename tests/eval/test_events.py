import numpy as np

from ecg_workbench.eval.events import match_events


def test_one_prediction_cannot_match_two_references():
    result = match_events(np.array([100, 110]), np.array([105]), 10)
    assert len(result.matched) == 1
    assert len(result.unmatched_reference) == 1
    assert len(result.unmatched_predicted) == 0


def test_match_prefers_smallest_error_and_tracks_extras():
    result = match_events(np.array([100, 300]), np.array([109, 101, 500]), 10)
    assert result.matched == ((0, 1, 1),)
    assert result.unmatched_reference == (1,)
    assert result.unmatched_predicted == (0, 2)

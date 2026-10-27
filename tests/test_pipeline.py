import numpy as np

from ecg_workbench import pipeline
from ecg_workbench.data.records import ECGRecord
from ecg_workbench.eval.metrics import end_to_end_metrics
from ecg_workbench.pipeline import BeatPrediction, heart_rate_series


def test_end_to_end_counts_missed_v_beat():
    truth = [(100, "N"), (300, "V")]
    predictions = [BeatPrediction(100, "N", 0.9, 0.8, False)]
    score = end_to_end_metrics(truth, predictions, fs=360)
    assert score["classes"]["V"]["recall"] == 0.0
    assert score["missed_beats"] == 1


def test_low_quality_does_not_overwrite_class_score():
    beat = BeatPrediction(100, "V", 0.91, 0.12, True)
    assert beat.score == 0.91 and beat.signal_quality == 0.12


def test_heart_rate_uses_only_previous_detections():
    beats = [BeatPrediction(sample, "N", 0.9, 0.8, False) for sample in (360, 720, 1080)]
    rates = heart_rate_series(beats, 360)
    assert [row["bpm"] for row in rates] == [None, 60.0, 60.0]


def test_v3_can_abstain_on_s_without_calling_clean_signal_noisy(monkeypatch):
    record = ECGRecord("sample", 360.0, np.zeros((1440, 2)), ("a", "b"), ())
    monkeypatch.setattr(pipeline, "preprocess", lambda signal, fs: signal)
    monkeypatch.setattr(pipeline, "estimate_quality", lambda *args, **kwargs: np.array([
        0.95 if kwargs.get("dc_offset_correction") else 0.7,
    ]))
    monkeypatch.setattr(pipeline, "cross_lead_quality_flags", lambda *args: np.array([False]))

    class Classifier:
        def predict_proba(self, features):
            return np.array([[0.02, 0.96, 0.02]])

    class Calibrator:
        def transform(self, probabilities):
            return probabilities

    prediction = pipeline.analyze(
        record, lambda signal, fs: np.array([360]), Classifier(), Calibrator(),
        0.8, 0.9, feature_version="v2", quality_version="v3",
        morphology_block_limit=0.02, severe_morphology_limit=0.04,
    )[0]
    assert prediction.signal_quality == 0.95
    assert prediction.abstained
    assert prediction.abstention_reason == "s_confirmation"


def test_v3_severe_morphology_abstains_even_when_leads_agree(monkeypatch):
    record = ECGRecord("sample", 360.0, np.zeros((1440, 2)), ("a", "b"), ())
    monkeypatch.setattr(pipeline, "preprocess", lambda signal, fs: signal)
    monkeypatch.setattr(
        pipeline, "estimate_quality", lambda *args, **kwargs: np.array([0.95]),
    )
    monkeypatch.setattr(pipeline, "morphology_block_scores", lambda *args: np.array([0.05]))
    monkeypatch.setattr(pipeline, "cross_lead_quality_flags", lambda *args: np.array([False]))

    class Classifier:
        def predict_proba(self, features):
            return np.array([[0.96, 0.02, 0.02]])

    class Calibrator:
        def transform(self, probabilities):
            return probabilities

    prediction = pipeline.analyze(
        record, lambda signal, fs: np.array([360]), Classifier(), Calibrator(),
        0.8, 0.9, feature_version="v2", quality_version="v3",
        morphology_block_limit=0.02, severe_morphology_limit=0.04,
    )[0]
    assert prediction.abstained
    assert prediction.abstention_reason == "low_signal_quality"

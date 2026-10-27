"""Frozen continuous-recording analysis pipeline."""

from dataclasses import asdict, dataclass

import numpy as np

from ecg_workbench.data.labels import CLASS_ORDER
from ecg_workbench.data.records import ECGRecord
from ecg_workbench.signal.detectors import detect_local
from ecg_workbench.signal.features import (
    extract_features,
    extract_features_v2,
    extract_features_v3,
)
from ecg_workbench.signal.filters import preprocess
from ecg_workbench.signal.quality import (
    QualityScaler,
    cross_lead_quality_flags,
    estimate_quality,
    morphology_block_scores,
)


@dataclass(frozen=True)
class BeatPrediction:
    sample: int
    group: str | None
    score: float
    signal_quality: float
    abstained: bool
    abstention_reason: str | None = None

    def to_dict(self, fs: float) -> dict:
        return {**asdict(self), "seconds": self.sample / fs}


def heart_rate_series(predictions: list[BeatPrediction], fs: float) -> list[dict]:
    """Past-only five-interval median heart rate from detected locations."""
    if fs <= 0:
        raise ValueError("sample rate must be positive")
    samples = np.array([row.sample for row in predictions], dtype=int)
    if len(samples) > 1 and np.any(np.diff(samples) <= 0):
        raise ValueError("detected beats must be strictly increasing")
    intervals = np.diff(samples) / fs
    result = []
    for index, sample in enumerate(samples):
        recent = intervals[max(0, index - 5) : index]
        valid = recent[(recent >= 0.25) & (recent <= 3.0)]
        bpm = float(60 / np.median(valid)) if len(valid) else None
        result.append({"sample": int(sample), "seconds": sample / fs, "bpm": bpm})
    return result


def analyze(
    record: ECGRecord, detector, classifier, calibrator,
    abstention_threshold: float, quality_floor: float,
    quality_scaler: QualityScaler | None = None,
    feature_version: str = "v1",
    quality_version: str = "v1",
    morphology_block_limit: float | None = None,
    severe_morphology_limit: float | None = None,
) -> list[BeatPrediction]:
    raw = record.signals[:, 0]
    filtered = preprocess(raw, record.fs)
    peaks = detector(filtered, record.fs)
    if len(peaks) == 0:
        return []
    if feature_version not in ("v1", "v2", "v3"):
        raise ValueError("unsupported feature version")
    feature_extractor = (
        extract_features_v3 if feature_version == "v3" else
        extract_features_v2 if feature_version == "v2" else extract_features
    )
    X = feature_extractor(filtered, peaks, record.fs)
    raw_probabilities = classifier.predict_proba(X)
    probabilities = calibrator.transform(raw_probabilities)
    alternate = detect_local(filtered, record.fs)
    if quality_version not in ("v1", "v3"):
        raise ValueError("unsupported quality version")
    quality = estimate_quality(
        raw, filtered, peaks, record.fs, quality_scaler, alternate,
        dc_offset_correction=quality_version == "v3",
    )
    conservative_quality = (
        estimate_quality(raw, filtered, peaks, record.fs, quality_scaler, alternate)
        if quality_version == "v3" else quality
    )
    if quality_version == "v3":
        if (morphology_block_limit is None or morphology_block_limit < 0 or
                severe_morphology_limit is None or severe_morphology_limit < morphology_block_limit):
            raise ValueError("v3 requires fitted morphology block limits")
        secondary = preprocess(record.signals[:, 1], record.fs)
        secondary_peaks = detector(secondary, record.fs)
        morphology = morphology_block_scores(filtered, peaks, record.fs)
        if len(secondary_peaks) / len(peaks) < 0.9:
            flags = morphology > morphology_block_limit
        else:
            flags = (
                cross_lead_quality_flags(peaks, secondary_peaks, record.fs)
                | (morphology > severe_morphology_limit)
            )
        quality[flags] = 0.0
    results = []
    for peak, prob, q, original_q in zip(
        peaks, probabilities, quality, conservative_quality, strict=True,
    ):
        group = CLASS_ORDER[int(np.argmax(prob))]
        score = float(np.max(prob))
        reason = (
            "low_classifier_score" if score < abstention_threshold else
            "low_signal_quality" if q < quality_floor else
            "s_confirmation" if quality_version == "v3" and group == "S"
            and original_q < quality_floor else None
        )
        results.append(BeatPrediction(
            int(peak), group, score, float(q), reason is not None, reason,
        ))
    return results

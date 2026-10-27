"""Bounded, explicit waveform response contract."""

from dataclasses import dataclass

from ecg_workbench.data.records import ECGRecord
from ecg_workbench.signal.filters import preprocess


@dataclass(frozen=True)
class WaveformWindow:
    record_id: str
    sample_rate_hz: float
    lead_names: tuple[str, ...]
    start_sample: int
    end_sample: int
    seconds: list[float]
    raw: list[list[float]]
    filtered_primary: list[float]

    @classmethod
    def from_record(cls, record: ECGRecord, left: int, right: int) -> "WaveformWindow":
        if left < 0 or right > len(record.signals) or right <= left:
            raise ValueError("invalid waveform bounds")
        # Match the full-record offline preprocessing used by the benchmark.
        filtered = preprocess(record.signals[:, 0], record.fs)[left:right]
        return cls(
            record_id=record.record_id, sample_rate_hz=record.fs,
            lead_names=record.lead_names, start_sample=left, end_sample=right,
            seconds=[sample / record.fs for sample in range(left, right)],
            raw=record.signals[left:right].T.tolist(),
            filtered_primary=filtered.tolist(),
        )

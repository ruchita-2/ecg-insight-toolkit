"""Validated WFDB access for MIT-BIH-style recordings."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

from ecg_workbench.data.labels import group_beat, is_beat_annotation

RECORD_ID = re.compile(r"[A-Za-z0-9_]+\Z")


@dataclass(frozen=True)
class Annotation:
    sample: int
    symbol: str
    aux: str


@dataclass(frozen=True)
class ECGRecord:
    record_id: str
    fs: float
    signals: np.ndarray
    lead_names: tuple[str, ...]
    annotations: tuple[Annotation, ...]


def load_record(root: Path, record_id: str) -> ECGRecord:
    if not RECORD_ID.fullmatch(record_id):
        raise ValueError(f"invalid record id: {record_id!r}")
    base = root / record_id
    if not base.with_suffix(".hea").is_file():
        raise FileNotFoundError(f"missing header for {record_id}")
    record = wfdb.rdrecord(str(base))
    marks = wfdb.rdann(str(base), "atr")
    signals = record.p_signal
    if signals is None or signals.ndim != 2 or signals.shape[1] != 2:
        raise ValueError(f"expected two ECG channels in {record_id}")
    if record.fs is None or record.fs <= 0:
        raise ValueError(f"invalid sample rate in {record_id}")
    if len(record.sig_name) != signals.shape[1]:
        raise ValueError(f"channel metadata mismatch in {record_id}")
    annotations = tuple(
        Annotation(int(sample), symbol, aux or "")
        for sample, symbol, aux in zip(
            marks.sample, marks.symbol, marks.aux_note, strict=True
        )
    )
    if any(mark.sample < 0 or mark.sample >= len(signals) for mark in annotations):
        raise ValueError(f"annotation outside signal in {record_id}")
    return ECGRecord(
        record_id=record_id,
        fs=float(record.fs),
        signals=signals,
        lead_names=tuple(record.sig_name),
        annotations=annotations,
    )


def audit_record(record: ECGRecord) -> dict:
    symbols = Counter(mark.symbol for mark in record.annotations)
    groups = Counter(
        group
        for mark in record.annotations
        if (group := group_beat(mark.symbol)) is not None
    )
    excluded = sum(
        is_beat_annotation(mark.symbol) and group_beat(mark.symbol) is None
        for mark in record.annotations
    )
    nonbeats = sum(
        not is_beat_annotation(mark.symbol) for mark in record.annotations
    )
    return {
        "record_id": record.record_id,
        "sample_rate_hz": record.fs,
        "channels": list(record.lead_names),
        "samples": len(record.signals),
        "original_symbols": dict(sorted(symbols.items())),
        "groups": {group: groups.get(group, 0) for group in ("N", "S", "V")},
        "excluded_beats": excluded,
        "nonbeat_annotations": nonbeats,
    }

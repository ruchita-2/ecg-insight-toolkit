import numpy as np
import pytest
import wfdb

from ecg_workbench.data.records import audit_record, load_record


def test_load_record_reads_two_channels_and_annotations(tmp_path):
    signal = np.zeros((720, 2), dtype=float)
    signal[100, 0] = 0.8
    signal[300, 1] = 1.2
    wfdb.wrsamp(
        record_name="100",
        fs=360,
        units=["mV", "mV"],
        sig_name=["MLII", "V1"],
        p_signal=signal,
        write_dir=str(tmp_path),
    )
    wfdb.wrann(
        record_name="100",
        extension="atr",
        sample=np.array([100, 300]),
        symbol=["N", "V"],
        aux_note=["", ""],
        write_dir=str(tmp_path),
    )
    record = load_record(tmp_path, "100")
    assert record.fs == 360
    assert record.signals.shape == (720, 2)
    assert record.lead_names == ("MLII", "V1")
    assert [(a.sample, a.symbol) for a in record.annotations] == [(100, "N"), (300, "V")]
    audit = audit_record(record)
    assert audit["groups"] == {"N": 1, "S": 0, "V": 1}
    assert audit["excluded_beats"] == 0


def test_load_record_rejects_missing_header(tmp_path):
    with pytest.raises(FileNotFoundError, match="header"):
        load_record(tmp_path, "100")


def test_load_record_rejects_path_escape(tmp_path):
    with pytest.raises(ValueError, match="record id"):
        load_record(tmp_path, "../100")

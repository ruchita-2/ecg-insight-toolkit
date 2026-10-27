import pytest

from ecg_workbench.data.splits import SplitManifest
from ecg_workbench.eval.noise import (
    assert_noise_isolation,
    interval_kind,
    parse_noise_record,
)


def test_noise_benchmark_rejects_training_source_overlap():
    split = SplitManifest(("118",), ("100",), ("101",), ("119",))
    with pytest.raises(ValueError, match="118"):
        assert_noise_isolation(split)


def test_snr_name_parser_and_schedule():
    assert parse_noise_record("118e_6") == ("118", -6)
    assert parse_noise_record("119e24") == ("119", 24)
    assert interval_kind(299) == "clean"
    assert interval_kind(301) == "noisy"
    assert interval_kind(421) == "clean"

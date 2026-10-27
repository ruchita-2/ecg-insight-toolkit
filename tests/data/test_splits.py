import pytest

from ecg_workbench.data.splits import (
    SplitManifest,
    create_split,
    location,
    support_by_split,
    validate_split,
)


def test_split_reserves_noise_sources_and_pairs_same_subject():
    counts = {
        "118": {"N": 2},
        "119": {"V": 2},
        "201": {"S": 2},
        "202": {"N": 2},
        "100": {"N": 2},
        "101": {"V": 2},
        "103": {"S": 2},
        "105": {"N": 2},
    }
    split = create_split(counts, seed=20261025)
    assert split.noise == ("118", "119")
    assert location(split, "201") == location(split, "202")
    assert create_split(counts, seed=20261025) == split
    assert set(split.train + split.validation + split.test + split.noise) == set(counts)
    support = support_by_split(split, counts)
    assert sum(section["N"] for section in support.values()) == 8
    assert sum(section["S"] for section in support.values()) == 4
    assert sum(section["V"] for section in support.values()) == 4


def test_validate_split_rejects_overlap():
    split = SplitManifest(
        train=("100", "201", "202"),
        validation=("100",),
        test=("103",),
        noise=("118", "119"),
    )
    with pytest.raises(ValueError, match="multiple partitions"):
        validate_split(split)


def test_validate_split_rejects_separated_pair():
    split = SplitManifest(
        train=("201",),
        validation=("202",),
        test=("103",),
        noise=("118", "119"),
    )
    with pytest.raises(ValueError, match="same-subject"):
        validate_split(split)

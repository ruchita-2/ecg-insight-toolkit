from hashlib import sha256

import pytest

from ecg_workbench.data.manifest import read_sha256sums, verify_dataset, verify_file


def test_verify_file_rejects_changed_bytes(tmp_path):
    path = tmp_path / "record.dat"
    path.write_bytes(b"original")
    expected = sha256(b"original").hexdigest()
    assert verify_file(path, expected)
    path.write_bytes(b"changed")
    assert not verify_file(path, expected)


def test_verify_dataset_reports_missing_file(tmp_path):
    expected = sha256(b"missing").hexdigest()
    (tmp_path / "SHA256SUMS.txt").write_text(f"{expected}  record.dat\n")
    assert verify_dataset(tmp_path) == ["missing: record.dat"]


def test_checksum_manifest_rejects_parent_path(tmp_path):
    expected = sha256(b"outside").hexdigest()
    (tmp_path / "SHA256SUMS.txt").write_text(f"{expected}  ../outside.dat\n")
    with pytest.raises(ValueError, match="unsafe path"):
        read_sha256sums(tmp_path)


def test_selected_benchmark_verification_ignores_unneeded_auxiliary_files(tmp_path):
    present = sha256(b"signal").hexdigest()
    absent = sha256(b"aux").hexdigest()
    (tmp_path / "SHA256SUMS.txt").write_text(
        f"{present}  beat.dat\n{absent}  auxiliary.dat\n"
    )
    (tmp_path / "beat.dat").write_bytes(b"signal")
    assert verify_dataset(tmp_path, ["beat.dat"]) == []
    assert verify_dataset(tmp_path) == ["missing: auxiliary.dat"]

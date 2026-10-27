"""Versioned dataset references and local checksum verification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    version: str
    source_url: str
    license: str
    expected_sha256: dict[str, str]


MITDB = DatasetSpec(
    name="MIT-BIH Arrhythmia Database",
    version="1.0.0",
    source_url="https://physionet.org/content/mitdb/1.0.0/",
    license="Open Data Commons Attribution License v1.0",
    expected_sha256={},
)

NSTDB = DatasetSpec(
    name="MIT-BIH Noise Stress Test Database",
    version="1.0.0",
    source_url="https://physionet.org/content/nstdb/1.0.0/",
    license="Open Data Commons Attribution License v1.0",
    expected_sha256={},
)


def verify_file(path: Path, expected_sha256: str) -> bool:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected_sha256.lower()


def read_sha256sums(root: Path) -> dict[str, str]:
    sums_file = root / "SHA256SUMS.txt"
    if not sums_file.is_file():
        raise FileNotFoundError(f"missing checksum manifest: {sums_file}")
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(sums_file.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"invalid SHA256SUMS.txt line {line_number}")
        relative = parts[1].lstrip("* ")
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"unsafe path on SHA256SUMS.txt line {line_number}")
        checksums[relative] = parts[0]
    if not checksums:
        raise ValueError("checksum manifest is empty")
    return checksums


def verify_dataset(root: Path, required_files: list[str] | None = None) -> list[str]:
    problems: list[str] = []
    checksums = read_sha256sums(root)
    if required_files is None:
        required_files = list(checksums)
    for relative in required_files:
        expected = checksums.get(relative)
        if expected is None:
            problems.append(f"not in checksum manifest: {relative}")
            continue
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            problems.append(f"unsafe required path: {relative}")
            continue
        path = root / relative
        if not path.is_file():
            problems.append(f"missing: {relative}")
        elif not verify_file(path, expected):
            problems.append(f"checksum mismatch: {relative}")
    return problems

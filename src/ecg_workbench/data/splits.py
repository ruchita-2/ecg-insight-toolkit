"""Deterministic subject-level partitions for MIT-BIH experiments."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Literal

from ecg_workbench.data.labels import CLASS_ORDER

Partition = Literal["train", "validation", "test", "noise"]
NOISE_RECORDS = ("118", "119")
PAIRED_SUBJECT = ("201", "202")


@dataclass(frozen=True)
class SplitManifest:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    noise: tuple[str, ...]

    def to_dict(self) -> dict[str, list[str]]:
        return {key: list(value) for key, value in asdict(self).items()}


def location(manifest: SplitManifest, record_id: str) -> Partition | None:
    for name in ("train", "validation", "test", "noise"):
        if record_id in getattr(manifest, name):
            return name
    return None


def validate_split(manifest: SplitManifest) -> None:
    groups = [
        set(manifest.train),
        set(manifest.validation),
        set(manifest.test),
        set(manifest.noise),
    ]
    if any(groups[i] & groups[j] for i in range(4) for j in range(i + 1, 4)):
        raise ValueError("record occurs in multiple partitions")
    if tuple(sorted(manifest.noise)) != NOISE_RECORDS:
        raise ValueError("noise source records must be reserved")
    if location(manifest, PAIRED_SUBJECT[0]) != location(manifest, PAIRED_SUBJECT[1]):
        raise ValueError("same-subject records 201 and 202 were separated")


def _group_records(
    record_class_counts: dict[str, dict[str, int]],
) -> list[tuple[tuple[str, ...], Counter[str]]]:
    groups: list[tuple[tuple[str, ...], Counter[str]]] = []
    seen: set[str] = set(NOISE_RECORDS)
    if set(PAIRED_SUBJECT) <= set(record_class_counts):
        counts = Counter()
        for record_id in PAIRED_SUBJECT:
            counts.update(record_class_counts[record_id])
        groups.append((PAIRED_SUBJECT, counts))
        seen.update(PAIRED_SUBJECT)
    for record_id in sorted(record_class_counts):
        if record_id not in seen:
            groups.append(((record_id,), Counter(record_class_counts[record_id])))
    return groups


def create_split(
    record_class_counts: dict[str, dict[str, int]], seed: int = 20261025
) -> SplitManifest:
    if not set(NOISE_RECORDS) <= set(record_class_counts):
        raise ValueError("record counts must include 118 and 119")
    if (PAIRED_SUBJECT[0] in record_class_counts) != (
        PAIRED_SUBJECT[1] in record_class_counts
    ):
        raise ValueError("record counts must include both 201 and 202")
    groups = _group_records(record_class_counts)
    if len(groups) < 3:
        raise ValueError("at least three development subjects are required")
    train_count = max(1, round(len(groups) * 0.6))
    validation_count = max(1, round(len(groups) * 0.2))
    if train_count + validation_count >= len(groups):
        train_count = len(groups) - validation_count - 1

    total = Counter()
    for _, counts in groups:
        total.update(counts)
    rng = random.Random(seed)
    best: SplitManifest | None = None
    best_score = float("inf")
    for _ in range(512):
        shuffled = groups.copy()
        rng.shuffle(shuffled)
        chunks = (
            shuffled[:train_count],
            shuffled[train_count : train_count + validation_count],
            shuffled[train_count + validation_count :],
        )
        partitions = [
            tuple(sorted(record_id for records, _ in chunk for record_id in records))
            for chunk in chunks
        ]
        candidate = SplitManifest(*partitions, NOISE_RECORDS)
        score = 0.0
        for fraction, chunk in zip((0.6, 0.2, 0.2), chunks, strict=True):
            counts = Counter()
            for _, group_counts in chunk:
                counts.update(group_counts)
            for label in CLASS_ORDER:
                if total[label]:
                    score += abs(counts[label] / total[label] - fraction)
                    if counts[label] == 0:
                        score += 10
        if score < best_score:
            best, best_score = candidate, score
    assert best is not None
    validate_split(best)
    expected = set(record_class_counts)
    actual = set(best.train + best.validation + best.test + best.noise)
    if actual != expected:
        raise ValueError("split does not cover every input record exactly once")
    return best


def split_digest(manifest: SplitManifest) -> str:
    encoded = json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def support_by_split(
    manifest: SplitManifest, record_class_counts: dict[str, dict[str, int]]
) -> dict[str, dict[str, int]]:
    return {
        partition: {
            label: sum(record_class_counts[record_id].get(label, 0) for record_id in record_ids)
            for label in CLASS_ORDER
        }
        for partition, record_ids in manifest.to_dict().items()
    }

"""One-to-one heartbeat event matching."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EventMatches:
    matched: tuple[tuple[int, int, int], ...]
    unmatched_reference: tuple[int, ...]
    unmatched_predicted: tuple[int, ...]


def match_events(
    reference: np.ndarray, predicted: np.ndarray, tolerance_samples: int
) -> EventMatches:
    """Match the closest eligible events, retaining original input indices."""
    ref = np.asarray(reference, dtype=int)
    pred = np.asarray(predicted, dtype=int)
    if ref.ndim != 1 or pred.ndim != 1 or tolerance_samples < 0:
        raise ValueError("event arrays must be 1-D and tolerance non-negative")
    if np.any(ref < 0) or np.any(pred < 0):
        raise ValueError("event positions must be non-negative")
    candidates: list[tuple[int, int, int]] = []
    pred_order = np.argsort(pred)
    sorted_pred = pred[pred_order]
    for ri, sample in enumerate(ref):
        left = np.searchsorted(sorted_pred, sample - tolerance_samples, side="left")
        right = np.searchsorted(sorted_pred, sample + tolerance_samples, side="right")
        for sorted_index in range(left, right):
            pi = int(pred_order[sorted_index])
            candidates.append((abs(int(pred[pi] - sample)), ri, pi))
    used_ref: set[int] = set()
    used_pred: set[int] = set()
    matched: list[tuple[int, int, int]] = []
    for _, ri, pi in sorted(candidates):
        if ri not in used_ref and pi not in used_pred:
            used_ref.add(ri)
            used_pred.add(pi)
            matched.append((ri, pi, int(pred[pi] - ref[ri])))
    return EventMatches(
        tuple(sorted(matched)),
        tuple(i for i in range(len(ref)) if i not in used_ref),
        tuple(i for i in range(len(pred)) if i not in used_pred),
    )

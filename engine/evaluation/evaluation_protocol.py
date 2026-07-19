"""Reusable purge/embargo/cluster helpers for chronological research splits.

These operate on plain source indices, segment IDs, and a horizon -- not on
any specific arena's internals -- so they apply equally to
``engine/stat_arb.py`` emissions, ``engine/legal_event.py`` event-study
rows, or any future research module with the same shape of problem: each
row at ``source_index`` makes a claim about an outcome observed over
``[source_index, source_index + horizon_steps]``, and that claim must not
leak into training if its outcome window overlaps the evaluation window.

This module does not change any frozen evaluation. It produces purge/embargo
*sensitivity* views alongside the existing chronological split so a
frozen-contract result and its purged/embargoed counterparts can both be
reported (see docs/evaluation-protocol.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class EvaluationProtocolError(RuntimeError):
    """Raised when interval/cluster inputs violate this module's contract."""


def target_interval(source_index: np.ndarray | int, horizon_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(target_start_index, target_end_index)`` for each source index.

    The interval is inclusive on both ends: a row observed at ``source_index``
    makes a claim about the outcome at ``source_index + horizon_steps``, and
    is considered to "occupy" every index in between (its state depends on
    the whole path, not just the endpoint).
    """
    if horizon_steps < 0:
        raise EvaluationProtocolError("horizon_steps must be non-negative")
    start = np.asarray(source_index, dtype=np.int64)
    end = start + int(horizon_steps)
    return start, end


def intervals_overlap(a_start: np.ndarray, a_end: np.ndarray,
                      b_start: np.ndarray, b_end: np.ndarray) -> np.ndarray:
    """Elementwise/broadcast inclusive-interval overlap test."""
    a_start = np.asarray(a_start, dtype=np.int64)
    a_end = np.asarray(a_end, dtype=np.int64)
    b_start = np.asarray(b_start, dtype=np.int64)
    b_end = np.asarray(b_end, dtype=np.int64)
    if np.any(a_end < a_start) or np.any(b_end < b_start):
        raise EvaluationProtocolError("interval end must not precede interval start")
    return (a_start <= b_end) & (b_start <= a_end)


def _merge_intervals(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(starts) == 0:
        return starts.astype(np.int64), ends.astype(np.int64)
    order = np.argsort(starts, kind="stable")
    sorted_starts = starts[order]
    sorted_ends = ends[order]
    merged_starts = [int(sorted_starts[0])]
    merged_ends = [int(sorted_ends[0])]
    for index in range(1, len(sorted_starts)):
        current_start = int(sorted_starts[index])
        current_end = int(sorted_ends[index])
        if current_start <= merged_ends[-1]:
            merged_ends[-1] = max(merged_ends[-1], current_end)
        else:
            merged_starts.append(current_start)
            merged_ends.append(current_end)
    return np.asarray(merged_starts, dtype=np.int64), np.asarray(merged_ends, dtype=np.int64)


def apply_embargo(held_out_start: np.ndarray, held_out_end: np.ndarray,
                  embargo_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Widen held-out intervals by ``embargo_steps`` on both sides.

    Widening both directions (not just after the held-out window) removes
    training rows whose own outcome window would otherwise butt directly
    against the evaluation window on either side, which is the symmetric
    generalization of the usual purged-k-fold embargo to a single
    chronological holdout.
    """
    if embargo_steps < 0:
        raise EvaluationProtocolError("embargo_steps must be non-negative")
    start = np.asarray(held_out_start, dtype=np.int64) - int(embargo_steps)
    end = np.asarray(held_out_end, dtype=np.int64) + int(embargo_steps)
    return start, end


def purge_overlapping_training_rows(train_start: np.ndarray, train_end: np.ndarray,
                                    held_out_start: np.ndarray, held_out_end: np.ndarray) -> np.ndarray:
    """Boolean mask, True where a training row's outcome window overlaps any held-out window.

    Callers should exclude ``True`` rows from training. Held-out intervals
    are merged first, so cost is ``O(n_train * n_merged_held_out_ranges)``;
    a chronological single-fold split has exactly one held-out range, so
    this is linear in practice.
    """
    train_start = np.asarray(train_start, dtype=np.int64)
    train_end = np.asarray(train_end, dtype=np.int64)
    merged_start, merged_end = _merge_intervals(np.asarray(held_out_start, dtype=np.int64),
                                                np.asarray(held_out_end, dtype=np.int64))
    purge = np.zeros(len(train_start), dtype=bool)
    for start, end in zip(merged_start, merged_end):
        purge |= (train_start <= end) & (start <= train_end)
    return purge


def assign_target_clusters(source_index: np.ndarray, horizon_steps: int,
                           segment_id: np.ndarray) -> np.ndarray:
    """Connected-component cluster id per row: rows whose outcome windows
    chain together via overlap (within the same causal segment) share a
    cluster. Use this instead of a raw row count when the sample size for a
    significance test needs to reflect independent observations rather than
    heavily overlapping ones.
    """
    source_index = np.asarray(source_index, dtype=np.int64)
    segment_id = np.asarray(segment_id, dtype=np.int64)
    if len(source_index) != len(segment_id):
        raise EvaluationProtocolError("source_index and segment_id must be the same length")
    starts, ends = target_interval(source_index, horizon_steps)
    order = np.argsort(source_index, kind="stable")
    cluster_ids = np.empty(len(source_index), dtype=np.int64)
    current_cluster = -1
    running_end = -1
    running_segment: int | None = None
    for position in order:
        if running_segment != int(segment_id[position]) or int(starts[position]) > running_end:
            current_cluster += 1
            running_end = int(ends[position])
            running_segment = int(segment_id[position])
        else:
            running_end = max(running_end, int(ends[position]))
        cluster_ids[position] = current_cluster
    return cluster_ids


@dataclass(frozen=True)
class PurgeSummary:
    train_rows_total: int
    purged_rows: int
    embargoed_extra_purged_rows: int
    remaining_train_rows: int


def build_evaluation_metadata(source_index: np.ndarray, horizon_steps: int, segment_id: np.ndarray,
                              is_training: np.ndarray, embargo_steps: int = 0) -> tuple[pd.DataFrame, PurgeSummary]:
    """Attach purge/embargo/cluster metadata to an existing chronological split.

    ``is_training`` marks the frozen chronological training partition exactly
    as already computed by the caller; this function never changes which
    rows are training vs. evaluation, it only annotates whether a training
    row's outcome window overlaps the evaluation partition's outcome windows
    (``purged_from_training``) and, separately, whether it would additionally
    be excluded under an ``embargo_steps`` sensitivity widening
    (``embargoed``).
    """
    source_index = np.asarray(source_index, dtype=np.int64)
    segment_id = np.asarray(segment_id, dtype=np.int64)
    is_training = np.asarray(is_training, dtype=bool)
    if not (len(source_index) == len(segment_id) == len(is_training)):
        raise EvaluationProtocolError("source_index, segment_id, and is_training must be the same length")
    starts, ends = target_interval(source_index, horizon_steps)
    cluster_ids = assign_target_clusters(source_index, horizon_steps, segment_id)

    held_out_start = starts[~is_training]
    held_out_end = ends[~is_training]
    purged = np.zeros(len(source_index), dtype=bool)
    embargoed = np.zeros(len(source_index), dtype=bool)
    if is_training.any() and (~is_training).any():
        purged[is_training] = purge_overlapping_training_rows(
            starts[is_training], ends[is_training], held_out_start, held_out_end)
        if embargo_steps > 0:
            wide_start, wide_end = apply_embargo(held_out_start, held_out_end, embargo_steps)
            embargoed[is_training] = purge_overlapping_training_rows(
                starts[is_training], ends[is_training], wide_start, wide_end)
        else:
            embargoed[is_training] = purged[is_training]

    metadata = pd.DataFrame({
        "target_start_index": starts,
        "target_end_index": ends,
        "target_cluster_id": cluster_ids,
        "purged_from_training": purged,
        "embargoed": embargoed,
    })
    train_total = int(is_training.sum())
    purged_count = int(purged[is_training].sum())
    embargoed_count = int(embargoed[is_training].sum())
    summary = PurgeSummary(
        train_rows_total=train_total,
        purged_rows=purged_count,
        embargoed_extra_purged_rows=max(0, embargoed_count - purged_count),
        remaining_train_rows=train_total - embargoed_count,
    )
    return metadata, summary

"""Shared canonical-data contracts used by simulation and frozen experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class ContractError(RuntimeError):
    """A canonical ordering or timestamp contract was violated."""


def canonical_pair_order(root: Path) -> tuple[str, ...]:
    """Load the only valid ten-instrument order from the canonical manifest."""
    manifest_path = root / "data_canonical" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs = tuple(manifest.get("instrument_index_order", ()))
    if len(pairs) != 10 or len(set(pairs)) != 10:
        raise ContractError("canonical manifest must define exactly ten unique instruments")
    pair_rows = manifest.get("pairs", {})
    if set(pairs) != set(pair_rows):
        raise ContractError("canonical manifest instrument order and pair set disagree")
    for index, pair in enumerate(pairs):
        if pair_rows[pair].get("index") != index:
            raise ContractError(f"canonical manifest index mismatch for {pair}")
    return pairs


def contiguous_60s(previous_ns: int, current_ns: int, dt_ns: int = 60_000_000_000) -> bool:
    """Whether a return/update may legally cross this observed timestamp edge."""
    return int(current_ns) - int(previous_ns) == dt_ns


def first_post_gap_mask(times_ns: np.ndarray, dt_ns: int = 60_000_000_000) -> np.ndarray:
    """Boolean mask for rows whose preceding observed arrival is noncontiguous."""
    if times_ns.ndim != 1 or len(times_ns) < 2 or not np.all(np.diff(times_ns) > 0):
        raise ContractError("timestamps must be a strictly increasing one-dimensional series")
    result = np.zeros(len(times_ns), dtype=bool)
    result[1:] = np.diff(times_ns) != dt_ns
    return result

"""Shared canonical-data contracts used by simulation and frozen experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class ContractError(RuntimeError):
    """A canonical ordering or timestamp contract was violated."""


INSTRUMENTS_CONFIG_RELATIVE_PATH = Path("engine") / "config" / "instruments.json"
GENERATED_MANIFEST_RELATIVE_PATH = Path("data") / "canonical" / "manifest.json"
INSTRUMENTS_CONFIG_SCHEMA_VERSION = "fxsim-instruments-v1"


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"{label} is missing: {path}") from exc
    except (json.JSONDecodeError, OSError) as exc:
        raise ContractError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must contain a JSON object: {path}")
    return value


def _validated_order(payload: dict[str, Any], label: str) -> tuple[str, ...]:
    raw_pairs = payload.get("instrument_index_order")
    if not isinstance(raw_pairs, list) or not all(isinstance(pair, str) for pair in raw_pairs):
        raise ContractError(f"{label} must define instrument_index_order as a string list")
    pairs = tuple(raw_pairs)
    if len(pairs) != 10 or len(set(pairs)) != 10:
        raise ContractError(f"{label} must define exactly ten unique instruments")
    return pairs


def canonical_pair_order(root: Path) -> tuple[str, ...]:
    """Load the tracked, versioned instrument-order contract.

    Generated canonical-data artifacts record this order but must never be the
    source of truth needed to import or test application code.
    """
    path = root / INSTRUMENTS_CONFIG_RELATIVE_PATH
    config = _load_json(path, "tracked instrument configuration")
    if config.get("schema_version") != INSTRUMENTS_CONFIG_SCHEMA_VERSION:
        raise ContractError("tracked instrument configuration has an unsupported schema_version")
    return _validated_order(config, "tracked instrument configuration")


def validate_generated_manifest(root: Path) -> dict[str, Any]:
    """Verify that a generated manifest faithfully records the tracked contract."""
    expected_pairs = canonical_pair_order(root)
    path = root / GENERATED_MANIFEST_RELATIVE_PATH
    manifest = _load_json(path, "generated canonical manifest")
    pairs = _validated_order(manifest, "generated canonical manifest")
    if pairs != expected_pairs:
        raise ContractError("generated manifest instrument order disagrees with tracked configuration")
    pair_rows = manifest.get("pairs")
    if not isinstance(pair_rows, dict):
        raise ContractError("generated manifest pairs must be an object")
    if set(pairs) != set(pair_rows):
        raise ContractError("generated manifest instrument order and pair set disagree")
    for index, pair in enumerate(pairs):
        row = pair_rows[pair]
        if not isinstance(row, dict) or row.get("index") != index:
            raise ContractError(f"generated manifest index mismatch for {pair}")
    return manifest


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

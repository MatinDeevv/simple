from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from contracts import canonical_pair_order, contiguous_60s, first_post_gap_mask
import quantum_kernel
import quantum_lindblad
import quantum_mps
import quantum_reservoir


ROOT = Path(__file__).resolve().parents[1]


def test_all_ten_pair_quantum_graphs_use_manifest_order() -> None:
    manifest = json.loads((ROOT / "data_canonical" / "manifest.json").read_text(encoding="utf-8"))
    expected = tuple(manifest["instrument_index_order"])
    assert canonical_pair_order(ROOT) == expected
    assert quantum_kernel.PAIRS == expected
    assert quantum_mps.PAIRS == expected
    assert quantum_reservoir.PAIRS == expected


def test_gap_contract_identifies_only_first_post_gap_row() -> None:
    dt = 60_000_000_000
    times = np.array([0, dt, 2 * dt, 8 * dt, 9 * dt], dtype=np.int64)
    assert contiguous_60s(times[1], times[2])
    assert not contiguous_60s(times[2], times[3])
    assert first_post_gap_mask(times).tolist() == [False, False, False, True, False]


def test_lindblad_cross_gap_counter_is_derived_from_emitted_rows() -> None:
    records = {
        "reason": ["gap_reset", "scored_contiguous", "scored_contiguous"],
        "previous_contiguous_60s": [False, True, False],
    }
    assert quantum_lindblad.count_cross_gap_state_updates(records) == 1

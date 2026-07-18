from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from contracts import (ContractError, canonical_pair_order, contiguous_60s,
                       first_post_gap_mask, validate_generated_manifest)
import estimate_coupling
import ingest
import quantum_kernel
import quantum_lindblad
import quantum_mps
import quantum_reservoir
import quantum_trajectories
import simulate_integrator as integrator


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PAIRS = (
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
    "USDCNH", "USDCHF", "EURGBP", "EURJPY", "GBPJPY",
)


def write_instrument_config(root: Path, order: tuple[str, ...] = EXPECTED_PAIRS) -> None:
    config_dir = root / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "instruments.json").write_text(
        json.dumps({"schema_version": "fxsim-instruments-v1", "instrument_index_order": list(order)}),
        encoding="utf-8",
    )


def write_manifest(root: Path, order: tuple[str, ...] = EXPECTED_PAIRS,
                   indices: tuple[int, ...] | None = None) -> None:
    manifest_dir = root / "data_canonical"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    actual_indices = indices if indices is not None else tuple(range(len(order)))
    payload = {
        "instrument_index_order": list(order),
        "pairs": {pair: {"index": index} for pair, index in zip(order, actual_indices, strict=True)},
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_tracked_order_bootstraps_without_generated_data(tmp_path: Path) -> None:
    write_instrument_config(tmp_path)
    assert canonical_pair_order(tmp_path) == EXPECTED_PAIRS
    assert not (tmp_path / "data_canonical" / "manifest.json").exists()


def test_tracked_order_rejects_an_unknown_config_schema(tmp_path: Path) -> None:
    write_instrument_config(tmp_path)
    config_path = tmp_path / "config" / "instruments.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["schema_version"] = "not-a-supported-schema"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ContractError, match="schema_version"):
        canonical_pair_order(tmp_path)


def test_all_pair_consumers_use_the_tracked_order() -> None:
    assert canonical_pair_order(ROOT) == EXPECTED_PAIRS
    assert tuple(ingest.PAIRS) == EXPECTED_PAIRS
    assert tuple(estimate_coupling.PAIRS) == EXPECTED_PAIRS
    assert quantum_kernel.PAIRS == EXPECTED_PAIRS
    assert quantum_mps.PAIRS == EXPECTED_PAIRS
    assert quantum_reservoir.PAIRS == EXPECTED_PAIRS
    assert quantum_lindblad.PAIRS == tuple(EXPECTED_PAIRS[index] for index in (0, 1, 5))
    assert quantum_trajectories.PAIRS == tuple(EXPECTED_PAIRS[index] for index in (0, 1, 5))
    assert integrator.PAIRS == tuple(EXPECTED_PAIRS[index] for index in integrator.PAIR_INDICES)


def test_generated_manifest_must_match_tracked_order_and_indices(tmp_path: Path) -> None:
    write_instrument_config(tmp_path)
    write_manifest(tmp_path)
    assert tuple(validate_generated_manifest(tmp_path)["instrument_index_order"]) == EXPECTED_PAIRS

    write_manifest(tmp_path, tuple(reversed(EXPECTED_PAIRS)))
    with pytest.raises(ContractError, match="disagrees"):
        validate_generated_manifest(tmp_path)

    write_manifest(tmp_path, EXPECTED_PAIRS, tuple(reversed(range(len(EXPECTED_PAIRS)))))
    with pytest.raises(ContractError, match="index mismatch"):
        validate_generated_manifest(tmp_path)


def test_ingestion_writes_the_tracked_order_into_its_generated_manifest(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_instrument_config(tmp_path)
    output_dir = tmp_path / "data_canonical"
    output_dir.mkdir()
    monkeypatch.setattr(ingest, "OUT_DIR", output_dir)
    entries = {pair: {"index": index} for index, pair in enumerate(EXPECTED_PAIRS)}
    ingest.write_manifest(entries)
    assert tuple(validate_generated_manifest(tmp_path)["instrument_index_order"]) == EXPECTED_PAIRS


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

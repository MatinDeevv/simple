from __future__ import annotations

from pathlib import Path

import pytest

from engine.experiments import dataset_binding as db
from engine.experiments import edge_tribunal as et
from engine.experiments.errors import DatasetBindingError

from test_edge_tribunal_evidence import make_dataset_manifest, run_pipeline_to_bound
from test_edge_tribunal_preregistration import make_plan


def _bind(tmp_path: Path, manifest: dict, name: str = "experiment") -> dict:
    return run_pipeline_to_bound(tmp_path, manifest=manifest, experiment_name=name)


def test_valid_manifest_binds_and_is_hash_consistent(tmp_path: Path) -> None:
    context = run_pipeline_to_bound(tmp_path)
    binding = context["binding"]
    assert db.verify_binding_integrity(binding)
    assert binding["holdout_claim_status"] == "SEALED_FOR_EXPERIMENT"
    assert binding["row_count_total"] == 1000000
    assert binding["execution_data_complete"] is False


def test_tampered_binding_fails_integrity() -> None:
    binding = {"binding_sha256": "0" * 64, "anything": 1}
    assert not db.verify_binding_integrity(binding)


@pytest.mark.parametrize("field,value,message", [
    ("frequency", "5min", "frequency mismatch"),
    ("price_side", "ASK", "price_side mismatch"),
    ("pair_scope", ["EURUSD", "USDJPY"], "pair_scope"),
])
def test_contract_mismatches_fail(tmp_path: Path, field: str, value, message: str) -> None:
    manifest = make_dataset_manifest()
    manifest[field] = value
    with pytest.raises(DatasetBindingError, match=message):
        _bind(tmp_path, manifest)


def test_missing_required_column_fails(tmp_path: Path) -> None:
    manifest = make_dataset_manifest()
    manifest["columns"] = ["timestamp_utc", "volume"]  # close_bid gone
    with pytest.raises(DatasetBindingError, match="required columns"):
        _bind(tmp_path, manifest)


def test_interval_not_covering_holdout_fails(tmp_path: Path) -> None:
    manifest = make_dataset_manifest()
    manifest["maximum_timestamp_utc"] = "2024-10-01T00:00:00+00:00"
    with pytest.raises(DatasetBindingError, match="does not cover"):
        _bind(tmp_path, manifest)


def test_untouched_interval_predating_cutoff_fails(tmp_path: Path) -> None:
    manifest = make_dataset_manifest()
    manifest["prior_inspection_cutoff_utc"] = "2024-09-01T00:00:00+00:00"
    with pytest.raises(DatasetBindingError, match="cannot be claimed untouched"):
        _bind(tmp_path, manifest)


def test_duplicate_file_logical_path_fails(tmp_path: Path) -> None:
    manifest = make_dataset_manifest()
    manifest["files"].append(dict(manifest["files"][0]))
    with pytest.raises(DatasetBindingError, match="duplicates"):
        _bind(tmp_path, manifest)


def test_windows_separators_normalize_to_same_logical_path(tmp_path: Path) -> None:
    manifest = make_dataset_manifest()
    manifest["files"][0]["logical_path"] = "data\\synthetic\\eurusd-1min.parquet"
    context = _bind(tmp_path, manifest)
    assert context["binding"]["files"][0]["logical_path"] == \
        "data/synthetic/eurusd-1min.parquet"


@pytest.mark.parametrize("path", [
    "../escape.parquet", "/absolute/path.parquet", "C:/windows/drive.parquet",
    "nested/../../escape.parquet",
])
def test_path_traversal_fails(tmp_path: Path, path: str) -> None:
    manifest = make_dataset_manifest()
    manifest["files"][0]["logical_path"] = path
    with pytest.raises(DatasetBindingError):
        _bind(tmp_path, manifest)


def test_missing_provenance_fails(tmp_path: Path) -> None:
    manifest = make_dataset_manifest()
    manifest["provenance"] = "   "
    with pytest.raises(DatasetBindingError, match="provenance"):
        _bind(tmp_path, manifest)


def test_invalid_file_hash_fails(tmp_path: Path) -> None:
    manifest = make_dataset_manifest()
    manifest["files"][0]["sha256"] = "UPPERCASE-NOT-HEX"
    with pytest.raises(DatasetBindingError, match="sha256"):
        _bind(tmp_path, manifest)


def test_negative_row_count_fails(tmp_path: Path) -> None:
    manifest = make_dataset_manifest()
    manifest["files"][0]["row_count"] = -5
    with pytest.raises(DatasetBindingError, match="row_count"):
        _bind(tmp_path, manifest)


def test_non_utc_timezone_fails(tmp_path: Path) -> None:
    manifest = make_dataset_manifest()
    manifest["timestamp_timezone"] = "Europe/Berlin"
    with pytest.raises(DatasetBindingError, match="UTC"):
        _bind(tmp_path, manifest)


def test_missing_execution_fields_lower_promotion_ceiling(tmp_path: Path) -> None:
    context = run_pipeline_to_bound(tmp_path)
    binding = context["binding"]
    assert binding["maximum_verdict_after_binding"] == "FORWARD_TEST_ELIGIBLE"
    assert binding["maximum_evidence_level"] == "LEVEL_1_HISTORICAL_BID_ONLY_RESEARCH"


def test_complete_execution_data_keeps_plan_ceiling_and_level_two(tmp_path: Path) -> None:
    manifest = make_dataset_manifest()
    manifest["execution_data"] = {field: True for field in db.EXECUTION_DATA_FIELDS}
    context = _bind(tmp_path, manifest)
    binding = context["binding"]
    assert binding["execution_data_complete"] is True
    assert binding["maximum_verdict_after_binding"] == "FORWARD_TEST_ELIGIBLE"
    assert binding["maximum_evidence_level"] == \
        "LEVEL_2_QUOTE_AND_COST_AWARE_HISTORICAL_RESEARCH"


def test_missing_manifest_field_fails(tmp_path: Path) -> None:
    manifest = make_dataset_manifest()
    del manifest["untouched_evidence"]
    with pytest.raises(DatasetBindingError, match="untouched_evidence"):
        _bind(tmp_path, manifest)


def test_binding_is_deterministic_for_fixed_inputs(tmp_path: Path) -> None:
    first = run_pipeline_to_bound(tmp_path / "one")["binding"]
    second = run_pipeline_to_bound(tmp_path / "two")["binding"]
    assert first == second  # same plan, timestamps, and deterministic IDs


def test_holdout_registry_records_the_claim(tmp_path: Path) -> None:
    context = run_pipeline_to_bound(tmp_path)
    from engine.experiments import holdout_registry as hr
    status = hr.holdout_status(context["registry_root"], context["binding"]["holdout_id"])
    assert status == hr.SEALED_FOR_EXPERIMENT

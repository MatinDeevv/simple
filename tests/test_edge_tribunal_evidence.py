from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from engine.experiments import edge_tribunal as et
from engine.experiments import evidence as evidence_module
from engine.experiments.canonical import load_strict_json_text, sha256_payload
from engine.experiments.errors import EvidenceValidationError

from test_edge_tribunal_preregistration import make_plan

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "examples" / "edge-tribunal" / "synthetic-dataset.json"

T_INIT = "2026-01-02T00:00:00+00:00"
T_SEAL = "2026-01-02T00:05:00+00:00"
T_BIND = "2026-01-02T00:10:00+00:00"
T_RECORD = "2026-01-02T00:15:00+00:00"
T_EVAL = "2026-01-02T00:20:00+00:00"


def make_dataset_manifest() -> dict:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def run_pipeline_to_bound(tmp_path: Path, *, plan: dict | None = None,
                          manifest: dict | None = None,
                          experiment_name: str = "experiment",
                          declare_forensic_reuse: bool = False) -> dict[str, Any]:
    """init + seal + bind-data against a per-test registry; returns context."""
    plan = plan or make_plan()
    manifest = manifest or make_dataset_manifest()
    experiment_dir = tmp_path / experiment_name
    registry_root = tmp_path / "registry"
    et.init_experiment(experiment_dir, plan, timestamp_utc=T_INIT)
    seal = et.seal_experiment(experiment_dir, timestamp_utc=T_SEAL)
    binding = et.bind_data(experiment_dir, manifest, registry_root=registry_root,
                           timestamp_utc=T_BIND,
                           declare_forensic_reuse=declare_forensic_reuse)
    return {"experiment_dir": experiment_dir, "registry_root": registry_root,
            "plan": plan, "seal": seal, "binding": binding, "manifest": manifest}


def make_evidence(plan: dict, seal: dict, binding: dict) -> dict[str, Any]:
    """A fully valid evidence bundle whose gates all pass."""
    from engine.experiments.robustness import planned_cell_registry
    policies = plan["entry_policy_contract"]["sensitivity_populations"]
    cells = [{"cell_id": cell_id, "dimensions": dimensions, "status": "scored",
              "sample_count": 300, "target_clusters": 120,
              "brier_improvement": 0.003, "log_loss_improvement": 0.006}
             for cell_id, dimensions in
             planned_cell_registry(plan["robustness_contract"]).items()]
    return {
        "evidence_version": "edge-tribunal-evidence-v1",
        "experiment": {
            "experiment_id": plan["identity"]["experiment_id"],
            "plan_sha256": seal["plan_sha256"],
            "seal_sha256": seal["seal_sha256"],
            "binding_sha256": binding["binding_sha256"]},
        "producer": {
            "code_commit_sha": plan["code_contract"]["commit_sha"],
            "model_contract_version": plan["code_contract"]["model_contract_version"],
            "source_file_sha256": plan["code_contract"]["source_file_sha256"],
            "configuration_sha256": plan["code_contract"]["configuration_sha256"],
            "execution_command": plan["code_contract"]["evidence_producer_command"],
            "python_version": "3.11.9",
            "dependency_snapshot_sha256": "5" * 64,
            "random_seeds": {"bootstrap": 7},
            "produced_at_utc": "2026-01-02T00:12:00+00:00"},
        "dataset": {
            "dataset_binding_sha256": binding["binding_sha256"],
            "dataset_file_sha256": {entry["logical_path"]: entry["sha256"]
                                    for entry in binding["files"]},
            "holdout_interval": dict(binding["holdout_interval"]),
            "row_count": 1000000, "valid_target_count": 950000},
        "population": {
            "training_rows": 8000, "oos_rows": 2000, "accepted_rows": 1200,
            "signal_episodes": 300, "target_clusters": 260, "causal_segments": 8,
            "class_counts": {"negative": 400, "neutral": 420, "positive": 380},
            "policy_counts": {policy: {"train_accepted": 5000, "oos_accepted": 900}
                              for policy in policies}},
        "primary_model": {
            "multiclass_brier": 0.612, "multiclass_log_loss": 1.021,
            "calibration_error": 0.031,
            "brier_improvement": 0.004,
            "brier_improvement_lower_bound_one_sided": 0.001,
            "brier_improvement_lower_bound_two_sided": 0.0008,
            "brier_improvement_upper_bound_two_sided": 0.0072,
            "log_loss_improvement": 0.008,
            "log_loss_improvement_lower_bound_one_sided": 0.002,
            "log_loss_improvement_lower_bound_two_sided": 0.0015,
            "log_loss_improvement_upper_bound_two_sided": 0.014,
            "probability_summary": {"rows": 1200, "min_probability": 0.02,
                                    "max_probability": 0.93, "mean_row_sum": 1.0}},
        "primary_comparator": {
            "identity": "train_only_conditional_three_class",
            "configuration": {"smoothing_alpha": 1.0, "minimum_cell_count": 30},
            "train_only_confirmed": True,
            "fallback_tier_counts": {"regime_conditional": 1100, "global_train_prior": 100},
            "multiclass_brier": 0.616, "multiclass_log_loss": 1.029},
        "negative_controls": {
            "global_train_prior": {"brier_improvement": 0.006,
                                   "log_loss_improvement": 0.011,
                                   "status": "behaved_as_required"},
            "time_shuffled_label_placebo": {"brier_improvement": -0.0031,
                                            "log_loss_improvement": -0.0064,
                                            "status": "behaved_as_required"},
            "sign_flipped_control": {"brier_improvement": -0.0042,
                                     "log_loss_improvement": -0.0087,
                                     "status": "behaved_as_required"},
            "no_regime_ablation": {"brier_improvement": -0.0011,
                                   "log_loss_improvement": -0.0019,
                                   "status": "behaved_as_required"},
            "static_factor_comparator": {"brier_improvement": -0.0009,
                                         "log_loss_improvement": -0.0014,
                                         "status": "behaved_as_required"},
            "persistence_comparator": {"brier_improvement": -0.0021,
                                       "log_loss_improvement": -0.0038,
                                       "status": "behaved_as_required"}},
        "robustness": {"cells": cells, "missing_data_sensitivity": 0.12},
        "concentration": {
            "largest_signal_episode_fraction": 0.08,
            "largest_target_cluster_fraction": 0.05,
            "largest_causal_segment_fraction": 0.3,
            "largest_day_fraction": 0.12,
            "largest_component_fraction": 0.4,
            "largest_regime_fraction": 0.55,
            "largest_fallback_tier_fraction": 0.85},
        "data_quality": {
            "missing_rows": 12, "duplicate_rows": 0, "gap_count": 12,
            "invalid_price_count": 0, "target_availability_rate": 0.97,
            "probability_validity_confirmed": True, "schema_validation_passed": True},
        "execution_data": dict(binding["execution_data"]),
        "multiplicity": {
            "family_id": plan["multiple_testing_contract"]["family_id"],
            "family_size": plan["multiple_testing_contract"]["family_size"],
            "hypothesis_ids": ["h1-basket-conditional-improvement"],
            "p_values": [0.011], "primary_hypothesis_index": 0},
        "artifacts": {
            "summary_path": "artifacts/runs/synthetic-example/summary.json",
            "manifest_path": "artifacts/runs/synthetic-example/manifest.json",
            "artifact_sha256": {
                "artifacts/runs/synthetic-example/summary.json": "6" * 64,
                "artifacts/runs/synthetic-example/manifest.json": "7" * 64},
            "audit_sha256": "8" * 64,
            "test_evidence_sha256": {"pytest-full-suite": "9" * 64}},
    }


def finalize_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    evidence["evidence_sha256"] = sha256_payload(body)
    return evidence


def _errors(context: dict[str, Any], evidence: dict[str, Any]) -> list[str]:
    return evidence_module.evidence_errors(
        evidence, plan=context["plan"], seal=context["seal"], binding=context["binding"])


@pytest.fixture()
def bound(tmp_path: Path) -> dict[str, Any]:
    return run_pipeline_to_bound(tmp_path)


def test_valid_evidence_passes(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    assert _errors(bound, evidence) == []
    validated = evidence_module.validate_evidence(
        evidence, plan=bound["plan"], seal=bound["seal"], binding=bound["binding"])
    assert validated["evidence_sha256"] == sha256_payload(
        {key: value for key, value in evidence.items() if key != "evidence_sha256"})


def test_mismatched_experiment_id_fails(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["experiment"]["experiment_id"] = "00000000-0000-4000-8000-00000000ffff"
    assert any("experiment_id" in error for error in _errors(bound, evidence))


@pytest.mark.parametrize("field", ["plan_sha256", "seal_sha256", "binding_sha256"])
def test_mismatched_chain_hashes_fail(bound: dict[str, Any], field: str) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["experiment"][field] = "0" * 64
    assert any(field in error for error in _errors(bound, evidence))


def test_mismatched_code_commit_fails(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["producer"]["code_commit_sha"] = "f" * 40
    assert any("code_commit_sha" in error for error in _errors(bound, evidence))


def test_mismatched_configuration_fails(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["producer"]["configuration_sha256"] = "f" * 64
    assert any("configuration_sha256" in error for error in _errors(bound, evidence))


def test_mismatched_dataset_hash_fails(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["dataset"]["dataset_binding_sha256"] = "0" * 64
    assert any("dataset_binding_sha256" in error for error in _errors(bound, evidence))


def test_missing_mandatory_comparator_fails(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    del evidence["negative_controls"]["persistence_comparator"]
    errors = _errors(bound, evidence)
    assert any("persistence_comparator" in error and "omit" in error for error in errors)


def test_missing_mandatory_policy_fails(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["robustness"]["cells"] = [
        cell for cell in evidence["robustness"]["cells"]
        if cell["dimensions"]["entry_policy"] != "non_overlapping_basket"]
    assert any("non_overlapping_basket" in error for error in _errors(bound, evidence))


def test_missing_mandatory_block_length_fails(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["robustness"]["cells"] = [
        cell for cell in evidence["robustness"]["cells"]
        if cell["dimensions"]["block_length"] != 390]
    assert any("390" in error for error in _errors(bound, evidence))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_metric_fails(bound: dict[str, Any], bad: float) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["primary_model"]["multiclass_brier"] = bad
    errors = _errors(bound, evidence)
    assert errors
    assert any("non-canonical" in error or "finite" in error for error in errors)


def test_probability_out_of_range_fails(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["primary_model"]["probability_summary"]["min_probability"] = -0.01
    assert any("min_probability" in error for error in _errors(bound, evidence))
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["primary_model"]["probability_summary"]["max_probability"] = 1.01
    assert any("max_probability" in error for error in _errors(bound, evidence))


def test_probability_rows_not_summing_to_one_fail(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["primary_model"]["probability_summary"]["mean_row_sum"] = 1.01
    assert any("mean_row_sum" in error for error in _errors(bound, evidence))


def test_class_counts_not_summing_fails(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["population"]["class_counts"]["neutral"] = 1
    assert any("class_counts" in error for error in _errors(bound, evidence))


def test_accepted_rows_greater_than_oos_fails(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["population"]["accepted_rows"] = 5000
    assert any("exceeds oos_rows" in error for error in _errors(bound, evidence))


def test_target_clusters_greater_than_accepted_fails(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["population"]["target_clusters"] = 1201
    assert any("exceeds accepted_rows" in error for error in _errors(bound, evidence))


def test_negative_sample_count_fails(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["population"]["oos_rows"] = -1
    assert any("oos_rows" in error for error in _errors(bound, evidence))


def test_confidence_bounds_in_wrong_order_fail(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["primary_model"]["brier_improvement_lower_bound_two_sided"] = 0.02
    evidence["primary_model"]["brier_improvement_upper_bound_two_sided"] = 0.01
    assert any("lower bound exceeds upper bound" in error
               for error in _errors(bound, evidence))


def test_one_sided_lower_bound_above_point_estimate_fails(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["primary_model"]["brier_improvement_lower_bound_one_sided"] = 0.9
    assert any("exceeds the point estimate" in error for error in _errors(bound, evidence))


def test_control_status_cannot_be_unreported(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    del evidence["negative_controls"]["sign_flipped_control"]["status"]
    assert any("not reported never means passed" in error
               for error in _errors(bound, evidence))


def test_multiline_execution_command_is_rejected(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["producer"]["execution_command"] = "echo ok\nrm -rf /"
    assert any("never executes it" in error for error in _errors(bound, evidence))


def test_evidence_module_never_executes_anything() -> None:
    source = Path(evidence_module.__file__).read_text(encoding="utf-8")
    for forbidden in ("subprocess", "os.system", "shell=True", "pickle", "__import__",
                      "importlib"):
        assert forbidden not in source


def test_tampered_self_hash_fails(bound: dict[str, Any]) -> None:
    evidence = finalize_evidence(make_evidence(bound["plan"], bound["seal"],
                                               bound["binding"]))
    evidence["population"]["oos_rows"] = 1999
    assert any("evidence_sha256" in error for error in _errors(bound, evidence))


def test_record_evidence_rejects_invalid_bundle(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["experiment"]["plan_sha256"] = "0" * 64
    with pytest.raises(EvidenceValidationError):
        et.record_evidence(bound["experiment_dir"], evidence, timestamp_utc=T_RECORD)
    # The failed ingestion left the experiment exactly where it was.
    assert et.current_state(bound["experiment_dir"]) == "DATA_BOUND"


def test_duplicate_robustness_cell_ids_fail(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["robustness"]["cells"].append(dict(evidence["robustness"]["cells"][0]))
    assert any("duplicates" in error for error in _errors(bound, evidence))


def test_different_cell_ids_cannot_reuse_identical_dimensions(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["robustness"]["cells"][1]["dimensions"] = dict(
        evidence["robustness"]["cells"][0]["dimensions"])
    errors = _errors(bound, evidence)
    assert any("cell_id does not match" in error for error in errors)


def test_load_strict_json_rejects_duplicate_keys() -> None:
    from engine.experiments.errors import CanonicalJsonError
    with pytest.raises(CanonicalJsonError, match="duplicate"):
        load_strict_json_text('{"a": 1, "a": 2}')


def test_producer_cannot_forge_independent_verification(bound: dict[str, Any]) -> None:
    evidence = make_evidence(bound["plan"], bound["seal"], bound["binding"])
    evidence["independent_verification"] = {"metrics_recomputed": True}
    with pytest.raises(EvidenceValidationError, match="Tribunal-generated"):
        et.record_evidence(bound["experiment_dir"], evidence, timestamp_utc=T_RECORD)

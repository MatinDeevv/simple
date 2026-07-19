import json
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
from engine.experiments.evaluation_rows import load_evaluation_rows
from engine.experiments.evidence_recompute import (recompute, assert_summary_matches,
                                                    deterministic_block_bootstrap)
from engine.experiments.errors import EvidenceValidationError
from engine.experiments import edge_tribunal as et
from engine.experiments.robustness import planned_cell_registry
from engine.experiments.robustness_evidence import recompute_robustness_rows, VERSION
from engine.experiments.canonical import sha256_payload
from test_edge_tribunal_preregistration import make_plan
from test_edge_tribunal_evidence import make_dataset_manifest, make_evidence, T_INIT, T_SEAL, T_BIND, T_RECORD, T_EVAL


def _row(row_id, probs, source_index=1):
    return {"row_id": row_id, "source_index": source_index, "timestamp_utc": "2024-01-01T00:00:00+00:00",
            "segment_id": "s", "target_cluster_id": "c", "signal_episode_id": "e",
            "policy_population": "p", "split_boundary": "reset", "class_label": "positive",
            "model_probabilities": probs, "comparator_probabilities": [0.2, 0.6, 0.2],
            "regime": "all", "component": "basket", "fallback_tier": "global"}


def test_rows_and_metrics_are_independently_recomputed(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(_row("r1", [0.1, .8, .1])) + "\n", encoding="utf-8")
    rows = load_evaluation_rows(path, class_order=["neutral", "positive", "negative"])
    result = recompute(rows, class_order=["neutral", "positive", "negative"])
    assert result["brier_improvement"] > 0
    assert_summary_matches(result, dict(result))
    with pytest.raises(EvidenceValidationError, match="differs"):
        assert_summary_matches(result, dict(result, model_brier=99))


def test_invalid_probability_row_fails(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(_row("r1", [0.1, .8, .8])) + "\n", encoding="utf-8")
    with pytest.raises(EvidenceValidationError, match="invalid model_probabilities"):
        load_evaluation_rows(path, class_order=["neutral", "positive", "negative"])


def test_block_bootstrap_is_deterministic(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(_row(f"r{i}", [0.1, .8, .1], i))
                              for i in range(8)) + "\n", encoding="utf-8")
    rows = load_evaluation_rows(path, class_order=["neutral", "positive", "negative"])
    kwargs = dict(class_order=["neutral", "positive", "negative"], block_length=2,
                  replicate_count=20, seed=7)
    assert deterministic_block_bootstrap(rows, **kwargs) == deterministic_block_bootstrap(rows, **kwargs)


def test_duplicate_source_index_and_backward_segment_time_fail(tmp_path: Path):
    first = _row("r1", [0.1, .8, .1], 2)
    second = _row("r2", [0.1, .8, .1], 2)
    path = tmp_path / "duplicate.jsonl"
    path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
    with pytest.raises(EvidenceValidationError, match="duplicate source_index"):
        load_evaluation_rows(path, class_order=["neutral", "positive", "negative"])
    second["source_index"] = 3
    first["timestamp_utc"] = "2024-01-02T00:00:00+00:00"
    path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
    with pytest.raises(EvidenceValidationError, match="moves backward"):
        load_evaluation_rows(path, class_order=["neutral", "positive", "negative"])


def test_policy_boundary_membership_and_day_concentration(tmp_path: Path):
    first = _row("r1", [0.1, .8, .1], 1)
    second = _row("r2", [0.1, .8, .1], 2)
    second["timestamp_utc"] = "2024-01-02T00:00:00+00:00"
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
    rows = load_evaluation_rows(path, class_order=["neutral", "positive", "negative"],
                                allowed_policies=["p"], allowed_boundaries=["reset"])
    assert recompute(rows, class_order=["neutral", "positive", "negative"])[
        "largest_day_fraction"] == .5
    first["policy_population"] = "invented"
    path.write_text(json.dumps(first) + "\n", encoding="utf-8")
    with pytest.raises(EvidenceValidationError, match="unregistered policy"):
        load_evaluation_rows(path, class_order=["neutral", "positive", "negative"],
                             allowed_policies=["p"], allowed_boundaries=["reset"])


def test_fully_physical_synthetic_case_is_forward_eligible_only(tmp_path: Path):
    plan = make_plan()
    robust = plan["robustness_contract"]
    robust.update({"required_time_slices": ["whole_oos"],
                   "required_regime_slices": ["all"],
                   "required_block_lengths": [1],
                   "required_perturbations": ["baseline"],
                   "minimum_samples_per_cell": 3,
                   "mandatory_cells": []})
    robust["mandatory_cells"] = [next(iter(planned_cell_registry(robust)))]
    plan["uncertainty_contract"].update({"block_sizes": [1], "replicate_count": 10})
    experiment = tmp_path / "experiment"; registry = tmp_path / "registry"
    data_files = [tmp_path / "eur.bin", tmp_path / "gbp.bin"]
    for index, path in enumerate(data_files):
        path.write_bytes(f"synthetic-{index}".encode())
    manifest = make_dataset_manifest()
    for entry, path in zip(manifest["files"], data_files):
        entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    et.init_experiment(experiment, plan, timestamp_utc=T_INIT)
    seal = et.seal_experiment(experiment, timestamp_utc=T_SEAL)
    binding = et.bind_data(
        experiment, manifest, registry_root=registry, timestamp_utc=T_BIND,
        dataset_root=tmp_path,
        physical_file_bindings=[{"physical_path": str(path),
                                 "logical_path": entry["logical_path"]}
                                for entry, path in zip(manifest["files"], data_files)])
    classes = plan["target_contract"]["class_order"]
    eval_path = tmp_path / "evaluation.jsonl"
    eval_rows = []
    for index in range(600):
        label_index = index % 3
        model = [.1, .1, .1]; comparator = [.2, .2, .2]
        model[label_index] = .8; comparator[label_index] = .6
        timestamp = (datetime(2024, 7, 1, tzinfo=timezone.utc)
                     + timedelta(days=index // 60, minutes=index % 60)).isoformat()
        eval_rows.append({"row_id": f"r{index}", "source_index": index,
                          "timestamp_utc": timestamp,
                          "segment_id": f"s{index // 100}", "target_cluster_id": f"c{index}",
                          "signal_episode_id": f"e{index}",
                          "policy_population": "independent_research_entries",
                          "split_boundary": "reset_at_oos_start", "class_label": classes[label_index],
                          "model_probabilities": model, "comparator_probabilities": comparator,
                          "regime": "low" if index % 2 else "high",
                          "component": f"component-{index % 3}",
                          "fallback_tier": f"tier-{index % 2}"})
    eval_path.write_text("\n".join(json.dumps(row) for row in eval_rows) + "\n", encoding="utf-8")
    rows = load_evaluation_rows(eval_path, class_order=classes,
                                allowed_policies=plan["entry_policy_contract"]["sensitivity_populations"],
                                allowed_boundaries=plan["entry_policy_contract"]["split_boundary_policies"])
    calculated = recompute(rows, class_order=classes)
    robust_path = tmp_path / "robustness.jsonl"
    lines = []
    for cell_id, dimensions in planned_cell_registry(robust).items():
        lines.append(json.dumps({"version": VERSION, "kind": "cell", "cell_id": cell_id,
                                 "dimensions": dimensions, "status": "scored"}))
        for index in range(3):
            label_index = index % 3
            model = [.1, .1, .1]; comparator = [.2, .2, .2]
            model[label_index] = .8; comparator[label_index] = .6
            lines.append(json.dumps({"version": VERSION, "kind": "row", "cell_id": cell_id,
                                     "row_id": f"{cell_id}-{index}",
                                     "target_cluster_id": f"{cell_id}-c{index}",
                                     "class_label": classes[label_index],
                                     "model_probabilities": model,
                                     "comparator_probabilities": comparator}))
    robust_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    robust_cells = recompute_robustness_rows(robust_path, contract=robust, class_order=classes)
    summary = tmp_path / "summary.json"; summary.write_text("{}", encoding="utf-8")
    run_manifest = tmp_path / "manifest.json"; run_manifest.write_text("{}", encoding="utf-8")
    log = tmp_path / "pytest.log"; log.write_text("synthetic tests passed\n", encoding="utf-8")
    receipt_body = {"receipt_version": "test-receipt-v2",
                    "command_argv": plan["code_contract"]["required_test_commands"][0],
                    "exit_code": 0, "commit_sha": plan["code_contract"]["commit_sha"],
                    "started_at_utc": "2026-01-02T00:11:00+00:00",
                    "completed_at_utc": "2026-01-02T00:12:00+00:00",
                    "python_version": "3.11", "dependency_snapshot_sha256": "b" * 64,
                    "logical_log_path": "logs/pytest.log", "runner_version": "synthetic-v2",
                    "working_tree_status": "clean", "command_environment_policy": "sealed-v1",
                    "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()}
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(dict(receipt_body,
                                       receipt_sha256=sha256_payload(receipt_body))), encoding="utf-8")
    evidence = make_evidence(plan, seal, binding)
    evidence["population"].update({"accepted_rows": 600, "signal_episodes": 600,
                                   "target_clusters": 600, "causal_segments": 6,
                                   "class_counts": calculated["class_counts"]})
    evidence["primary_model"].update({"multiclass_brier": calculated["model_brier"],
                                      "multiclass_log_loss": calculated["model_log_loss"],
                                      "brier_improvement": calculated["brier_improvement"],
                                      "log_loss_improvement": calculated["log_loss_improvement"]})
    evidence["primary_model"]["probability_summary"].update(
        {"rows": 600, "min_probability": .1, "max_probability": .8})
    evidence["primary_comparator"].update(
        {"multiclass_brier": calculated["comparator_brier"],
         "multiclass_log_loss": calculated["comparator_log_loss"]})
    for key in evidence["concentration"]:
        if key in calculated:
            evidence["concentration"][key] = calculated[key]
    bounds = deterministic_block_bootstrap(
        rows, class_order=classes, block_length=1, replicate_count=10, seed=7,
        confidence_level=.95, two_sided_confidence_level=.9)
    mapping = {"brier_lower_one_sided": "brier_improvement_lower_bound_one_sided",
               "brier_lower_two_sided": "brier_improvement_lower_bound_two_sided",
               "brier_upper_two_sided": "brier_improvement_upper_bound_two_sided",
               "log_loss_lower_one_sided": "log_loss_improvement_lower_bound_one_sided",
               "log_loss_lower_two_sided": "log_loss_improvement_lower_bound_two_sided",
               "log_loss_upper_two_sided": "log_loss_improvement_upper_bound_two_sided"}
    evidence["primary_model"]["bootstrap_by_block_length"] = {
        "1": {submitted: bounds[key] for key, submitted in mapping.items()}}
    evidence["primary_model"].update({submitted: bounds[key] for key, submitted in mapping.items()})
    evidence["robustness"]["cells"] = robust_cells
    bindings = [{"physical_path": str(path), "logical_path": logical} for path, logical in (
        (summary, evidence["artifacts"]["summary_path"]),
        (run_manifest, evidence["artifacts"]["manifest_path"]),
        (eval_path, "artifacts/evaluation.jsonl"),
        (robust_path, "artifacts/robustness.jsonl"))]
    et.record_evidence(
        experiment, evidence, timestamp_utc=T_RECORD, artifact_root=tmp_path,
        artifact_bindings=bindings, evaluation_rows_logical_path="artifacts/evaluation.jsonl",
        robustness_rows_logical_path="artifacts/robustness.jsonl",
        test_receipts=[{"receipt_path": str(receipt), "log_path": str(log)}])
    verdict = et.evaluate(experiment, registry_root=registry, timestamp_utc=T_EVAL)
    assert verdict["verdict"] == "FORWARD_TEST_ELIGIBLE"
    assert verdict["trading_authorization"] is False

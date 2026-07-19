from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from engine.experiments import edge_tribunal as et
from engine.experiments import verdict as verdict_module

from test_edge_tribunal_evidence import (
    T_EVAL,
    T_RECORD,
    finalize_evidence,
    make_evidence,
    run_pipeline_to_bound,
)
from test_edge_tribunal_preregistration import make_plan


def run_to_verdict(tmp_path: Path, *, mutate=None, plan: dict | None = None,
                   experiment_name: str = "experiment",
                   declare_forensic_reuse: bool = False,
                   tamper=None) -> dict[str, Any]:
    context = run_pipeline_to_bound(tmp_path, plan=plan,
                                    experiment_name=experiment_name,
                                    declare_forensic_reuse=declare_forensic_reuse)
    evidence = make_evidence(context["plan"], context["seal"], context["binding"])
    if mutate is not None:
        mutate(evidence)
    finalize_evidence(evidence)
    et.record_evidence(context["experiment_dir"], evidence, timestamp_utc=T_RECORD)
    if tamper is not None:
        tamper(context["experiment_dir"])
    return et.evaluate(context["experiment_dir"],
                       registry_root=context["registry_root"], timestamp_utc=T_EVAL)


def test_fully_passed_fresh_experiment_is_forward_test_eligible(tmp_path: Path) -> None:
    verdict = run_to_verdict(tmp_path)
    assert verdict["verdict"] == "FORWARD_TEST_ELIGIBLE"
    assert verdict["validity_status"] == "VALID"
    assert verdict["maximum_next_stage"] == "LEVEL_3_PREREGISTERED_PAPER_FORWARD_TEST"
    assert verdict["trading_authorization"] is False
    # It still explains why nothing higher was possible.
    assert any("execution" in reason for reason in verdict["why_not_higher"])


def test_tampered_evidence_produces_invalid_experiment(tmp_path: Path) -> None:
    def tamper(experiment_dir: Path) -> None:
        version_dir = experiment_dir / "versions" / et.current_version_name(experiment_dir)
        payload = json.loads((version_dir / "evidence.json").read_text(encoding="utf-8"))
        payload["producer"]["configuration_sha256"] = "f" * 64  # post-hoc config swap
        (version_dir / "evidence.json").write_text(json.dumps(payload), encoding="utf-8")

    verdict = run_to_verdict(tmp_path, tamper=tamper)
    assert verdict["verdict"] == "INVALID_EXPERIMENT"
    assert verdict["validity_status"] == "INVALID"
    assert any("hard_configuration_match" in reason or "configuration" in reason
               for reason in verdict["invalid_reasons"])


def test_clear_gate_failure_produces_rejected(tmp_path: Path) -> None:
    def losing_model(evidence: dict) -> None:
        evidence["primary_model"]["brier_improvement"] = -0.004
        evidence["primary_model"]["brier_improvement_lower_bound_one_sided"] = -0.006
        evidence["primary_model"]["brier_improvement_lower_bound_two_sided"] = -0.007
        evidence["primary_model"]["brier_improvement_upper_bound_two_sided"] = -0.001
        evidence["primary_model"]["log_loss_improvement"] = -0.008
        evidence["primary_model"]["log_loss_improvement_lower_bound_one_sided"] = -0.012
        evidence["primary_model"]["log_loss_improvement_lower_bound_two_sided"] = -0.013
        evidence["primary_model"]["log_loss_improvement_upper_bound_two_sided"] = -0.002

    verdict = run_to_verdict(tmp_path, mutate=losing_model)
    assert verdict["verdict"] == "REJECTED"
    assert verdict["validity_status"] == "VALID"
    failed_ids = {item["gate_id"] for item in verdict["failed_gates"]}
    assert "primary_brier_improvement" in failed_ids
    # Every failed gate carries an explanation.
    assert all(item["reason"] for item in verdict["failed_gates"])


def test_underpowered_evidence_produces_inconclusive(tmp_path: Path) -> None:
    def underpowered(evidence: dict) -> None:
        evidence["population"]["target_clusters"] = 20  # below the plan's 30

    verdict = run_to_verdict(tmp_path, mutate=underpowered)
    assert verdict["verdict"] == "INCONCLUSIVE"
    insufficient_ids = {item["gate_id"] for item in verdict["insufficient_gates"]}
    assert "power_min_target_clusters" in insufficient_ids
    assert any("power" in reason or "not proof" in reason
               for reason in verdict["why_not_higher"])


def test_reused_holdout_with_passing_statistics_is_research_only(tmp_path: Path) -> None:
    first = run_to_verdict(tmp_path, experiment_name="first")
    assert first["verdict"] == "FORWARD_TEST_ELIGIBLE"
    second_plan = make_plan()
    second_plan["identity"]["experiment_id"] = "00000000-0000-4000-8000-000000000042"
    second_plan["identity"]["title"] = "renamed experiment cannot evade reuse"
    verdict = run_to_verdict(tmp_path, plan=second_plan, experiment_name="second",
                             declare_forensic_reuse=True)
    assert verdict["verdict"] == "RESEARCH_ONLY"
    assert any("forensic" in reason for reason in verdict["why_not_higher"])


def test_amended_plan_blocks_clean_promotion(tmp_path: Path) -> None:
    from engine.experiments import preregistration as prereg
    parent = make_plan()
    child = prereg.create_amendment(
        parent, new_experiment_id="00000000-0000-4000-8000-000000000077",
        amendment_reason="threshold reconsidered after internal review",
        created_at_utc="2026-01-02T00:00:00+00:00",
        changes={"concentration_limits": {**parent["concentration_limits"],
                                          "max_single_day_fraction": 0.25}})
    verdict = run_to_verdict(tmp_path, plan=child, experiment_name="amended")
    assert verdict["verdict"] == "RESEARCH_ONLY"
    assert verdict["amendment"] is True
    assert any("amend" in reason for reason in verdict["why_not_higher"])


def test_raw_pass_adjusted_fail_produces_rejected(tmp_path: Path) -> None:
    plan = make_plan()
    plan["multiple_testing_contract"].update({
        "correction_method": "BONFERRONI", "family_size": 5,
        "registered_sibling_experiment_ids": ["sib-1", "sib-2", "sib-3", "sib-4"]})

    def five_hypotheses(evidence: dict) -> None:
        evidence["multiplicity"].update({
            "family_size": 5,
            "hypothesis_ids": ["h1", "h2", "h3", "h4", "h5"],
            "p_values": [0.03, 0.5, 0.6, 0.7, 0.8]})  # raw pass, adjusted fail

    verdict = run_to_verdict(tmp_path, plan=plan, mutate=five_hypotheses)
    assert verdict["verdict"] == "REJECTED"
    failed_ids = {item["gate_id"] for item in verdict["failed_gates"]}
    assert "multiplicity_adjusted_primary_gate" in failed_ids or \
        "multiplicity_adjusted_primary" in failed_ids


def test_bid_only_evidence_can_never_produce_a_live_verdict() -> None:
    assert set(verdict_module.FORBIDDEN_VERDICTS).isdisjoint(verdict_module.VERDICTS)
    for forbidden in verdict_module.FORBIDDEN_VERDICTS:
        assert forbidden not in verdict_module.VERDICTS


def test_verdict_is_deterministic(tmp_path: Path) -> None:
    first = run_to_verdict(tmp_path / "one")
    second = run_to_verdict(tmp_path / "two")
    assert first == second  # byte-identical payload including verdict_sha256


def test_verdict_integrity_verification() -> None:
    assert not verdict_module.verify_verdict_integrity({})
    assert not verdict_module.verify_verdict_integrity({"verdict_sha256": "0" * 64})


def test_tampered_verdict_fails_integrity(tmp_path: Path) -> None:
    verdict = run_to_verdict(tmp_path)
    assert verdict_module.verify_verdict_integrity(verdict)
    tampered = dict(verdict, verdict="FORWARD_TEST_ELIGIBLE",
                    maximum_next_stage="LEVEL_4_MICRO_LIVE_HUMAN_REVIEW")
    assert not verdict_module.verify_verdict_integrity(tampered)
    hijacked = dict(verdict)
    hijacked["trading_authorization"] = True
    assert not verdict_module.verify_verdict_integrity(hijacked)


def test_dangerous_concentration_produces_rejected(tmp_path: Path) -> None:
    def concentrated(evidence: dict) -> None:
        evidence["concentration"]["largest_signal_episode_fraction"] = 0.75

    verdict = run_to_verdict(tmp_path, mutate=concentrated)
    assert verdict["verdict"] == "REJECTED"
    failed_ids = {item["gate_id"] for item in verdict["failed_gates"]}
    assert "evidence_concentration" in failed_ids or \
        "concentration_single_episode" in failed_ids


def test_no_boolean_success_shortcut_in_verdict(tmp_path: Path) -> None:
    # Every outcome is an explicit enum plus explanations, never a bare
    # top-level success flag.
    verdict = run_to_verdict(tmp_path)
    assert "success" not in verdict and "passed" not in verdict
    assert verdict["verdict"] in verdict_module.VERDICTS
    import inspect
    source = inspect.getsource(verdict_module)
    assert '"success"' not in source
    assert "PROFITABLE" not in verdict_module.VERDICTS

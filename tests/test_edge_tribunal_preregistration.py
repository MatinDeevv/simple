from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from engine.experiments import preregistration as prereg
from engine.experiments.errors import PlanValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "examples" / "edge-tribunal" / "synthetic-plan.json"


def make_plan() -> dict:
    """Fresh deep copy of the committed valid synthetic plan."""
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def test_valid_plan_passes() -> None:
    plan = make_plan()
    assert prereg.plan_errors(plan) == []
    prereg.validate_plan(plan)


def test_empty_plan_fails() -> None:
    errors = prereg.plan_errors({})
    assert errors
    assert any("missing required section" in error for error in errors)
    with pytest.raises(PlanValidationError):
        prereg.validate_plan({})


def test_non_object_plan_fails() -> None:
    assert prereg.plan_errors([]) == ["plan must be a JSON object"]


def test_missing_primary_comparator_fails() -> None:
    plan = make_plan()
    del plan["primary_comparator"]
    assert any("primary_comparator" in error for error in prereg.plan_errors(plan))


def test_missing_mandatory_secondary_comparator_kind_fails() -> None:
    plan = make_plan()
    plan["secondary_comparators"] = [
        item for item in plan["secondary_comparators"]
        if item["kind"] != "time_shuffled_label_placebo"]
    assert any("time_shuffled_label_placebo" in error for error in prereg.plan_errors(plan))


def test_missing_rejection_condition_fails() -> None:
    plan = make_plan()
    plan["automatic_rejection_conditions"].remove("holdout_reuse")
    assert any("holdout_reuse" in error for error in prereg.plan_errors(plan))


def test_vague_gate_fails() -> None:
    plan = make_plan()
    plan["acceptance_gates"][0]["threshold"] = "looks promising"
    errors = prereg.plan_errors(plan)
    assert any("machine-evaluable" in error for error in errors)


def test_non_finite_threshold_fails() -> None:
    plan = make_plan()
    plan["acceptance_gates"][0]["threshold"] = math.nan
    assert any("finite" in error for error in prereg.plan_errors(plan))
    plan["acceptance_gates"][0]["threshold"] = math.inf
    assert any("finite" in error for error in prereg.plan_errors(plan))


def test_invalid_utc_timestamp_fails() -> None:
    plan = make_plan()
    plan["identity"]["created_at_utc"] = "2026-01-02T00:00:00"  # naive
    assert any("created_at_utc" in error for error in prereg.plan_errors(plan))


def test_invalid_commit_sha_fails() -> None:
    plan = make_plan()
    plan["code_contract"]["commit_sha"] = "not-a-sha"
    assert any("commit_sha" in error for error in prereg.plan_errors(plan))


def test_uppercase_commit_sha_fails() -> None:
    plan = make_plan()
    plan["code_contract"]["commit_sha"] = plan["code_contract"]["commit_sha"].upper()
    assert any("commit_sha" in error for error in prereg.plan_errors(plan))


def test_invalid_configuration_hash_fails() -> None:
    plan = make_plan()
    plan["code_contract"]["configuration_sha256"] = "abc"
    assert any("configuration_sha256" in error for error in prereg.plan_errors(plan))


def test_plan_hash_is_deterministic_and_key_order_independent() -> None:
    plan = make_plan()
    first = prereg.plan_sha256(plan)
    assert first == prereg.plan_sha256(make_plan())
    reordered = dict(reversed(list(plan.items())))
    reordered["identity"] = dict(reversed(list(plan["identity"].items())))
    assert prereg.plan_sha256(reordered) == first


def test_plan_hash_changes_when_a_gate_changes() -> None:
    plan = make_plan()
    original = prereg.plan_sha256(plan)
    plan["acceptance_gates"][0]["threshold"] = 0.123
    assert prereg.plan_sha256(plan) != original


def test_required_primary_gate_ids_are_mandatory() -> None:
    plan = make_plan()
    plan["acceptance_gates"] = [gate for gate in plan["acceptance_gates"]
                                if gate["gate_id"] != "primary_brier_improvement"]
    assert any("primary_brier_improvement" in error for error in prereg.plan_errors(plan))


def test_entry_policy_order_is_frozen() -> None:
    plan = make_plan()
    plan["entry_policy_contract"]["sensitivity_populations"] = list(
        reversed(plan["entry_policy_contract"]["sensitivity_populations"]))
    assert any("sensitivity_populations" in error for error in prereg.plan_errors(plan))


def test_untouched_window_before_inspection_cutoff_fails() -> None:
    plan = make_plan()
    plan["data_contract"]["untouched_evaluation_start_utc"] = "2024-01-01T00:00:00+00:00"
    assert any("inspection cutoff" in error for error in prereg.plan_errors(plan))


def test_none_single_primary_requires_family_of_one() -> None:
    plan = make_plan()
    plan["multiple_testing_contract"]["family_size"] = 2
    errors = prereg.plan_errors(plan)
    assert any("NONE_SINGLE_PRIMARY" in error for error in errors)


def test_family_size_must_match_registered_siblings() -> None:
    plan = make_plan()
    plan["multiple_testing_contract"]["correction_method"] = "HOLM_BONFERRONI"
    plan["multiple_testing_contract"]["family_size"] = 3
    plan["multiple_testing_contract"]["registered_sibling_experiment_ids"] = ["sib-1"]
    assert any("registered siblings" in error for error in prereg.plan_errors(plan))


def test_promotion_ceiling_cannot_express_live_verdicts() -> None:
    plan = make_plan()
    for forbidden in ("LIVE_READY", "PRODUCTION_READY", "PROFITABLE", "DEPLOY"):
        plan["promotion_ceiling"]["maximum_verdict"] = forbidden
        assert any("maximum_verdict" in error for error in prereg.plan_errors(plan))


def test_amendment_creates_child_and_blocks_identity_rewrite() -> None:
    parent = make_plan()
    child = prereg.create_amendment(
        parent, new_experiment_id="00000000-0000-4000-8000-000000000002",
        amendment_reason="tightened concentration limit after review",
        created_at_utc="2026-01-03T00:00:00+00:00",
        changes={"concentration_limits": {
            **parent["concentration_limits"], "max_single_day_fraction": 0.25}})
    assert child["identity"]["parent_experiment_id"] == parent["identity"]["experiment_id"]
    assert child["identity"]["amendment_reason"]
    assert prereg.is_amendment(child)
    assert not prereg.is_amendment(parent)
    # The parent plan was never mutated.
    assert parent == make_plan()
    assert prereg.plan_sha256(child) != prereg.plan_sha256(parent)
    with pytest.raises(PlanValidationError):
        prereg.create_amendment(parent, new_experiment_id="not-a-uuid",
                                amendment_reason="x",
                                created_at_utc="2026-01-03T00:00:00+00:00")
    with pytest.raises(PlanValidationError):
        prereg.create_amendment(
            parent, new_experiment_id="00000000-0000-4000-8000-000000000003",
            amendment_reason="identity rewrite attempt",
            created_at_utc="2026-01-03T00:00:00+00:00",
            changes={"identity": {"experiment_id": "hijack"}})


def test_amendment_with_no_change_fails() -> None:
    parent = make_plan()
    with pytest.raises(PlanValidationError, match="nothing changed"):
        prereg.create_amendment(
            parent, new_experiment_id="00000000-0000-4000-8000-000000000004",
            amendment_reason="no-op", created_at_utc="2026-01-03T00:00:00+00:00")


def test_amendment_reason_required_when_parent_set() -> None:
    plan = make_plan()
    plan["identity"]["parent_experiment_id"] = "00000000-0000-4000-8000-000000000009"
    plan["identity"]["amendment_reason"] = None
    assert any("amendment_reason" in error for error in prereg.plan_errors(plan))


def test_unknown_extra_section_fails() -> None:
    plan = make_plan()
    plan["surprise"] = {"anything": 1}
    assert any("unknown sections" in error for error in prereg.plan_errors(plan))

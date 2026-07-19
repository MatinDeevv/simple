"""Preregistration plan contract.

A plan declares — before any untouched outcome is examined — the hypothesis,
the exact code and data contracts, the comparators that must be beaten, the
negative controls that must fail, the uncertainty method, the robustness
matrix, the multiplicity correction, every acceptance gate, and the maximum
verdict the experiment could ever earn. Every gate must be machine-evaluable:
a numeric threshold, an explicit comparison, and a metric path into the
evidence bundle. Vague criteria ("looks promising", "good enough") are
structurally impossible to register.
"""

from __future__ import annotations

import copy
from typing import Any

from engine.experiments.canonical import (
    is_finite_number,
    is_git_commit_sha,
    is_sha256_hex,
    normalize_logical_path,
    normalize_utc_timestamp,
    normalize_uuid,
    sha256_payload,
)
from engine.experiments.errors import CanonicalJsonError, PathSecurityError, PlanValidationError

PLAN_VERSION = "edge-tribunal-plan-v1"

ENTRY_POLICY_ORDER: tuple[str, ...] = (
    "independent_research_entries",
    "first_entry_per_signal_episode",
    "non_overlapping_global",
    "non_overlapping_component",
    "non_overlapping_basket",
    "one_representative_per_target_cluster",
)

SPLIT_BOUNDARY_POLICIES: tuple[str, ...] = (
    "reset_at_oos_start",
    "carry_chronological_state_into_oos",
)

REQUIRED_SECONDARY_COMPARATOR_KINDS: frozenset[str] = frozenset({
    "global_train_prior",
    "time_shuffled_label_placebo",
    "sign_flipped_control",
    "no_regime_ablation",
    "static_factor_comparator",
    "persistence_comparator",
})

PRIMARY_METRICS: tuple[str, ...] = ("multiclass_brier", "multiclass_log_loss")

GATE_CATEGORIES: frozenset[str] = frozenset({
    "statistical", "negative_control", "robustness",
    "concentration", "power", "data_quality",
})

GATE_COMPARISONS: frozenset[str] = frozenset({">=", ">", "<=", "<", "=="})

GATE_INSUFFICIENCY_POLICIES: frozenset[str] = frozenset({"inconclusive", "invalid", "fail"})

REQUIRED_PRIMARY_GATE_IDS: tuple[str, ...] = (
    "primary_brier_improvement",
    "primary_log_loss_improvement",
    "primary_brier_improvement_lower_bound",
    "primary_log_loss_improvement_lower_bound",
)

REQUIRED_REJECTION_CONDITIONS: frozenset[str] = frozenset({
    "code_mismatch",
    "configuration_mismatch",
    "dataset_mismatch",
    "holdout_reuse",
    "missing_mandatory_comparator",
    "missing_mandatory_metric",
    "invalid_probabilities",
    "non_finite_values",
    "insufficient_target_availability",
    "incomplete_audit_chain",
    "unsealed_plan",
    "evidence_produced_before_dataset_binding",
    "manual_deletion_of_registered_failures",
})

CORRECTION_METHODS: frozenset[str] = frozenset({
    "NONE_SINGLE_PRIMARY", "HOLM_BONFERRONI", "BONFERRONI",
})

PLAN_SECTIONS: tuple[str, ...] = (
    "plan_version", "identity", "hypothesis", "code_contract", "data_contract",
    "target_contract", "primary_comparator", "secondary_comparators",
    "primary_metrics", "secondary_metrics", "uncertainty_contract",
    "entry_policy_contract", "robustness_contract", "concentration_limits",
    "multiple_testing_contract", "acceptance_gates",
    "automatic_rejection_conditions", "promotion_ceiling",
)

# Verdicts a plan may declare as its ceiling. Live/production verdicts are not
# representable: the Tribunal grammar simply has no such value.
PERMITTED_CEILING_VERDICTS: tuple[str, ...] = (
    "RESEARCH_ONLY",
    "FORWARD_TEST_ELIGIBLE",
    "PAPER_FORWARD_TEST_PASSED",
)


def _require_str(errors: list[str], section: dict[str, Any], key: str, where: str,
                 *, allow_none: bool = False) -> None:
    value = section.get(key)
    if value is None and allow_none:
        return
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{where}.{key} must be a non-empty string")


def _require_int(errors: list[str], section: dict[str, Any], key: str, where: str,
                 *, minimum: int | None = None) -> None:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{where}.{key} must be an integer")
        return
    if minimum is not None and value < minimum:
        errors.append(f"{where}.{key} must be >= {minimum}")


def _require_fraction(errors: list[str], section: dict[str, Any], key: str, where: str) -> None:
    value = section.get(key)
    if not is_finite_number(value) or not (0.0 < float(value) <= 1.0):
        errors.append(f"{where}.{key} must be a finite number in (0, 1]")


def _require_utc(errors: list[str], section: dict[str, Any], key: str, where: str) -> str | None:
    value = section.get(key)
    try:
        return normalize_utc_timestamp(value)
    except CanonicalJsonError as exc:
        errors.append(f"{where}.{key}: {exc}")
        return None


def _require_str_list(errors: list[str], section: dict[str, Any], key: str, where: str,
                      *, allow_empty: bool = False) -> list[str]:
    value = section.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        errors.append(f"{where}.{key} must be a list of non-empty strings")
        return []
    if not value and not allow_empty:
        errors.append(f"{where}.{key} must not be empty")
    return value


def _validate_identity(errors: list[str], identity: Any) -> None:
    where = "identity"
    if not isinstance(identity, dict):
        errors.append(f"{where} must be an object")
        return
    try:
        normalize_uuid(identity.get("experiment_id"))
    except CanonicalJsonError as exc:
        errors.append(f"{where}.experiment_id: {exc}")
    _require_str(errors, identity, "title", where)
    _require_str(errors, identity, "researcher", where)
    _require_utc(errors, identity, "created_at_utc", where)
    parent = identity.get("parent_experiment_id")
    if parent is not None:
        try:
            normalize_uuid(parent)
        except CanonicalJsonError as exc:
            errors.append(f"{where}.parent_experiment_id: {exc}")
        reason = identity.get("amendment_reason")
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"{where}.amendment_reason is required when parent_experiment_id is set")
    elif identity.get("amendment_reason") is not None:
        errors.append(f"{where}.amendment_reason must be null when there is no parent experiment")


def _validate_hypothesis(errors: list[str], hypothesis: Any) -> None:
    where = "hypothesis"
    if not isinstance(hypothesis, dict):
        errors.append(f"{where} must be an object")
        return
    for key in ("statement", "economic_mechanism", "expected_direction",
                "expected_holding_horizon", "expected_failure_mode",
                "why_effect_exists", "why_effect_might_disappear"):
        _require_str(errors, hypothesis, key, where)


def _validate_code_contract(errors: list[str], contract: Any) -> None:
    where = "code_contract"
    if not isinstance(contract, dict):
        errors.append(f"{where} must be an object")
        return
    _require_str(errors, contract, "repository", where)
    if not is_git_commit_sha(contract.get("commit_sha")):
        errors.append(f"{where}.commit_sha must be a 40-char lowercase hex git SHA")
    tree = contract.get("git_tree_sha")
    if tree is not None and not is_git_commit_sha(tree):
        errors.append(f"{where}.git_tree_sha must be null or a 40-char lowercase hex SHA")
    _require_str(errors, contract, "package_version", where)
    _require_str(errors, contract, "model_contract_version", where)
    if not is_sha256_hex(contract.get("configuration_sha256")):
        errors.append(f"{where}.configuration_sha256 must be a 64-char lowercase hex SHA-256")
    sources = contract.get("source_file_sha256")
    if not isinstance(sources, dict) or not sources:
        errors.append(f"{where}.source_file_sha256 must be a non-empty object")
    else:
        for path, digest in sources.items():
            try:
                normalize_logical_path(path)
            except PathSecurityError as exc:
                errors.append(f"{where}.source_file_sha256: {exc}")
            if not is_sha256_hex(digest):
                errors.append(f"{where}.source_file_sha256[{path!r}] must be a SHA-256 hex digest")
    _require_str(errors, contract, "evidence_producer_command", where)
    _require_str_list(errors, contract, "allowed_implementation_files", where)
    allowed = contract.get("allowed_implementation_files")
    if isinstance(sources, dict) and isinstance(allowed, list):
        outside = sorted(set(sources) - set(allowed))
        if outside:
            errors.append(f"{where}.source_file_sha256 contains files outside "
                          f"allowed_implementation_files: {outside}")
    commands = contract.get("required_test_commands")
    if (not isinstance(commands, list) or not commands
            or not all(isinstance(command, list) and command
                       and all(isinstance(arg, str) and arg for arg in command)
                       for command in commands)):
        errors.append(f"{where}.required_test_commands must be non-empty canonical argv lists")


def _validate_data_contract(errors: list[str], contract: Any) -> None:
    where = "data_contract"
    if not isinstance(contract, dict):
        errors.append(f"{where} must be an object")
        return
    _require_str_list(errors, contract, "universe", where)
    _require_str_list(errors, contract, "pair_order", where)
    _require_str(errors, contract, "frequency", where)
    _require_str(errors, contract, "price_side", where)
    _require_str_list(errors, contract, "required_columns", where)
    if contract.get("timestamp_timezone") != "UTC":
        errors.append(f"{where}.timestamp_timezone must be exactly 'UTC'")
    _require_int(errors, contract, "expected_observation_interval_seconds", where, minimum=1)
    minimum_start = _require_utc(errors, contract, "minimum_start_utc", where)
    inspection_cutoff = _require_utc(errors, contract, "maximum_permitted_inspection_utc", where)
    untouched_start = _require_utc(errors, contract, "untouched_evaluation_start_utc", where)
    untouched_end = _require_utc(errors, contract, "untouched_evaluation_end_utc", where)
    if untouched_start and untouched_end and untouched_start >= untouched_end:
        errors.append(f"{where}: untouched_evaluation_start_utc must precede untouched_evaluation_end_utc")
    if inspection_cutoff and untouched_start and untouched_start < inspection_cutoff:
        errors.append(f"{where}: untouched evaluation window begins before the inspection cutoff")
    if minimum_start and untouched_start and untouched_start < minimum_start:
        errors.append(f"{where}: untouched evaluation window begins before minimum_start_utc")
    for key in ("warmup_policy", "missing_bar_policy", "duplicate_row_policy", "gap_reset_policy"):
        _require_str(errors, contract, key, where)
    _require_str_list(errors, contract, "dataset_provenance_requirements", where)


def _validate_target_contract(errors: list[str], contract: Any) -> None:
    where = "target_contract"
    if not isinstance(contract, dict):
        errors.append(f"{where} must be an object")
        return
    _require_str(errors, contract, "target_name", where)
    class_order = contract.get("class_order")
    if not isinstance(class_order, list) or len(class_order) != 3 or \
            not all(isinstance(item, str) and item for item in class_order) or \
            len(set(class_order)) != 3:
        errors.append(f"{where}.class_order must be a list of exactly 3 distinct class names")
    _require_int(errors, contract, "target_horizon_steps", where, minimum=1)
    _require_str(errors, contract, "neutral_zone_definition", where)
    _require_str(errors, contract, "target_availability_rules", where)
    _require_str_list(errors, contract, "leakage_prohibitions", where)


def _validate_primary_comparator(errors: list[str], comparator: Any) -> None:
    where = "primary_comparator"
    if not isinstance(comparator, dict):
        errors.append(f"{where} must be an object")
        return
    _require_str(errors, comparator, "identity", where)
    _require_str_list(errors, comparator, "fallback_hierarchy", where)
    alpha = comparator.get("smoothing_alpha")
    if not is_finite_number(alpha) or float(alpha) < 0:
        errors.append(f"{where}.smoothing_alpha must be a finite number >= 0")
    _require_int(errors, comparator, "minimum_cell_count", where, minimum=1)
    if comparator.get("train_only") is not True:
        errors.append(f"{where}.train_only must be exactly true")


def _validate_secondary_comparators(errors: list[str], comparators: Any) -> None:
    where = "secondary_comparators"
    if not isinstance(comparators, list) or not comparators:
        errors.append(f"{where} must be a non-empty list")
        return
    kinds: set[str] = set()
    for index, comparator in enumerate(comparators):
        if not isinstance(comparator, dict):
            errors.append(f"{where}[{index}] must be an object")
            continue
        _require_str(errors, comparator, "comparator_id", f"{where}[{index}]")
        kind = comparator.get("kind")
        if not isinstance(kind, str) or not kind:
            errors.append(f"{where}[{index}].kind must be a non-empty string")
        else:
            kinds.add(kind)
    missing = sorted(REQUIRED_SECONDARY_COMPARATOR_KINDS - kinds)
    if missing:
        errors.append(f"{where} is missing mandatory comparator kinds: {missing}")


def _validate_uncertainty(errors: list[str], contract: Any) -> None:
    where = "uncertainty_contract"
    if not isinstance(contract, dict):
        errors.append(f"{where} must be an object")
        return
    _require_str(errors, contract, "bootstrap_method", where)
    _require_str(errors, contract, "block_units", where)
    blocks = contract.get("block_sizes")
    if not isinstance(blocks, list) or not blocks or \
            not all(isinstance(item, int) and not isinstance(item, bool) and item >= 1 for item in blocks):
        errors.append(f"{where}.block_sizes must be a non-empty list of integers >= 1")
    _require_int(errors, contract, "replicate_count", where, minimum=1)
    _require_int(errors, contract, "random_seed", where, minimum=0)
    for key in ("one_sided_confidence", "two_sided_confidence"):
        value = contract.get(key)
        if not is_finite_number(value) or not (0.5 <= float(value) < 1.0):
            errors.append(f"{where}.{key} must be a finite number in [0.5, 1)")
    _require_int(errors, contract, "minimum_independent_target_clusters", where, minimum=1)
    _require_int(errors, contract, "minimum_causal_segments", where, minimum=1)
    _require_int(errors, contract, "minimum_oos_rows", where, minimum=1)


def _validate_entry_policy(errors: list[str], contract: Any) -> None:
    where = "entry_policy_contract"
    if not isinstance(contract, dict):
        errors.append(f"{where} must be an object")
        return
    if contract.get("primary_population") != ENTRY_POLICY_ORDER[0]:
        errors.append(f"{where}.primary_population must be {ENTRY_POLICY_ORDER[0]!r}")
    sensitivities = contract.get("sensitivity_populations")
    if sensitivities != list(ENTRY_POLICY_ORDER):
        errors.append(f"{where}.sensitivity_populations must be exactly {list(ENTRY_POLICY_ORDER)}")
    boundaries = contract.get("split_boundary_policies")
    if boundaries != list(SPLIT_BOUNDARY_POLICIES):
        errors.append(f"{where}.split_boundary_policies must be exactly {list(SPLIT_BOUNDARY_POLICIES)}")
    _require_str(errors, contract, "target_cluster_representative_rule", where)


def _validate_robustness(errors: list[str], contract: Any) -> None:
    where = "robustness_contract"
    if not isinstance(contract, dict):
        errors.append(f"{where} must be an object")
        return
    for key in ("required_time_slices", "required_regime_slices", "required_policy_slices",
                "required_split_boundary_policies", "required_perturbations"):
        _require_str_list(errors, contract, key, where)
    blocks = contract.get("required_block_lengths")
    if not isinstance(blocks, list) or not blocks or \
            not all(isinstance(item, int) and not isinstance(item, bool) and item >= 1 for item in blocks):
        errors.append(f"{where}.required_block_lengths must be a non-empty list of integers >= 1")
    _require_str_list(errors, contract, "mandatory_cells", where)
    _require_fraction(errors, contract, "minimum_pass_proportion", where)
    _require_str_list(errors, contract, "insufficient_allowed_cells", where, allow_empty=True)
    _require_int(errors, contract, "minimum_samples_per_cell", where, minimum=1)
    if contract.get("missing_cell_policy") not in ("count_as_failed", "count_in_denominator"):
        errors.append(f"{where}.missing_cell_policy must be 'count_as_failed' or 'count_in_denominator'")
    value = contract.get("max_missing_data_sensitivity")
    if not is_finite_number(value) or float(value) < 0:
        errors.append(f"{where}.max_missing_data_sensitivity must be a finite number >= 0")


def _validate_concentration(errors: list[str], limits: Any) -> None:
    where = "concentration_limits"
    if not isinstance(limits, dict):
        errors.append(f"{where} must be an object")
        return
    for key in ("max_single_signal_episode_fraction", "max_single_target_cluster_fraction",
                "max_single_causal_segment_fraction", "max_single_day_fraction",
                "max_single_component_fraction", "max_single_regime_fraction",
                "max_single_fallback_tier_fraction"):
        _require_fraction(errors, limits, key, where)
    _require_int(errors, limits, "small_sample_rows_threshold", where, minimum=1)


def _validate_multiplicity(errors: list[str], contract: Any) -> None:
    where = "multiple_testing_contract"
    if not isinstance(contract, dict):
        errors.append(f"{where} must be an object")
        return
    _require_str(errors, contract, "family_id", where)
    _require_int(errors, contract, "family_size", where, minimum=1)
    method = contract.get("correction_method")
    if method not in CORRECTION_METHODS:
        errors.append(f"{where}.correction_method must be one of {sorted(CORRECTION_METHODS)}")
    alpha = contract.get("familywise_alpha")
    if not is_finite_number(alpha) or not (0.0 < float(alpha) < 1.0):
        errors.append(f"{where}.familywise_alpha must be a finite number in (0, 1)")
    _require_int(errors, contract, "primary_hypothesis_index", where, minimum=0)
    family_size = contract.get("family_size")
    primary_index = contract.get("primary_hypothesis_index")
    if isinstance(family_size, int) and isinstance(primary_index, int) and \
            not isinstance(family_size, bool) and not isinstance(primary_index, bool) and \
            primary_index >= family_size >= 1:
        errors.append(f"{where}.primary_hypothesis_index must be < family_size")
    if method == "NONE_SINGLE_PRIMARY" and family_size != 1:
        errors.append(f"{where}: NONE_SINGLE_PRIMARY requires family_size == 1")
    if not isinstance(contract.get("single_registered_primary"), bool):
        errors.append(f"{where}.single_registered_primary must be a boolean")
    if contract.get("secondary_metric_treatment") != "exploratory_only":
        errors.append(f"{where}.secondary_metric_treatment must be 'exploratory_only'")
    siblings = contract.get("registered_sibling_experiment_ids")
    if not isinstance(siblings, list) or not all(isinstance(item, str) for item in siblings):
        errors.append(f"{where}.registered_sibling_experiment_ids must be a list of strings")
    elif isinstance(family_size, int) and not isinstance(family_size, bool) and \
            len(siblings) + 1 != family_size:
        errors.append(
            f"{where}: family_size ({family_size}) must equal registered siblings + 1 "
            f"({len(siblings) + 1})")


def _validate_gates(errors: list[str], gates: Any) -> None:
    where = "acceptance_gates"
    if not isinstance(gates, list) or not gates:
        errors.append(f"{where} must be a non-empty list")
        return
    seen_ids: set[str] = set()
    categories: set[str] = set()
    for index, gate in enumerate(gates):
        prefix = f"{where}[{index}]"
        if not isinstance(gate, dict):
            errors.append(f"{prefix} must be an object")
            continue
        gate_id = gate.get("gate_id")
        if not isinstance(gate_id, str) or not gate_id:
            errors.append(f"{prefix}.gate_id must be a non-empty string")
        elif gate_id in seen_ids:
            errors.append(f"{prefix}.gate_id duplicates {gate_id!r}")
        else:
            seen_ids.add(gate_id)
        category = gate.get("category")
        if category not in GATE_CATEGORIES:
            errors.append(f"{prefix}.category must be one of {sorted(GATE_CATEGORIES)}")
        else:
            categories.add(category)
        _require_str(errors, gate, "description", prefix)
        metric_path = gate.get("metric_path")
        if not isinstance(metric_path, str) or not metric_path or " " in metric_path:
            errors.append(f"{prefix}.metric_path must be a non-empty dot path with no spaces")
        if gate.get("comparison") not in GATE_COMPARISONS:
            errors.append(f"{prefix}.comparison must be one of {sorted(GATE_COMPARISONS)}")
        threshold = gate.get("threshold")
        if not is_finite_number(threshold):
            errors.append(f"{prefix}.threshold must be a finite number (vague gates are not "
                          f"machine-evaluable)")
        if not isinstance(gate.get("required"), bool):
            errors.append(f"{prefix}.required must be a boolean")
        if gate.get("on_insufficient") not in GATE_INSUFFICIENCY_POLICIES:
            errors.append(f"{prefix}.on_insufficient must be one of "
                          f"{sorted(GATE_INSUFFICIENCY_POLICIES)}")
    for required_id in REQUIRED_PRIMARY_GATE_IDS:
        if required_id not in seen_ids:
            errors.append(f"{where} must include mandatory gate {required_id!r}")
    for required_category in ("negative_control", "power", "concentration", "data_quality"):
        if required_category not in categories:
            errors.append(f"{where} must include at least one {required_category!r} gate")


def _validate_rejection_conditions(errors: list[str], conditions: Any) -> None:
    where = "automatic_rejection_conditions"
    if not isinstance(conditions, list) or not all(isinstance(item, str) for item in conditions):
        errors.append(f"{where} must be a list of strings")
        return
    missing = sorted(REQUIRED_REJECTION_CONDITIONS - set(conditions))
    if missing:
        errors.append(f"{where} is missing mandatory conditions: {missing}")


def _validate_promotion_ceiling(errors: list[str], ceiling: Any) -> None:
    where = "promotion_ceiling"
    if not isinstance(ceiling, dict):
        errors.append(f"{where} must be an object")
        return
    if ceiling.get("maximum_verdict") not in PERMITTED_CEILING_VERDICTS:
        errors.append(f"{where}.maximum_verdict must be one of {list(PERMITTED_CEILING_VERDICTS)}")
    _require_str(errors, ceiling, "explanation", where)
    if ceiling.get("execution_data_required") is not True:
        errors.append(f"{where}.execution_data_required must be exactly true")
    if ceiling.get("forward_test_required") is not True:
        errors.append(f"{where}.forward_test_required must be exactly true")


def plan_errors(plan: Any) -> list[str]:
    """Return every validation error for ``plan`` (empty list means valid)."""
    errors: list[str] = []
    if not isinstance(plan, dict):
        return ["plan must be a JSON object"]
    if plan.get("plan_version") != PLAN_VERSION:
        errors.append(f"plan_version must be {PLAN_VERSION!r}")
    for section in PLAN_SECTIONS:
        if section not in plan:
            errors.append(f"plan is missing required section {section!r}")
    extra = sorted(set(plan) - set(PLAN_SECTIONS))
    if extra:
        errors.append(f"plan contains unknown sections: {extra}")
    if errors:
        return errors
    _validate_identity(errors, plan["identity"])
    _validate_hypothesis(errors, plan["hypothesis"])
    _validate_code_contract(errors, plan["code_contract"])
    _validate_data_contract(errors, plan["data_contract"])
    _validate_target_contract(errors, plan["target_contract"])
    _validate_primary_comparator(errors, plan["primary_comparator"])
    _validate_secondary_comparators(errors, plan["secondary_comparators"])
    if plan.get("primary_metrics") != list(PRIMARY_METRICS):
        errors.append(f"primary_metrics must be exactly {list(PRIMARY_METRICS)}")
    _require_str_list(errors, plan, "secondary_metrics", "plan")
    _validate_uncertainty(errors, plan["uncertainty_contract"])
    _validate_entry_policy(errors, plan["entry_policy_contract"])
    _validate_robustness(errors, plan["robustness_contract"])
    _validate_concentration(errors, plan["concentration_limits"])
    _validate_multiplicity(errors, plan["multiple_testing_contract"])
    _validate_gates(errors, plan["acceptance_gates"])
    _validate_rejection_conditions(errors, plan["automatic_rejection_conditions"])
    _validate_promotion_ceiling(errors, plan["promotion_ceiling"])
    return errors


def validate_plan(plan: Any) -> None:
    errors = plan_errors(plan)
    if errors:
        raise PlanValidationError("invalid preregistration plan:\n- " + "\n- ".join(errors))


def plan_sha256(plan: dict[str, Any]) -> str:
    """Canonical hash of a validated plan. Key order never matters."""
    return sha256_payload(plan)


def is_amendment(plan: dict[str, Any]) -> bool:
    return plan.get("identity", {}).get("parent_experiment_id") is not None


def create_amendment(
    parent_plan: dict[str, Any],
    *,
    new_experiment_id: str,
    amendment_reason: str,
    created_at_utc: str,
    changes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a child plan from a sealed parent. The parent is never mutated.

    The child records its parent's experiment ID and the amendment reason;
    the verdict engine automatically blocks clean promotion for any amended
    experiment because its plan was, by definition, written after something
    about the original was reconsidered.
    """
    validate_plan(parent_plan)
    if not isinstance(amendment_reason, str) or not amendment_reason.strip():
        raise PlanValidationError("amendment_reason must be a non-empty string")
    child = copy.deepcopy(parent_plan)
    for section, replacement in (changes or {}).items():
        if section == "identity":
            raise PlanValidationError("amendment changes may not rewrite identity directly")
        if section not in PLAN_SECTIONS:
            raise PlanValidationError(f"amendment changes reference unknown section {section!r}")
        child[section] = copy.deepcopy(replacement)
    child["identity"] = dict(child["identity"])
    try:
        child["identity"]["experiment_id"] = normalize_uuid(new_experiment_id)
        child["identity"]["parent_experiment_id"] = normalize_uuid(
            parent_plan["identity"]["experiment_id"])
        child["identity"]["created_at_utc"] = normalize_utc_timestamp(created_at_utc)
    except CanonicalJsonError as exc:
        raise PlanValidationError(str(exc)) from exc
    child["identity"]["amendment_reason"] = amendment_reason
    validate_plan(child)
    parent_body = {key: value for key, value in parent_plan.items() if key != "identity"}
    child_body = {key: value for key, value in child.items() if key != "identity"}
    if sha256_payload(child_body) == sha256_payload(parent_body):
        raise PlanValidationError("amendment produced an identical plan body; nothing changed")
    return child

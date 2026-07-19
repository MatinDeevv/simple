"""Evidence-bundle contract and ingestion.

The Tribunal consumes one structured JSON evidence bundle per experiment. It
never scrapes console output, never executes anything an evidence file names
(``execution_command`` is an inert provenance string), and never treats a
missing result as a passed result: every mandatory comparator, control,
policy view, and block length must be physically present or ingestion fails.
"""

from __future__ import annotations

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
from engine.experiments.errors import (
    CanonicalJsonError,
    EvidenceValidationError,
    GateEvaluationError,
    PathSecurityError,
)
from engine.experiments.preregistration import (
    ENTRY_POLICY_ORDER,
    REQUIRED_SECONDARY_COMPARATOR_KINDS,
    SPLIT_BOUNDARY_POLICIES,
)

EVIDENCE_VERSION = "edge-tribunal-evidence-v1"

ROBUSTNESS_CELL_STATUSES: frozenset[str] = frozenset({
    "scored",
    "insufficient_train_population",
    "insufficient_oos_population",
    "insufficient_class_support",
    "insufficient_target_clusters",
    "insufficient_bootstrap_support",
    "missing",
})

_TOP_LEVEL_SECTIONS: tuple[str, ...] = (
    "evidence_version", "experiment", "producer", "dataset", "population",
    "primary_model", "primary_comparator", "negative_controls", "robustness",
    "concentration", "data_quality", "execution_data", "multiplicity",
    "artifacts",
)

_CONCENTRATION_FIELDS: tuple[str, ...] = (
    "largest_signal_episode_fraction", "largest_target_cluster_fraction",
    "largest_causal_segment_fraction", "largest_day_fraction",
    "largest_component_fraction", "largest_regime_fraction",
    "largest_fallback_tier_fraction",
)

_EXECUTION_FIELDS: tuple[str, ...] = (
    "ask_available", "spread_available", "fill_available", "commission_available",
    "latency_available", "impact_available", "contract_notional_available",
    "conversion_price_available",
)


def _nonneg_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite(errors: list[str], section: dict[str, Any], key: str, where: str) -> None:
    if not is_finite_number(section.get(key)):
        errors.append(f"{where}.{key} must be a finite number")


def _fraction(errors: list[str], section: dict[str, Any], key: str, where: str) -> None:
    value = section.get(key)
    if not is_finite_number(value) or not (0.0 <= float(value) <= 1.0):
        errors.append(f"{where}.{key} must be a finite number in [0, 1]")


def _check_experiment(errors: list[str], evidence: dict[str, Any], seal: dict[str, Any],
                      binding: dict[str, Any]) -> None:
    section = evidence.get("experiment")
    where = "experiment"
    if not isinstance(section, dict):
        errors.append(f"{where} must be an object")
        return
    try:
        claimed_id = normalize_uuid(section.get("experiment_id"))
    except CanonicalJsonError as exc:
        errors.append(f"{where}.experiment_id: {exc}")
        claimed_id = None
    if claimed_id is not None and claimed_id != binding["experiment_id"]:
        errors.append(f"{where}.experiment_id does not match the bound experiment")
    if section.get("plan_sha256") != seal["plan_sha256"]:
        errors.append(f"{where}.plan_sha256 does not match the sealed plan")
    if section.get("seal_sha256") != seal["seal_sha256"]:
        errors.append(f"{where}.seal_sha256 does not match the seal")
    if section.get("binding_sha256") != binding["binding_sha256"]:
        errors.append(f"{where}.binding_sha256 does not match the dataset binding")


def _check_producer(errors: list[str], evidence: dict[str, Any], plan: dict[str, Any]) -> None:
    section = evidence.get("producer")
    where = "producer"
    if not isinstance(section, dict):
        errors.append(f"{where} must be an object")
        return
    contract = plan["code_contract"]
    commit = section.get("code_commit_sha")
    if not is_git_commit_sha(commit):
        errors.append(f"{where}.code_commit_sha must be a 40-char hex git SHA")
    elif commit != contract["commit_sha"]:
        errors.append(f"{where}.code_commit_sha does not match the preregistered commit")
    if section.get("model_contract_version") != contract["model_contract_version"]:
        errors.append(f"{where}.model_contract_version does not match the plan")
    if section.get("configuration_sha256") != contract["configuration_sha256"]:
        errors.append(f"{where}.configuration_sha256 does not match the preregistered configuration")
    sources = section.get("source_file_sha256")
    if not isinstance(sources, dict) or sources != contract["source_file_sha256"]:
        errors.append(f"{where}.source_file_sha256 does not match the preregistered source hashes")
    command = section.get("execution_command")
    if not isinstance(command, str) or not command.strip():
        errors.append(f"{where}.execution_command must be a non-empty string")
    elif "\n" in command or "\x00" in command:
        errors.append(f"{where}.execution_command must be a single line (it is provenance text; "
                      f"the Tribunal never executes it)")
    if not isinstance(section.get("python_version"), str) or not section.get("python_version"):
        errors.append(f"{where}.python_version must be a non-empty string")
    if not is_sha256_hex(section.get("dependency_snapshot_sha256")):
        errors.append(f"{where}.dependency_snapshot_sha256 must be a SHA-256 hex digest")
    seeds = section.get("random_seeds")
    if not isinstance(seeds, dict) or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in seeds.values()):
        errors.append(f"{where}.random_seeds must be an object of integer seeds")
    try:
        normalize_utc_timestamp(section.get("produced_at_utc"))
    except CanonicalJsonError as exc:
        errors.append(f"{where}.produced_at_utc: {exc}")


def _check_dataset(errors: list[str], evidence: dict[str, Any], binding: dict[str, Any]) -> None:
    section = evidence.get("dataset")
    where = "dataset"
    if not isinstance(section, dict):
        errors.append(f"{where} must be an object")
        return
    if section.get("dataset_binding_sha256") != binding["binding_sha256"]:
        errors.append(f"{where}.dataset_binding_sha256 does not match the dataset binding")
    file_hashes = section.get("dataset_file_sha256")
    bound_hashes = {entry["logical_path"]: entry["sha256"] for entry in binding["files"]}
    if not isinstance(file_hashes, dict) or file_hashes != bound_hashes:
        errors.append(f"{where}.dataset_file_sha256 does not match the bound dataset files")
    interval = section.get("holdout_interval")
    if not isinstance(interval, dict) or \
            interval.get("start_utc") != binding["holdout_interval"]["start_utc"] or \
            interval.get("end_utc") != binding["holdout_interval"]["end_utc"]:
        errors.append(f"{where}.holdout_interval does not match the bound holdout interval")
    if not _nonneg_int(section.get("row_count")):
        errors.append(f"{where}.row_count must be a non-negative integer")
    if not _nonneg_int(section.get("valid_target_count")):
        errors.append(f"{where}.valid_target_count must be a non-negative integer")


def _check_population(errors: list[str], evidence: dict[str, Any], plan: dict[str, Any]) -> None:
    section = evidence.get("population")
    where = "population"
    if not isinstance(section, dict):
        errors.append(f"{where} must be an object")
        return
    for key in ("training_rows", "oos_rows", "accepted_rows", "signal_episodes",
                "target_clusters", "causal_segments"):
        if not _nonneg_int(section.get(key)):
            errors.append(f"{where}.{key} must be a non-negative integer")
    accepted = section.get("accepted_rows")
    oos = section.get("oos_rows")
    clusters = section.get("target_clusters")
    if _nonneg_int(accepted) and _nonneg_int(oos) and accepted > oos:
        errors.append(f"{where}: accepted_rows ({accepted}) exceeds oos_rows ({oos})")
    if _nonneg_int(accepted) and _nonneg_int(clusters) and clusters > accepted:
        errors.append(f"{where}: target_clusters ({clusters}) exceeds accepted_rows ({accepted})")
    class_counts = section.get("class_counts")
    expected_classes = plan["target_contract"]["class_order"]
    if not isinstance(class_counts, dict) or sorted(class_counts) != sorted(expected_classes):
        errors.append(f"{where}.class_counts must have exactly the classes {expected_classes}")
    elif not all(_nonneg_int(value) for value in class_counts.values()):
        errors.append(f"{where}.class_counts values must be non-negative integers")
    elif _nonneg_int(accepted) and sum(class_counts.values()) != accepted:
        errors.append(f"{where}.class_counts sum to {sum(class_counts.values())}, "
                      f"expected accepted_rows ({accepted})")
    policy_counts = section.get("policy_counts")
    if not isinstance(policy_counts, dict):
        errors.append(f"{where}.policy_counts must be an object")
    else:
        missing = [policy for policy in ENTRY_POLICY_ORDER if policy not in policy_counts]
        if missing:
            errors.append(f"{where}.policy_counts is missing policies: {missing}")


def _check_metric_block(errors: list[str], section: Any, where: str,
                        *, require_bounds: bool) -> None:
    if not isinstance(section, dict):
        errors.append(f"{where} must be an object")
        return
    for key in ("multiclass_brier", "multiclass_log_loss"):
        _finite(errors, section, key, where)
    if require_bounds:
        for metric in ("brier_improvement", "log_loss_improvement"):
            _finite(errors, section, metric, where)
            lower = section.get(f"{metric}_lower_bound_one_sided")
            two_lower = section.get(f"{metric}_lower_bound_two_sided")
            two_upper = section.get(f"{metric}_upper_bound_two_sided")
            for bound_key in (f"{metric}_lower_bound_one_sided",
                              f"{metric}_lower_bound_two_sided",
                              f"{metric}_upper_bound_two_sided"):
                _finite(errors, section, bound_key, where)
            if is_finite_number(two_lower) and is_finite_number(two_upper) and \
                    float(two_lower) > float(two_upper):
                errors.append(f"{where}.{metric}: two-sided lower bound exceeds upper bound")
            if is_finite_number(lower) and is_finite_number(section.get(metric)) and \
                    float(lower) > float(section[metric]):
                errors.append(f"{where}.{metric}: one-sided lower bound exceeds the point estimate")


def _check_primary_model(errors: list[str], evidence: dict[str, Any]) -> None:
    section = evidence.get("primary_model")
    where = "primary_model"
    _check_metric_block(errors, section, where, require_bounds=True)
    if not isinstance(section, dict):
        return
    _finite(errors, section, "calibration_error", where)
    summary = section.get("probability_summary")
    if not isinstance(summary, dict):
        errors.append(f"{where}.probability_summary must be an object")
        return
    if not _nonneg_int(summary.get("rows")):
        errors.append(f"{where}.probability_summary.rows must be a non-negative integer")
    _fraction(errors, summary, "min_probability", f"{where}.probability_summary")
    _fraction(errors, summary, "max_probability", f"{where}.probability_summary")
    mean_row_sum = summary.get("mean_row_sum")
    if not is_finite_number(mean_row_sum) or abs(float(mean_row_sum) - 1.0) > 1e-6:
        errors.append(f"{where}.probability_summary.mean_row_sum must equal 1 within 1e-6")


def _check_primary_comparator(errors: list[str], evidence: dict[str, Any],
                              plan: dict[str, Any]) -> None:
    section = evidence.get("primary_comparator")
    where = "primary_comparator"
    if not isinstance(section, dict):
        errors.append(f"{where} must be an object")
        return
    if section.get("identity") != plan["primary_comparator"]["identity"]:
        errors.append(f"{where}.identity does not match the preregistered primary comparator")
    if section.get("train_only_confirmed") is not True:
        errors.append(f"{where}.train_only_confirmed must be exactly true")
    tiers = section.get("fallback_tier_counts")
    if not isinstance(tiers, dict) or not all(_nonneg_int(value) for value in tiers.values()):
        errors.append(f"{where}.fallback_tier_counts must be an object of non-negative integers")
    _check_metric_block(errors, section, where, require_bounds=False)


def _check_negative_controls(errors: list[str], evidence: dict[str, Any]) -> None:
    section = evidence.get("negative_controls")
    where = "negative_controls"
    if not isinstance(section, dict):
        errors.append(f"{where} must be an object")
        return
    missing = sorted(REQUIRED_SECONDARY_COMPARATOR_KINDS - set(section))
    if missing:
        errors.append(f"{where} is missing mandatory controls (a failed control may never be "
                      f"omitted): {missing}")
    for name, control in section.items():
        prefix = f"{where}.{name}"
        if not isinstance(control, dict):
            errors.append(f"{prefix} must be an object")
            continue
        for key in ("brier_improvement", "log_loss_improvement"):
            _finite(errors, control, key, prefix)
        if control.get("status") not in ("behaved_as_required", "violated_expectation"):
            errors.append(f"{prefix}.status must be 'behaved_as_required' or "
                          f"'violated_expectation' (not reported never means passed)")


def _check_robustness(errors: list[str], evidence: dict[str, Any], plan: dict[str, Any]) -> None:
    section = evidence.get("robustness")
    where = "robustness"
    if not isinstance(section, dict):
        errors.append(f"{where} must be an object")
        return
    _finite(errors, section, "missing_data_sensitivity", where)
    cells = section.get("cells")
    if not isinstance(cells, list) or not cells:
        errors.append(f"{where}.cells must be a non-empty list")
        return
    seen_ids: set[str] = set()
    policies_seen: set[str] = set()
    boundaries_seen: set[str] = set()
    blocks_seen: set[int] = set()
    times_seen: set[str] = set()
    regimes_seen: set[str] = set()
    perturbations_seen: set[str] = set()
    for index, cell in enumerate(cells):
        prefix = f"{where}.cells[{index}]"
        if not isinstance(cell, dict):
            errors.append(f"{prefix} must be an object")
            continue
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            errors.append(f"{prefix}.cell_id must be a non-empty string")
        elif cell_id in seen_ids:
            errors.append(f"{prefix}.cell_id duplicates {cell_id!r}")
        else:
            seen_ids.add(cell_id)
        dimensions = cell.get("dimensions")
        if not isinstance(dimensions, dict):
            errors.append(f"{prefix}.dimensions must be an object")
            dimensions = {}
        if dimensions:
            from engine.experiments.robustness import cell_id_for_dimensions
            try:
                expected_cell_id = cell_id_for_dimensions(dimensions)
                if cell_id != expected_cell_id:
                    errors.append(f"{prefix}.cell_id does not match its canonical dimensions")
            except GateEvaluationError as exc:
                errors.append(f"{prefix}.dimensions: {exc}")
        policy = dimensions.get("entry_policy")
        if policy in ENTRY_POLICY_ORDER:
            policies_seen.add(policy)
        boundary = dimensions.get("split_boundary")
        if boundary in SPLIT_BOUNDARY_POLICIES:
            boundaries_seen.add(boundary)
        block = dimensions.get("block_length")
        if isinstance(block, int) and not isinstance(block, bool):
            blocks_seen.add(block)
        for key, target in (("time_slice", times_seen), ("regime", regimes_seen),
                            ("perturbation", perturbations_seen)):
            value = dimensions.get(key)
            if isinstance(value, str) and value:
                target.add(value)
        status = cell.get("status")
        if status not in ROBUSTNESS_CELL_STATUSES:
            errors.append(f"{prefix}.status must be one of {sorted(ROBUSTNESS_CELL_STATUSES)}")
        if status == "scored":
            for key in ("brier_improvement", "log_loss_improvement"):
                _finite(errors, cell, key, prefix)
            if not _nonneg_int(cell.get("sample_count")):
                errors.append(f"{prefix}.sample_count must be a non-negative integer")
    missing_policies = [policy for policy in ENTRY_POLICY_ORDER if policy not in policies_seen]
    if missing_policies:
        errors.append(f"{where}: mandatory entry policies absent from cells: {missing_policies}")
    missing_boundaries = [b for b in SPLIT_BOUNDARY_POLICIES if b not in boundaries_seen]
    if missing_boundaries:
        errors.append(f"{where}: mandatory split boundaries absent from cells: {missing_boundaries}")
    required_blocks = plan["robustness_contract"]["required_block_lengths"]
    missing_blocks = [block for block in required_blocks if block not in blocks_seen]
    if missing_blocks:
        errors.append(f"{where}: mandatory block lengths absent from cells: {missing_blocks}")
    # V2 plans declare the full cell registry.  Legacy v1 plans retain their
    # historical loose coverage semantics but cannot claim v2 verification.
    if plan["robustness_contract"].get("required_split_boundary_policies"):
        for key, seen in (("required_time_slices", times_seen),
                          ("required_regime_slices", regimes_seen),
                          ("required_perturbations", perturbations_seen)):
            missing = [value for value in plan["robustness_contract"][key] if value not in seen]
            if missing:
                errors.append(f"{where}: {key} absent from cells: {missing}")
        from engine.experiments.robustness import planned_cell_registry
        planned = set(planned_cell_registry(plan["robustness_contract"]))
        if seen_ids != planned:
            missing = sorted(planned - seen_ids)
            unexpected = sorted(seen_ids - planned)
            if missing:
                errors.append(f"{where}: planned robustness cells missing: {missing[:10]}"
                              f" (total {len(missing)})")
            if unexpected:
                errors.append(f"{where}: unregistered robustness cells: {unexpected[:10]}")


def _check_concentration(errors: list[str], evidence: dict[str, Any]) -> None:
    section = evidence.get("concentration")
    where = "concentration"
    if not isinstance(section, dict):
        errors.append(f"{where} must be an object")
        return
    for key in _CONCENTRATION_FIELDS:
        _fraction(errors, section, key, where)


def _check_data_quality(errors: list[str], evidence: dict[str, Any]) -> None:
    section = evidence.get("data_quality")
    where = "data_quality"
    if not isinstance(section, dict):
        errors.append(f"{where} must be an object")
        return
    for key in ("missing_rows", "duplicate_rows", "gap_count", "invalid_price_count"):
        if not _nonneg_int(section.get(key)):
            errors.append(f"{where}.{key} must be a non-negative integer")
    _fraction(errors, section, "target_availability_rate", where)
    for key in ("probability_validity_confirmed", "schema_validation_passed"):
        if not isinstance(section.get(key), bool):
            errors.append(f"{where}.{key} must be a boolean")


def _check_execution_data(errors: list[str], evidence: dict[str, Any],
                          binding: dict[str, Any]) -> None:
    section = evidence.get("execution_data")
    where = "execution_data"
    if not isinstance(section, dict):
        errors.append(f"{where} must be an object")
        return
    for key in _EXECUTION_FIELDS:
        if not isinstance(section.get(key), bool):
            errors.append(f"{where}.{key} must be a boolean")
        elif section[key] != binding["execution_data"][key]:
            errors.append(f"{where}.{key} contradicts the dataset binding")


def _check_multiplicity(errors: list[str], evidence: dict[str, Any], plan: dict[str, Any]) -> None:
    section = evidence.get("multiplicity")
    where = "multiplicity"
    if not isinstance(section, dict):
        errors.append(f"{where} must be an object")
        return
    contract = plan["multiple_testing_contract"]
    if section.get("family_id") != contract["family_id"]:
        errors.append(f"{where}.family_id does not match the preregistered family")
    if section.get("family_size") != contract["family_size"]:
        errors.append(f"{where}.family_size ({section.get('family_size')!r}) does not match the "
                      f"preregistered family size ({contract['family_size']})")
    hypothesis_ids = section.get("hypothesis_ids")
    p_values = section.get("p_values")
    if not isinstance(hypothesis_ids, list) or \
            not all(isinstance(item, str) and item for item in hypothesis_ids):
        errors.append(f"{where}.hypothesis_ids must be a list of non-empty strings")
    if not isinstance(p_values, list) or not all(
            is_finite_number(value) and 0.0 <= float(value) <= 1.0 for value in p_values):
        errors.append(f"{where}.p_values must be a list of finite numbers in [0, 1]")
    if isinstance(hypothesis_ids, list) and isinstance(p_values, list):
        if len(hypothesis_ids) != len(p_values):
            errors.append(f"{where}: hypothesis_ids and p_values lengths differ")
        elif len(p_values) != contract["family_size"]:
            errors.append(f"{where}: exactly family_size ({contract['family_size']}) p-values are "
                          f"required, got {len(p_values)}")
        if isinstance(hypothesis_ids, list) and len(set(hypothesis_ids)) != len(hypothesis_ids):
            errors.append(f"{where}.hypothesis_ids must be unique")


def _check_artifacts(errors: list[str], evidence: dict[str, Any]) -> None:
    section = evidence.get("artifacts")
    where = "artifacts"
    if not isinstance(section, dict):
        errors.append(f"{where} must be an object")
        return
    for key in ("summary_path", "manifest_path"):
        try:
            normalize_logical_path(section.get(key, ""))
        except PathSecurityError as exc:
            errors.append(f"{where}.{key}: {exc}")
    hashes = section.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        errors.append(f"{where}.artifact_sha256 must be a non-empty object")
    else:
        for path, digest in hashes.items():
            try:
                normalize_logical_path(path)
            except PathSecurityError as exc:
                errors.append(f"{where}.artifact_sha256: {exc}")
            if not is_sha256_hex(digest):
                errors.append(f"{where}.artifact_sha256[{path!r}] must be a SHA-256 hex digest")
    if not is_sha256_hex(section.get("audit_sha256")):
        errors.append(f"{where}.audit_sha256 must be a SHA-256 hex digest")
    tests = section.get("test_evidence_sha256")
    if not isinstance(tests, dict) or not all(is_sha256_hex(value) for value in tests.values()):
        errors.append(f"{where}.test_evidence_sha256 must be an object of SHA-256 hex digests")


def evidence_errors(evidence: Any, *, plan: dict[str, Any], seal: dict[str, Any],
                    binding: dict[str, Any]) -> list[str]:
    """Every validation error for an evidence bundle (empty list means valid)."""
    if not isinstance(evidence, dict):
        return ["evidence bundle must be a JSON object"]
    errors: list[str] = []
    if evidence.get("evidence_version") != EVIDENCE_VERSION:
        errors.append(f"evidence_version must be {EVIDENCE_VERSION!r}")
    for sectionname in _TOP_LEVEL_SECTIONS:
        if sectionname not in evidence:
            errors.append(f"evidence is missing required section {sectionname!r}")
    if errors:
        return errors
    try:
        # Any NaN/inf smuggled in as Python floats is caught here.
        sha256_payload({key: value for key, value in evidence.items()
                        if key != "evidence_sha256"})
    except CanonicalJsonError as exc:
        return [f"evidence contains non-canonical values: {exc}"]
    _check_experiment(errors, evidence, seal, binding)
    _check_producer(errors, evidence, plan)
    _check_dataset(errors, evidence, binding)
    _check_population(errors, evidence, plan)
    _check_primary_model(errors, evidence)
    _check_primary_comparator(errors, evidence, plan)
    _check_negative_controls(errors, evidence)
    _check_robustness(errors, evidence, plan)
    _check_concentration(errors, evidence)
    _check_data_quality(errors, evidence)
    _check_execution_data(errors, evidence, binding)
    _check_multiplicity(errors, evidence, plan)
    _check_artifacts(errors, evidence)
    claimed = evidence.get("evidence_sha256")
    if claimed is not None:
        body = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
        if sha256_payload(body) != claimed:
            errors.append("evidence_sha256 does not match the evidence contents")
    return errors


def validate_evidence(evidence: Any, *, plan: dict[str, Any], seal: dict[str, Any],
                      binding: dict[str, Any]) -> dict[str, Any]:
    """Validate and return the evidence bundle with its self-hash attached."""
    errors = evidence_errors(evidence, plan=plan, seal=seal, binding=binding)
    if errors:
        raise EvidenceValidationError("invalid evidence bundle:\n- " + "\n- ".join(errors))
    result = dict(evidence)
    body = {key: value for key, value in result.items() if key != "evidence_sha256"}
    result["evidence_sha256"] = sha256_payload(body)
    return result

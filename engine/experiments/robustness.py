"""Robustness matrix and evidence-concentration evaluation.

The matrix is *planned* (which cells must exist, which are mandatory, what
pass proportion is required, how missing cells count) before evidence exists.
The evaluator never selects the best-performing cell, never drops a missing
cell from the denominator unless the plan preregistered that rule, and never
treats an insufficient cell as a pass.
"""

from __future__ import annotations

import statistics
import itertools
from typing import Any

from engine.experiments.canonical import is_finite_number
from engine.experiments.errors import GateEvaluationError

DIMENSION_KEYS = ("entry_policy", "split_boundary", "block_length", "time_slice",
                  "regime", "perturbation")


def cell_id_for_dimensions(dimensions: dict[str, Any]) -> str:
    if set(dimensions) != set(DIMENSION_KEYS):
        raise GateEvaluationError("robustness dimensions are incomplete or unexpected")
    return "|".join(f"{key}={dimensions[key]}" for key in DIMENSION_KEYS)


def planned_cell_registry(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sets = (contract["required_policy_slices"],
            contract["required_split_boundary_policies"],
            contract["required_block_lengths"], contract["required_time_slices"],
            contract["required_regime_slices"], contract["required_perturbations"])
    result: dict[str, dict[str, Any]] = {}
    for values in itertools.product(*sets):
        dimensions = dict(zip(DIMENSION_KEYS, values))
        result[cell_id_for_dimensions(dimensions)] = dimensions
    return result


def evaluate_robustness_matrix(
    *,
    robustness_contract: dict[str, Any],
    cells: list[dict[str, Any]],
    pass_rule: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score every reported cell against the preregistered contract.

    A cell passes when it is scored, meets the minimum per-cell sample count,
    and both improvements are strictly positive unless the caller supplies a
    preregistered ``pass_rule`` with explicit minimum improvements.
    """
    minimum_samples = robustness_contract["minimum_samples_per_cell"]
    mandatory = set(robustness_contract["mandatory_cells"])
    insufficient_allowed = set(robustness_contract["insufficient_allowed_cells"])
    missing_policy = robustness_contract["missing_cell_policy"]
    min_brier = float((pass_rule or {}).get("min_brier_improvement", 0.0))
    min_log_loss = float((pass_rule or {}).get("min_log_loss_improvement", 0.0))

    by_id: dict[str, dict[str, Any]] = {}
    for cell in cells:
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or not cell_id:
            raise GateEvaluationError("robustness cell without a cell_id")
        if cell_id in by_id:
            raise GateEvaluationError(f"duplicate robustness cell_id {cell_id!r}")
        by_id[cell_id] = cell

    reported_ids = set(by_id)
    missing_mandatory = sorted(mandatory - reported_ids)

    passed: list[str] = []
    failed: list[str] = []
    insufficient: list[str] = []
    improvements_brier: list[float] = []
    improvements_log_loss: list[float] = []
    mandatory_failures: list[str] = list(missing_mandatory)

    for cell_id in sorted(reported_ids):
        cell = by_id[cell_id]
        status = cell.get("status")
        if status == "missing":
            if cell_id in mandatory:
                mandatory_failures.append(cell_id)
            continue
        if status != "scored":
            insufficient.append(cell_id)
            if cell_id in mandatory and cell_id not in insufficient_allowed:
                mandatory_failures.append(cell_id)
            continue
        sample_count = cell.get("sample_count", 0)
        brier = cell.get("brier_improvement")
        log_loss = cell.get("log_loss_improvement")
        if not (is_finite_number(brier) and is_finite_number(log_loss)):
            raise GateEvaluationError(f"scored cell {cell_id!r} has non-finite improvements")
        improvements_brier.append(float(brier))
        improvements_log_loss.append(float(log_loss))
        if sample_count < minimum_samples:
            insufficient.append(cell_id)
            if cell_id in mandatory and cell_id not in insufficient_allowed:
                mandatory_failures.append(cell_id)
            continue
        if float(brier) > min_brier and float(log_loss) > min_log_loss:
            passed.append(cell_id)
        else:
            failed.append(cell_id)
            if cell_id in mandatory:
                mandatory_failures.append(cell_id)

    explicitly_missing = sorted(
        [cell_id for cell_id, cell in by_id.items() if cell.get("status") == "missing"]
        + missing_mandatory)
    declared_planned = (set(planned_cell_registry(robustness_contract))
                        if "required_split_boundary_policies" in robustness_contract
                        else set(robustness_contract.get("planned_cell_ids", [])))
    planned_ids = ((set(declared_planned) | mandatory) if declared_planned
                   else (set(reported_ids) | mandatory))
    unexpected = sorted(reported_ids - planned_ids) if declared_planned else []
    if unexpected:
        raise GateEvaluationError(f"unregistered robustness cells: {unexpected}")
    omitted = sorted(planned_ids - reported_ids)
    explicitly_missing = sorted(set(explicitly_missing) | set(omitted))
    mandatory_failures.extend(sorted(mandatory & set(omitted)))
    total_planned = len(planned_ids)
    evaluated = len(passed) + len(failed) + len(insufficient)

    # The denominator follows the preregistered missing-cell policy: cells the
    # plan demanded but the evidence omitted never silently disappear.
    if missing_policy == "count_as_failed":
        denominator = evaluated + len(explicitly_missing)
        numerator = len(passed)
    else:  # count_in_denominator
        denominator = total_planned
        numerator = len(passed)
    pass_ratio = (numerator / denominator) if denominator else 0.0

    sign_consistent = sum(1 for value in improvements_brier if value > 0)
    return {
        "total_planned_cells": total_planned,
        "evaluated_cells": evaluated,
        "passed_cells": sorted(passed),
        "failed_cells": sorted(failed),
        "insufficient_cells": sorted(insufficient),
        "missing_cells": explicitly_missing,
        "pass_ratio": pass_ratio,
        "minimum_pass_proportion": robustness_contract["minimum_pass_proportion"],
        "pass_ratio_met": pass_ratio >= robustness_contract["minimum_pass_proportion"],
        "mandatory_cell_failures": sorted(set(mandatory_failures)),
        "worst_brier_improvement": min(improvements_brier) if improvements_brier else None,
        "worst_log_loss_improvement": min(improvements_log_loss) if improvements_log_loss else None,
        "median_brier_improvement": (statistics.median(improvements_brier)
                                     if improvements_brier else None),
        "median_log_loss_improvement": (statistics.median(improvements_log_loss)
                                        if improvements_log_loss else None),
        "sign_consistency_rate": (sign_consistent / len(improvements_brier)
                                  if improvements_brier else None),
    }


def evaluate_concentration(
    *,
    concentration_limits: dict[str, Any],
    concentration_evidence: dict[str, Any],
    accepted_rows: int,
) -> dict[str, Any]:
    """Compare observed concentration fractions against preregistered limits.

    Distinguishes three situations explicitly: within limits; concentrated
    because the sample is tiny (below the preregistered small-sample
    threshold — inconclusive, not damning); and dangerously concentrated
    evidence in an adequately sized sample (a hard failure — one prolonged
    episode must never carry a promotion).
    """
    small_sample = accepted_rows < concentration_limits["small_sample_rows_threshold"]
    checks: list[dict[str, Any]] = []
    breaches = 0
    pairs = (
        ("largest_signal_episode_fraction", "max_single_signal_episode_fraction"),
        ("largest_target_cluster_fraction", "max_single_target_cluster_fraction"),
        ("largest_causal_segment_fraction", "max_single_causal_segment_fraction"),
        ("largest_day_fraction", "max_single_day_fraction"),
        ("largest_component_fraction", "max_single_component_fraction"),
        ("largest_regime_fraction", "max_single_regime_fraction"),
        ("largest_fallback_tier_fraction", "max_single_fallback_tier_fraction"),
    )
    for observed_key, limit_key in pairs:
        observed = concentration_evidence[observed_key]
        limit = concentration_limits[limit_key]
        breached = float(observed) > float(limit)
        if breached:
            breaches += 1
        checks.append({
            "dimension": observed_key,
            "observed": float(observed),
            "limit": float(limit),
            "breached": breached,
        })
    if breaches == 0:
        status = "within_limits"
    elif small_sample:
        status = "inconclusive_small_sample"
    else:
        status = "dangerously_concentrated"
    return {
        "accepted_rows": accepted_rows,
        "small_sample_rows_threshold": concentration_limits["small_sample_rows_threshold"],
        "small_sample": small_sample,
        "checks": checks,
        "breach_count": breaches,
        "status": status,
    }

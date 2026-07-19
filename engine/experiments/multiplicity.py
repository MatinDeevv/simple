"""Explicit multiple-testing control.

Supported methods: NONE_SINGLE_PRIMARY (family of exactly one),
BONFERRONI, and HOLM_BONFERRONI with deterministic ordering and stable tie
handling (ties broken by hypothesis ID, never by input order). Correction is
applied only to the preregistered primary family — it never transforms
secondary exploratory metrics into primary gates, and the family size is
fixed at plan time, before any result exists.
"""

from __future__ import annotations

from typing import Any

from engine.experiments.canonical import is_finite_number
from engine.experiments.errors import GateEvaluationError


def _validate_inputs(hypothesis_ids: list[str], p_values: list[float], alpha: float,
                     family_size: int) -> None:
    if not (is_finite_number(alpha) and 0.0 < float(alpha) < 1.0):
        raise GateEvaluationError(f"familywise alpha must be in (0, 1), got {alpha!r}")
    if not isinstance(family_size, int) or isinstance(family_size, bool) or family_size < 1:
        raise GateEvaluationError(f"family_size must be a positive integer, got {family_size!r}")
    if len(hypothesis_ids) != len(p_values):
        raise GateEvaluationError("hypothesis_ids and p_values must have equal length")
    if len(hypothesis_ids) != family_size:
        raise GateEvaluationError(
            f"declared family_size ({family_size}) does not match the number of registered "
            f"hypotheses ({len(hypothesis_ids)}); a family may never shrink after testing")
    if len(set(hypothesis_ids)) != len(hypothesis_ids):
        raise GateEvaluationError("hypothesis_ids must be unique")
    for hypothesis_id, p_value in zip(hypothesis_ids, p_values):
        if not (is_finite_number(p_value) and 0.0 <= float(p_value) <= 1.0):
            raise GateEvaluationError(
                f"p-value for {hypothesis_id!r} must be a finite number in [0, 1], got {p_value!r}")


def apply_correction(
    *,
    method: str,
    hypothesis_ids: list[str],
    p_values: list[float],
    familywise_alpha: float,
    family_size: int,
    primary_hypothesis_index: int,
) -> dict[str, Any]:
    """Return per-hypothesis rejection decisions plus the primary outcome.

    The result records, for every hypothesis: the raw p-value, the adjusted
    threshold it was compared against, and an explicit rejected / not-rejected
    decision. "Rejected" means the *null* is rejected, i.e. the hypothesis
    survives multiplicity control.
    """
    _validate_inputs(hypothesis_ids, p_values, familywise_alpha, family_size)
    if not (0 <= primary_hypothesis_index < family_size):
        raise GateEvaluationError("primary_hypothesis_index is outside the family")

    results: list[dict[str, Any]] = []
    if method == "NONE_SINGLE_PRIMARY":
        if family_size != 1:
            raise GateEvaluationError("NONE_SINGLE_PRIMARY requires a family of exactly one "
                                      "registered primary hypothesis")
        rejected = float(p_values[0]) <= float(familywise_alpha)
        results.append({
            "hypothesis_id": hypothesis_ids[0],
            "p_value": float(p_values[0]),
            "adjusted_threshold": float(familywise_alpha),
            "rejected": rejected,
        })
    elif method == "BONFERRONI":
        threshold = float(familywise_alpha) / family_size
        for hypothesis_id, p_value in zip(hypothesis_ids, p_values):
            results.append({
                "hypothesis_id": hypothesis_id,
                "p_value": float(p_value),
                "adjusted_threshold": threshold,
                "rejected": float(p_value) <= threshold,
            })
    elif method == "HOLM_BONFERRONI":
        # Deterministic order: ascending p-value, ties broken by hypothesis ID.
        order = sorted(range(family_size),
                       key=lambda index: (float(p_values[index]), hypothesis_ids[index]))
        decisions: dict[int, dict[str, Any]] = {}
        stopped = False
        for rank, index in enumerate(order):
            threshold = float(familywise_alpha) / (family_size - rank)
            p_value = float(p_values[index])
            rejected = (not stopped) and p_value <= threshold
            if not rejected:
                stopped = True
            decisions[index] = {
                "hypothesis_id": hypothesis_ids[index],
                "p_value": p_value,
                "adjusted_threshold": threshold,
                "rejected": rejected,
            }
        results = [decisions[index] for index in range(family_size)]
    else:
        raise GateEvaluationError(f"unsupported multiple-testing method: {method!r}")

    primary = results[primary_hypothesis_index]
    return {
        "method": method,
        "family_size": family_size,
        "familywise_alpha": float(familywise_alpha),
        "results": results,
        "primary_hypothesis_id": primary["hypothesis_id"],
        "primary_p_value": primary["p_value"],
        "primary_adjusted_threshold": primary["adjusted_threshold"],
        "primary_rejected": primary["rejected"],
    }

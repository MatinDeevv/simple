"""Gate system: hard validity gates first, statistical evidence gates second.

Every gate result is a structured record — gate_id, category, description,
required, observed_value, threshold, comparison, passed, status, reason,
evidence_path — never a bare boolean. A fact that was not reported is MISSING
and counts as not passed; the Tribunal never defaults to success.
"""

from __future__ import annotations

import math
from typing import Any

from engine.experiments.canonical import is_finite_number
from engine.experiments.errors import GateEvaluationError

PASS = "PASS"
FAIL = "FAIL"
INSUFFICIENT = "INSUFFICIENT"
MISSING = "MISSING"
INVALID = "INVALID"
NOT_APPLICABLE = "NOT_APPLICABLE"

GATE_STATUSES: tuple[str, ...] = (PASS, FAIL, INSUFFICIENT, MISSING, INVALID, NOT_APPLICABLE)

# Fixed, ordered registry of hard validity gates. Every one is required; a
# single hard failure makes the whole experiment INVALID_EXPERIMENT.
HARD_VALIDITY_GATES: tuple[tuple[str, str], ...] = (
    ("hard_state_chain", "experiment state chain is valid and complete"),
    ("hard_plan_hash", "plan bytes match the sealed plan hash"),
    ("hard_seal_integrity", "seal artifact is hash-consistent"),
    ("hard_dataset_binding", "dataset binding is hash-consistent with the seal"),
    ("hard_no_undeclared_amendment", "no undeclared amendment of the sealed plan"),
    ("hard_no_clean_holdout_reuse", "holdout not reused while claiming to be untouched"),
    ("hard_code_commit_match", "evidence code commit matches preregistration"),
    ("hard_configuration_match", "evidence configuration hash matches preregistration"),
    ("hard_model_contract_match", "evidence model contract matches preregistration"),
    ("hard_dataset_hash_match", "evidence dataset hashes match the binding"),
    ("hard_evidence_complete", "evidence bundle is complete and internally valid"),
    ("hard_audit_chain", "audit hash chain verifies end to end"),
    ("hard_probabilities_valid", "probability outputs were validated by the producer"),
    ("hard_finite_metrics", "all reported metrics are finite"),
    ("hard_mandatory_controls", "every mandatory negative control is present"),
    ("hard_mandatory_policies", "every mandatory entry-policy view is present"),
    ("hard_mandatory_block_lengths", "every mandatory bootstrap block length is present"),
    ("hard_no_path_traversal", "no artifact path escapes its root"),
    ("hard_no_duplicate_evidence_ids", "no duplicate evidence identifiers"),
)


def gate_result(
    *,
    gate_id: str,
    category: str,
    description: str,
    required: bool,
    observed_value: Any,
    threshold: Any,
    comparison: str,
    passed: bool,
    status: str,
    reason: str,
    evidence_path: str,
) -> dict[str, Any]:
    if status not in GATE_STATUSES:
        raise GateEvaluationError(f"unknown gate status {status!r}")
    return {
        "gate_id": gate_id,
        "category": category,
        "description": description,
        "required": required,
        "observed_value": observed_value,
        "threshold": threshold,
        "comparison": comparison,
        "passed": bool(passed),
        "status": status,
        "reason": reason,
        "evidence_path": evidence_path,
    }


def evaluate_hard_validity_gates(facts: dict[str, tuple[bool, str]]) -> list[dict[str, Any]]:
    """Evaluate the fixed hard-gate registry against caller-established facts.

    ``facts`` maps gate_id to ``(holds, reason)``. A gate with no recorded
    fact is MISSING and not passed — silence is never validity.
    """
    results: list[dict[str, Any]] = []
    for gate_id, description in HARD_VALIDITY_GATES:
        if gate_id not in facts:
            results.append(gate_result(
                gate_id=gate_id, category="hard_validity", description=description,
                required=True, observed_value=None, threshold=True, comparison="==",
                passed=False, status=MISSING,
                reason="no fact was recorded for this hard gate; unreported is not passed",
                evidence_path=""))
            continue
        holds, reason = facts[gate_id]
        results.append(gate_result(
            gate_id=gate_id, category="hard_validity", description=description,
            required=True, observed_value=bool(holds), threshold=True, comparison="==",
            passed=bool(holds), status=PASS if holds else FAIL,
            reason=reason, evidence_path=""))
    return results


_MISSING = object()


def resolve_metric_path(namespace: dict[str, Any], path: str) -> Any:
    """Resolve a dot path like ``primary_model.brier_improvement``; returns a
    sentinel when any step is absent."""
    value: Any = namespace
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _compare(observed: float, comparison: str, threshold: float) -> bool:
    if comparison == ">=":
        return observed >= threshold
    if comparison == ">":
        return observed > threshold
    if comparison == "<=":
        return observed <= threshold
    if comparison == "<":
        return observed < threshold
    if comparison == "==":
        return math.isclose(observed, threshold, rel_tol=1e-9, abs_tol=1e-12)
    raise GateEvaluationError(f"unsupported gate comparison {comparison!r}")


def evaluate_statistical_gates(
    acceptance_gates: list[dict[str, Any]],
    namespace: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate every preregistered acceptance gate against the evidence
    namespace (the evidence bundle plus Tribunal-derived values under
    ``derived.*``). Gate order follows the plan and is deterministic."""
    results: list[dict[str, Any]] = []
    for gate in acceptance_gates:
        observed = resolve_metric_path(namespace, gate["metric_path"])
        base = dict(
            gate_id=gate["gate_id"], category=gate["category"],
            description=gate["description"], required=gate["required"],
            threshold=gate["threshold"], comparison=gate["comparison"],
            evidence_path=gate["metric_path"])
        if observed is _MISSING:
            results.append(gate_result(
                **base, observed_value=None, passed=False, status=MISSING,
                reason=f"metric path {gate['metric_path']!r} is absent from the evidence"))
            continue
        if isinstance(observed, bool):
            observed = int(observed)
        if not is_finite_number(observed):
            results.append(gate_result(
                **base, observed_value=None, passed=False, status=INVALID,
                reason=f"metric at {gate['metric_path']!r} is not a finite number"))
            continue
        passed = _compare(float(observed), gate["comparison"], float(gate["threshold"]))
        if passed:
            status = PASS
            reason = (f"observed {float(observed):.6g} {gate['comparison']} "
                      f"threshold {float(gate['threshold']):.6g}")
        elif gate["category"] == "power":
            # A failed power gate means the experiment lacked evidence, not
            # that the hypothesis failed; the verdict engine maps this to
            # INCONCLUSIVE rather than REJECTED.
            status = INSUFFICIENT
            reason = (f"observed {float(observed):.6g} violates {gate['comparison']} "
                      f"{float(gate['threshold']):.6g}: insufficient evidence, not a rejection")
        else:
            status = FAIL
            reason = (f"observed {float(observed):.6g} violates {gate['comparison']} "
                      f"{float(gate['threshold']):.6g}")
        results.append(gate_result(**base, observed_value=float(observed),
                                   passed=passed, status=status, reason=reason))
    return results


def failed_gates(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [result for result in results if result["status"] == FAIL]


def insufficient_gates(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [result for result in results
            if result["status"] in (INSUFFICIENT, MISSING, INVALID)]

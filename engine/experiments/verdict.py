"""Deterministic verdict engine.

Hierarchy (worst first): INVALID_EXPERIMENT, REJECTED, INCONCLUSIVE,
RESEARCH_ONLY, FORWARD_TEST_ELIGIBLE, then two future-evidence verdicts —
PAPER_FORWARD_TEST_PASSED and MICRO_LIVE_REVIEW_REQUIRED — whose schemas are
supported but which can never be produced from historical evidence.

There is no LIVE_READY, PRODUCTION_READY, PROFITABLE, or DEPLOY verdict; the
grammar cannot express them. Under the current BID-only data contract the
highest historical verdict is FORWARD_TEST_ELIGIBLE, and the Tribunal only
ever recommends advancing one research-reality level at a time.
"""

from __future__ import annotations

from typing import Any

from engine.experiments.canonical import (
    normalize_utc_timestamp,
    normalize_uuid,
    sha256_payload,
)
from engine.experiments.errors import VerdictIntegrityError
from engine.experiments.gates import FAIL, INSUFFICIENT, INVALID, MISSING, PASS

INVALID_EXPERIMENT = "INVALID_EXPERIMENT"
REJECTED = "REJECTED"
INCONCLUSIVE = "INCONCLUSIVE"
RESEARCH_ONLY = "RESEARCH_ONLY"
FORWARD_TEST_ELIGIBLE = "FORWARD_TEST_ELIGIBLE"
PAPER_FORWARD_TEST_PASSED = "PAPER_FORWARD_TEST_PASSED"
MICRO_LIVE_REVIEW_REQUIRED = "MICRO_LIVE_REVIEW_REQUIRED"

VERDICTS: tuple[str, ...] = (
    INVALID_EXPERIMENT, REJECTED, INCONCLUSIVE, RESEARCH_ONLY,
    FORWARD_TEST_ELIGIBLE, PAPER_FORWARD_TEST_PASSED, MICRO_LIVE_REVIEW_REQUIRED,
)

# These strings are deliberately not verdicts. Nothing in this module can
# return them; the tuple exists so tests can assert the prohibition.
FORBIDDEN_VERDICTS: tuple[str, ...] = ("LIVE_READY", "PRODUCTION_READY", "PROFITABLE", "DEPLOY")

_PROMOTION_RANK = {RESEARCH_ONLY: 0, FORWARD_TEST_ELIGIBLE: 1, PAPER_FORWARD_TEST_PASSED: 2}

RESEARCH_REALITY_LADDER: tuple[str, ...] = (
    "LEVEL_0_SYNTHETIC_VALIDATION_ONLY",
    "LEVEL_1_HISTORICAL_BID_ONLY_RESEARCH",
    "LEVEL_2_QUOTE_AND_COST_AWARE_HISTORICAL_RESEARCH",
    "LEVEL_3_PREREGISTERED_PAPER_FORWARD_TEST",
    "LEVEL_4_MICRO_LIVE_HUMAN_REVIEW",
    "LEVEL_5_PRODUCTION_GOVERNANCE",
)

VERDICT_VERSION = "edge-tribunal-verdict-v1"


def _next_stage_for(verdict: str, evidence_level: str) -> str:
    """One level at a time: the maximum recommended next stage."""
    if verdict != FORWARD_TEST_ELIGIBLE:
        return evidence_level
    return "LEVEL_3_PREREGISTERED_PAPER_FORWARD_TEST"


def decide_verdict(
    *,
    plan: dict[str, Any],
    binding: dict[str, Any],
    hard_gate_results: list[dict[str, Any]],
    statistical_gate_results: list[dict[str, Any]],
    robustness_report: dict[str, Any],
    concentration_report: dict[str, Any],
    multiplicity_report: dict[str, Any],
    evidence_sha256: str,
    verdict_id: str,
    decided_at_utc: str,
    evidence_type: str = "historical",
) -> dict[str, Any]:
    """Issue one deterministic verdict from evaluated gates.

    ``evidence_type`` is ``"historical"`` for every current experiment; the
    ``"paper_forward"`` type exists so a future preregistered paper forward
    test can be represented, and even then the terminal recommendation is
    MICRO_LIVE_REVIEW_REQUIRED — independent human, regulatory, and risk
    review — never an authorization to trade.
    """
    if evidence_type not in ("historical", "paper_forward"):
        raise VerdictIntegrityError(f"unknown evidence_type {evidence_type!r}")

    on_insufficient = {gate["gate_id"]: gate["on_insufficient"]
                       for gate in plan["acceptance_gates"]}

    hard_failures = [result for result in hard_gate_results if result["status"] != PASS]

    invalid_reasons: list[str] = [
        f"{result['gate_id']}: {result['reason']}" for result in hard_failures]
    failed: list[dict[str, Any]] = []
    inconclusive: list[dict[str, Any]] = []
    for result in statistical_gate_results:
        status = result["status"]
        if status == PASS:
            continue
        if status == FAIL:
            if result["required"]:
                failed.append(result)
            continue
        if status in (MISSING, INVALID):
            if result["required"]:
                invalid_reasons.append(
                    f"{result['gate_id']}: required gate could not be evaluated "
                    f"({status}): {result['reason']}")
            continue
        if status == INSUFFICIENT:
            policy = on_insufficient.get(result["gate_id"], "inconclusive")
            if policy == "invalid":
                invalid_reasons.append(f"{result['gate_id']}: insufficient and the plan "
                                       f"declared insufficiency invalid")
            elif policy == "fail":
                if result["required"]:
                    failed.append(result)
            else:
                inconclusive.append(result)

    if not robustness_report["pass_ratio_met"]:
        failed.append({
            "gate_id": "robustness_pass_ratio", "required": True,
            "reason": (f"robustness pass ratio {robustness_report['pass_ratio']:.4f} is below "
                       f"the preregistered minimum "
                       f"{robustness_report['minimum_pass_proportion']:.4f}")})
    if robustness_report["mandatory_cell_failures"]:
        failed.append({
            "gate_id": "robustness_mandatory_cells", "required": True,
            "reason": (f"mandatory robustness cells failed or were missing: "
                       f"{robustness_report['mandatory_cell_failures']}")})
    if concentration_report["status"] == "dangerously_concentrated":
        failed.append({
            "gate_id": "evidence_concentration", "required": True,
            "reason": "evidence is dangerously concentrated in a single episode/cluster/"
                      "segment despite an adequately sized sample"})
    elif concentration_report["status"] == "inconclusive_small_sample":
        inconclusive.append({
            "gate_id": "evidence_concentration", "required": True,
            "reason": "concentration limits breached in a below-threshold sample: "
                      "inconclusive, not damning"})
    if not multiplicity_report["primary_rejected"]:
        failed.append({
            "gate_id": "multiplicity_adjusted_primary", "required": True,
            "reason": (f"primary p-value {multiplicity_report['primary_p_value']:.6g} does not "
                       f"survive {multiplicity_report['method']} at adjusted threshold "
                       f"{multiplicity_report['primary_adjusted_threshold']:.6g}")})

    is_amendment = plan["identity"].get("parent_experiment_id") is not None
    holdout_blocked = bool(binding.get("promotion_blocked_by_holdout")) or \
        binding.get("holdout_claim_status") == "FORENSIC_REUSE"

    why_not_higher: list[str] = []
    if invalid_reasons:
        verdict = INVALID_EXPERIMENT
        validity_status = "INVALID"
        why_not_higher.append("the experiment is invalid; no evidential verdict is possible")
    elif failed:
        verdict = REJECTED
        validity_status = "VALID"
        why_not_higher.append("at least one mandatory preregistered gate clearly failed")
    elif inconclusive:
        verdict = INCONCLUSIVE
        validity_status = "VALID"
        why_not_higher.append("the evidence lacks the preregistered power or independence to "
                              "establish success; absence of rejection is not proof")
    else:
        validity_status = "VALID"
        ceiling = plan["promotion_ceiling"]["maximum_verdict"]
        ceiling_rank = _PROMOTION_RANK.get(ceiling, 0)
        binding_ceiling = binding["maximum_verdict_after_binding"]
        binding_rank = _PROMOTION_RANK.get(binding_ceiling, 0)
        effective_rank = min(ceiling_rank, binding_rank)
        # Historical evidence can never exceed forward-test eligibility.
        if evidence_type == "historical":
            effective_rank = min(effective_rank, _PROMOTION_RANK[FORWARD_TEST_ELIGIBLE])
            why_not_higher.append("historical BID-only evidence cannot demonstrate execution "
                                  "realism; a preregistered paper forward test is required next")
        if holdout_blocked:
            effective_rank = 0
            why_not_higher.append("the holdout was reused (forensic reuse): statistical passes "
                                  "on re-examined data cannot earn clean promotion")
        if is_amendment:
            effective_rank = 0
            why_not_higher.append("this experiment amends a sealed parent plan; amended plans "
                                  "are promotion-blocked by construction")
        if not binding.get("execution_data_complete", False):
            why_not_higher.append("execution inputs (ask/spread/fill/commission/latency/impact/"
                                  "notional/conversion) are absent from the data contract")
        verdict = {0: RESEARCH_ONLY, 1: FORWARD_TEST_ELIGIBLE,
                   2: FORWARD_TEST_ELIGIBLE}[effective_rank]

    assert verdict in VERDICTS and verdict not in FORBIDDEN_VERDICTS

    evidence_level = binding.get("maximum_evidence_level",
                                 "LEVEL_1_HISTORICAL_BID_ONLY_RESEARCH")
    payload: dict[str, Any] = {
        "verdict_version": VERDICT_VERSION,
        "verdict_id": normalize_uuid(verdict_id),
        "experiment_id": binding["experiment_id"],
        "plan_sha256": binding["plan_sha256"],
        "seal_sha256": binding["seal_sha256"],
        "binding_sha256": binding["binding_sha256"],
        "evidence_sha256": evidence_sha256,
        "decided_at_utc": normalize_utc_timestamp(decided_at_utc),
        "evidence_type": evidence_type,
        "verdict": verdict,
        "validity_status": validity_status,
        "invalid_reasons": invalid_reasons,
        "failed_gates": [
            {"gate_id": item["gate_id"], "reason": item["reason"]} for item in failed],
        "insufficient_gates": [
            {"gate_id": item["gate_id"], "reason": item["reason"]} for item in inconclusive],
        "hard_gate_results": hard_gate_results,
        "statistical_gate_results": statistical_gate_results,
        "robustness_report": robustness_report,
        "concentration_report": concentration_report,
        "multiplicity_report": multiplicity_report,
        "holdout_claim_status": binding.get("holdout_claim_status"),
        "amendment": is_amendment,
        "current_evidence_level": evidence_level,
        "maximum_next_stage": _next_stage_for(verdict, evidence_level),
        "why_not_higher": why_not_higher,
        "trading_authorization": False,
    }
    payload["verdict_sha256"] = sha256_payload(payload)
    return payload


def verify_verdict_integrity(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict) or "verdict_sha256" not in payload:
        return False
    body = {key: value for key, value in payload.items() if key != "verdict_sha256"}
    if payload.get("verdict") not in VERDICTS:
        return False
    if payload.get("trading_authorization") is not False:
        return False
    return sha256_payload(body) == payload["verdict_sha256"]

"""Deterministic human and machine reporting.

The Markdown report leads with the verdict, validity status, primary gates,
hard blockers, and maximum next stage — failures always appear before
successes, and a rejection can never be buried under attractive summaries.
Identical inputs produce byte-identical reports.
"""

from __future__ import annotations

from typing import Any

from engine.experiments.canonical import sha256_payload, strict_json_text
from engine.experiments.errors import VerdictIntegrityError
from engine.experiments.gates import PASS
from engine.experiments.verdict import verify_verdict_integrity

SCORECARD_VERSION = "edge-tribunal-scorecard-v1"
VERIFICATION_VERSION = "edge-tribunal-verification-v1"


def build_scorecard(verdict_payload: dict[str, Any]) -> dict[str, Any]:
    if not verify_verdict_integrity(verdict_payload):
        raise VerdictIntegrityError("refusing to build a scorecard from an unverified verdict")
    hard = verdict_payload["hard_gate_results"]
    statistical = verdict_payload["statistical_gate_results"]
    scorecard = {
        "scorecard_version": SCORECARD_VERSION,
        "experiment_id": verdict_payload["experiment_id"],
        "verdict": verdict_payload["verdict"],
        "validity_status": verdict_payload["validity_status"],
        "hard_gates_total": len(hard),
        "hard_gates_passed": sum(1 for gate in hard if gate["status"] == PASS),
        "statistical_gates_total": len(statistical),
        "statistical_gates_passed": sum(1 for gate in statistical if gate["status"] == PASS),
        "failed_gates": verdict_payload["failed_gates"],
        "insufficient_gates": verdict_payload["insufficient_gates"],
        "robustness_pass_ratio": verdict_payload["robustness_report"]["pass_ratio"],
        "concentration_status": verdict_payload["concentration_report"]["status"],
        "multiplicity_primary_rejected": verdict_payload["multiplicity_report"]["primary_rejected"],
        "maximum_next_stage": verdict_payload["maximum_next_stage"],
        "trading_authorization": False,
        "verdict_sha256": verdict_payload["verdict_sha256"],
    }
    scorecard["scorecard_sha256"] = sha256_payload(scorecard)
    return scorecard


def build_verification(
    *,
    verdict_payload: dict[str, Any],
    audit_head_sha256: str,
    audit_event_count: int,
    artifact_hashes: dict[str, str],
) -> dict[str, Any]:
    verification = {
        "verification_version": VERIFICATION_VERSION,
        "experiment_id": verdict_payload["experiment_id"],
        "verdict_sha256": verdict_payload["verdict_sha256"],
        "audit_chain_head_sha256": audit_head_sha256,
        "audit_event_count": audit_event_count,
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
        "verdict_integrity": verify_verdict_integrity(verdict_payload),
    }
    verification["verification_sha256"] = sha256_payload(verification)
    return verification


def _gate_line(gate: dict[str, Any]) -> str:
    observed = gate["observed_value"]
    observed_text = "-" if observed is None else f"{observed}"
    return (f"| {gate['gate_id']} | {gate['status']} | {observed_text} | "
            f"{gate['comparison']} {gate['threshold']} | {gate['reason']} |")


def _gate_table(gates: list[dict[str, Any]]) -> list[str]:
    lines = ["| gate | status | observed | requirement | reason |",
             "| --- | --- | --- | --- | --- |"]
    # Failures first, then insufficiencies, then passes; stable within groups.
    order = {"FAIL": 0, "INVALID": 1, "MISSING": 2, "INSUFFICIENT": 3,
             "NOT_APPLICABLE": 4, "PASS": 5}
    for gate in sorted(gates, key=lambda item: (order[item["status"]], item["gate_id"])):
        lines.append(_gate_line(gate))
    return lines


def build_report_markdown(
    *,
    plan: dict[str, Any],
    seal: dict[str, Any],
    binding: dict[str, Any],
    verdict_payload: dict[str, Any],
    verification: dict[str, Any],
) -> str:
    if not verify_verdict_integrity(verdict_payload):
        raise VerdictIntegrityError("refusing to report an unverified verdict")
    v = verdict_payload
    hard = v["hard_gate_results"]
    statistical = v["statistical_gate_results"]
    hard_blockers = [gate for gate in hard if gate["status"] != PASS]
    primary_gates = [gate for gate in statistical
                     if gate["gate_id"].startswith("primary_")]
    robustness = v["robustness_report"]
    concentration = v["concentration_report"]
    multiplicity = v["multiplicity_report"]
    execution = binding["execution_data"]

    lines: list[str] = []
    push = lines.append
    push("# Edge Tribunal report")
    push("")
    push(f"## VERDICT: {v['verdict']}")
    push("")
    push(f"- VALIDITY STATUS: **{v['validity_status']}**")
    push(f"- PRIMARY GATES: "
         f"{sum(1 for g in primary_gates if g['status'] == PASS)}/{len(primary_gates)} passed")
    push(f"- HARD BLOCKERS: {len(hard_blockers)}")
    for gate in hard_blockers:
        push(f"  - {gate['gate_id']}: {gate['reason']}")
    push(f"- MAXIMUM NEXT STAGE: {v['maximum_next_stage']}")
    push("")
    push("## WHY THIS IS NOT A TRADING AUTHORIZATION")
    push("")
    push("This verdict is a research-governance decision, nothing more. It does not prove")
    push("future profitability, does not authorize order placement, does not assess")
    push("execution costs, and confers no permission that skips the preregistered next")
    push("stage. The evidence contains no ask prices, spreads, fills, commissions,")
    push("latency, impact, capacity, notionals, or conversion prices; the highest")
    push("possible historical verdict is FORWARD_TEST_ELIGIBLE.")
    push("")
    push("## 1. Failed gates")
    push("")
    if v["invalid_reasons"]:
        push("Validity failures (these alone force INVALID_EXPERIMENT):")
        for reason in v["invalid_reasons"]:
            push(f"- {reason}")
        push("")
    if v["failed_gates"]:
        for item in v["failed_gates"]:
            push(f"- {item['gate_id']}: {item['reason']}")
    elif not v["invalid_reasons"]:
        push("None.")
    push("")
    push("## 2. Insufficient gates")
    push("")
    if v["insufficient_gates"]:
        for item in v["insufficient_gates"]:
            push(f"- {item['gate_id']}: {item['reason']}")
    else:
        push("None.")
    push("")
    push("## 3. Why the experiment did not receive a higher verdict")
    push("")
    for reason in v["why_not_higher"]:
        push(f"- {reason}")
    push("")
    push("## 4. Experiment identity")
    push("")
    push(f"- Experiment ID: `{v['experiment_id']}`")
    push(f"- Title: {plan['identity']['title']}")
    push(f"- Researcher: {plan['identity']['researcher']}")
    parent = plan["identity"].get("parent_experiment_id")
    push(f"- Amendment of: {f'`{parent}`' if parent else 'none (original registration)'}")
    push("")
    push("## 5. Hypothesis")
    push("")
    push(f"- Statement: {plan['hypothesis']['statement']}")
    push(f"- Mechanism: {plan['hypothesis']['economic_mechanism']}")
    push(f"- Expected failure mode: {plan['hypothesis']['expected_failure_mode']}")
    push("")
    push("## 6. Preregistration and code binding")
    push("")
    push(f"- Plan hash: `{v['plan_sha256']}`")
    push(f"- Seal hash: `{v['seal_sha256']}`")
    push(f"- Sealed at: {seal['sealed_at_utc']}")
    push(f"- Code commit: `{seal['code_commit_sha']}`")
    push(f"- Configuration hash: `{seal['configuration_sha256']}`")
    push("")
    push("## 7. Dataset binding and holdout")
    push("")
    push(f"- Binding hash: `{v['binding_sha256']}`")
    push(f"- Dataset: {binding['dataset_name']} ({binding['provenance']})")
    push(f"- Holdout interval: {binding['holdout_interval']['start_utc']} .. "
         f"{binding['holdout_interval']['end_utc']}")
    push(f"- Holdout claim status: {v['holdout_claim_status']} "
         f"(reuse kind: {binding['holdout_reuse_kind']})")
    push(f"- Price side: {binding['price_side']}")
    push("")
    push("## 8. Execution-data limitations")
    push("")
    for field in sorted(execution):
        push(f"- {field}: {'available' if execution[field] else 'ABSENT'}")
    push("")
    push("## 9. Primary metrics and confidence bounds")
    push("")
    push(_gate_table(primary_gates)[0])
    push(_gate_table(primary_gates)[1])
    for line in _gate_table(primary_gates)[2:]:
        push(line)
    push("")
    push("## 10. Multiplicity adjustment")
    push("")
    push(f"- Method: {multiplicity['method']} (family size {multiplicity['family_size']}, "
         f"familywise alpha {multiplicity['familywise_alpha']})")
    push(f"- Primary hypothesis `{multiplicity['primary_hypothesis_id']}`: "
         f"p = {multiplicity['primary_p_value']:.6g}, adjusted threshold "
         f"{multiplicity['primary_adjusted_threshold']:.6g}, "
         f"{'survives' if multiplicity['primary_rejected'] else 'DOES NOT survive'} correction")
    push("")
    push("## 11. Robustness matrix")
    push("")
    push(f"- Planned cells: {robustness['total_planned_cells']}, evaluated: "
         f"{robustness['evaluated_cells']}")
    push(f"- Passed: {len(robustness['passed_cells'])}, failed: "
         f"{len(robustness['failed_cells'])}, insufficient: "
         f"{len(robustness['insufficient_cells'])}, missing: "
         f"{len(robustness['missing_cells'])}")
    push(f"- Pass ratio: {robustness['pass_ratio']:.4f} "
         f"(required >= {robustness['minimum_pass_proportion']:.4f}; "
         f"{'met' if robustness['pass_ratio_met'] else 'NOT MET'})")
    push(f"- Mandatory-cell failures: {robustness['mandatory_cell_failures'] or 'none'}")
    push(f"- Worst Brier improvement: {robustness['worst_brier_improvement']}")
    push(f"- Median Brier improvement: {robustness['median_brier_improvement']}")
    push(f"- Sign-consistency rate: {robustness['sign_consistency_rate']}")
    push("")
    push("## 12. Evidence concentration")
    push("")
    push(f"- Status: {concentration['status']} (breaches: {concentration['breach_count']}, "
         f"accepted rows: {concentration['accepted_rows']})")
    for check in concentration["checks"]:
        marker = "BREACH" if check["breached"] else "ok"
        push(f"- {check['dimension']}: {check['observed']:.4f} vs limit "
             f"{check['limit']:.4f} [{marker}]")
    push("")
    push("## 13. Hard validity gates")
    push("")
    for line in _gate_table(hard):
        push(line)
    push("")
    push("## 14. Statistical gates (full)")
    push("")
    for line in _gate_table(statistical):
        push(line)
    push("")
    push("## 15. Audit-chain verification and artifact hashes")
    push("")
    push(f"- Audit chain head: `{verification['audit_chain_head_sha256']}`")
    push(f"- Audit events: {verification['audit_event_count']}")
    push(f"- Verdict hash: `{v['verdict_sha256']}`")
    for path, digest in sorted(verification["artifact_sha256"].items()):
        push(f"- `{path}`: `{digest}`")
    push("")
    push("---")
    push("A passed Tribunal verdict does not prove future profitability, does not")
    push("authorize trading, and only permits the next preregistered research stage.")
    push("")
    return "\n".join(lines)


def render_verification_json(verification: dict[str, Any]) -> str:
    return strict_json_text(verification)

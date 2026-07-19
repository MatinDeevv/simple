from __future__ import annotations

import math

import pytest

from engine.experiments import gates
from engine.experiments.errors import GateEvaluationError

REQUIRED_RESULT_FIELDS = (
    "gate_id", "category", "description", "required", "observed_value",
    "threshold", "comparison", "passed", "status", "reason", "evidence_path",
)


def _all_facts_true() -> dict[str, tuple[bool, str]]:
    return {gate_id: (True, "ok") for gate_id, _ in gates.HARD_VALIDITY_GATES}


def test_hard_gates_all_pass_with_true_facts() -> None:
    results = gates.evaluate_hard_validity_gates(_all_facts_true())
    assert len(results) == len(gates.HARD_VALIDITY_GATES)
    assert all(result["status"] == gates.PASS for result in results)
    for result in results:
        for field in REQUIRED_RESULT_FIELDS:
            assert field in result  # never a bare boolean


def test_hard_gate_missing_fact_is_missing_not_passed() -> None:
    facts = _all_facts_true()
    del facts["hard_audit_chain"]
    results = gates.evaluate_hard_validity_gates(facts)
    by_id = {result["gate_id"]: result for result in results}
    missing = by_id["hard_audit_chain"]
    assert missing["status"] == gates.MISSING
    assert missing["passed"] is False
    assert "unreported is not passed" in missing["reason"]


def test_hard_gate_false_fact_fails_with_reason() -> None:
    facts = _all_facts_true()
    facts["hard_plan_hash"] = (False, "plan bytes were altered after sealing")
    results = gates.evaluate_hard_validity_gates(facts)
    by_id = {result["gate_id"]: result for result in results}
    assert by_id["hard_plan_hash"]["status"] == gates.FAIL
    assert "altered" in by_id["hard_plan_hash"]["reason"]


def test_hard_gate_order_is_deterministic() -> None:
    first = [r["gate_id"] for r in gates.evaluate_hard_validity_gates(_all_facts_true())]
    second = [r["gate_id"] for r in gates.evaluate_hard_validity_gates(_all_facts_true())]
    assert first == second == [gate_id for gate_id, _ in gates.HARD_VALIDITY_GATES]


def _gate(gate_id: str = "g1", *, category: str = "statistical",
          metric_path: str = "a.b", comparison: str = ">=", threshold: float = 1.0,
          required: bool = True) -> dict:
    return {"gate_id": gate_id, "category": category, "description": "test gate",
            "metric_path": metric_path, "comparison": comparison,
            "threshold": threshold, "required": required,
            "on_insufficient": "inconclusive"}


def test_statistical_gate_pass_and_fail() -> None:
    namespace = {"a": {"b": 2.0}}
    passed = gates.evaluate_statistical_gates([_gate()], namespace)[0]
    assert passed["status"] == gates.PASS and passed["passed"] is True
    assert passed["observed_value"] == 2.0
    failed = gates.evaluate_statistical_gates(
        [_gate(threshold=5.0)], namespace)[0]
    assert failed["status"] == gates.FAIL and failed["passed"] is False
    assert "violates" in failed["reason"]


def test_missing_metric_path_is_missing() -> None:
    result = gates.evaluate_statistical_gates(
        [_gate(metric_path="does.not.exist")], {"a": {"b": 1}})[0]
    assert result["status"] == gates.MISSING
    assert result["observed_value"] is None


def test_non_finite_metric_is_invalid() -> None:
    result = gates.evaluate_statistical_gates(
        [_gate()], {"a": {"b": math.nan}})[0]
    assert result["status"] == gates.INVALID
    result = gates.evaluate_statistical_gates(
        [_gate()], {"a": {"b": "a string"}})[0]
    assert result["status"] == gates.INVALID


def test_failed_power_gate_is_insufficient_not_fail() -> None:
    result = gates.evaluate_statistical_gates(
        [_gate(category="power", metric_path="population.oos_rows", threshold=500)],
        {"population": {"oos_rows": 100}})[0]
    assert result["status"] == gates.INSUFFICIENT
    assert result["passed"] is False
    assert "not a rejection" in result["reason"]


@pytest.mark.parametrize("comparison,observed,threshold,expected", [
    (">=", 1.0, 1.0, True), (">", 1.0, 1.0, False), ("<=", 0.5, 1.0, True),
    ("<", 1.0, 0.5, False), ("==", 0.3, 0.3, True), ("==", 0.3000001, 0.3, False),
])
def test_comparisons(comparison: str, observed: float, threshold: float,
                     expected: bool) -> None:
    result = gates.evaluate_statistical_gates(
        [_gate(comparison=comparison, threshold=threshold)],
        {"a": {"b": observed}})[0]
    assert result["passed"] is expected


def test_boolean_metric_is_coerced_to_integer() -> None:
    result = gates.evaluate_statistical_gates(
        [_gate(metric_path="flag", comparison=">=", threshold=1)],
        {"flag": True})[0]
    assert result["passed"] is True and result["observed_value"] == 1.0


def test_unknown_comparison_raises() -> None:
    with pytest.raises(GateEvaluationError):
        gates.evaluate_statistical_gates(
            [_gate(comparison="~=")], {"a": {"b": 1.0}})


def test_unknown_status_rejected_by_result_factory() -> None:
    with pytest.raises(GateEvaluationError):
        gates.gate_result(gate_id="x", category="statistical", description="d",
                          required=True, observed_value=1, threshold=1,
                          comparison=">=", passed=True, status="GREEN",
                          reason="r", evidence_path="p")


def test_failed_and_insufficient_helpers() -> None:
    namespace = {"a": {"b": 0.0}, "population": {"oos_rows": 10}}
    results = gates.evaluate_statistical_gates(
        [_gate(gate_id="fail-me", threshold=1.0),
         _gate(gate_id="power-me", category="power",
               metric_path="population.oos_rows", threshold=500),
         _gate(gate_id="miss-me", metric_path="nope")],
        namespace)
    assert [r["gate_id"] for r in gates.failed_gates(results)] == ["fail-me"]
    assert {r["gate_id"] for r in gates.insufficient_gates(results)} == \
        {"power-me", "miss-me"}


def test_gate_evaluation_order_follows_the_plan() -> None:
    plan_gates = [_gate(gate_id=f"g{index}", metric_path="a.b")
                  for index in range(10)]
    results = gates.evaluate_statistical_gates(plan_gates, {"a": {"b": 5.0}})
    assert [result["gate_id"] for result in results] == \
        [gate["gate_id"] for gate in plan_gates]

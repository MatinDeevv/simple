from __future__ import annotations

import time
from pathlib import Path

import pytest

from engine.experiments import robustness
from engine.experiments.errors import GateEvaluationError

CONTRACT = {
    "mandatory_cells": ["cell-a", "cell-b"],
    "minimum_pass_proportion": 0.7,
    "insufficient_allowed_cells": [],
    "minimum_samples_per_cell": 50,
    "missing_cell_policy": "count_in_denominator",
}


def cell(cell_id: str, *, status: str = "scored", samples: int = 100,
         brier: float = 0.002, log_loss: float = 0.004) -> dict:
    return {"cell_id": cell_id, "dimensions": {}, "status": status,
            "sample_count": samples, "brier_improvement": brier,
            "log_loss_improvement": log_loss}


def evaluate(cells: list[dict], contract: dict | None = None) -> dict:
    return robustness.evaluate_robustness_matrix(
        robustness_contract=contract or CONTRACT, cells=cells)


def test_all_pass_produces_full_ratio() -> None:
    report = evaluate([cell("cell-a"), cell("cell-b"), cell("cell-c")])
    assert report["passed_cells"] == ["cell-a", "cell-b", "cell-c"]
    assert report["pass_ratio"] == pytest.approx(1.0)
    assert report["pass_ratio_met"] is True
    assert report["mandatory_cell_failures"] == []
    assert report["sign_consistency_rate"] == pytest.approx(1.0)


def test_missing_mandatory_cell_fails() -> None:
    report = evaluate([cell("cell-a"), cell("cell-c")])
    assert "cell-b" in report["mandatory_cell_failures"]
    assert "cell-b" in report["missing_cells"]


def test_failed_mandatory_cell_is_reported() -> None:
    report = evaluate([cell("cell-a"), cell("cell-b", brier=-0.001), cell("cell-c")])
    assert "cell-b" in report["mandatory_cell_failures"]
    assert "cell-b" in report["failed_cells"]


def test_insufficient_cell_is_not_silently_passed() -> None:
    report = evaluate([cell("cell-a"), cell("cell-b", samples=10), cell("cell-c")])
    assert "cell-b" in report["insufficient_cells"]
    assert "cell-b" not in report["passed_cells"]
    assert "cell-b" in report["mandatory_cell_failures"]


def test_insufficient_allowed_cell_does_not_fail_mandatory() -> None:
    contract = dict(CONTRACT, insufficient_allowed_cells=["cell-b"])
    report = evaluate([cell("cell-a"),
                       cell("cell-b", status="insufficient_oos_population"),
                       cell("cell-c")], contract)
    assert "cell-b" in report["insufficient_cells"]
    assert report["mandatory_cell_failures"] == []


def test_pass_ratio_denominator_follows_missing_cell_policy() -> None:
    cells = [cell("cell-a"), cell("cell-c", brier=-0.01)]
    in_denominator = evaluate(cells, dict(CONTRACT,
                                          missing_cell_policy="count_in_denominator"))
    as_failed = evaluate(cells, dict(CONTRACT, missing_cell_policy="count_as_failed"))
    # cell-b is mandatory but missing; both policies keep it in the denominator.
    assert in_denominator["pass_ratio"] == pytest.approx(1 / 3)
    assert as_failed["pass_ratio"] == pytest.approx(1 / 3)
    assert in_denominator["pass_ratio_met"] is False


def test_missing_cells_never_leave_the_denominator() -> None:
    report = evaluate([cell("cell-a"), cell("cell-b"),
                       cell("cell-x", status="missing")])
    assert "cell-x" in report["missing_cells"]
    assert report["pass_ratio"] < 1.0


def test_duplicate_cell_ids_rejected() -> None:
    with pytest.raises(GateEvaluationError, match="duplicate"):
        evaluate([cell("cell-a"), cell("cell-a")])


def test_non_finite_improvement_rejected() -> None:
    with pytest.raises(GateEvaluationError, match="non-finite"):
        evaluate([cell("cell-a", brier=float("nan")), cell("cell-b")])


def test_sign_reversal_is_visible_in_consistency_rate() -> None:
    report = evaluate([cell("cell-a"), cell("cell-b"),
                       cell("cell-c", brier=-0.002), cell("cell-d", brier=-0.003)])
    assert report["sign_consistency_rate"] == pytest.approx(0.5)
    assert report["worst_brier_improvement"] == pytest.approx(-0.003)


def test_no_score_based_selection_every_cell_is_reported() -> None:
    cells = [cell(f"cell-{index}", brier=0.001 * index) for index in range(1, 8)]
    cells.extend([cell("cell-a"), cell("cell-b")])
    report = evaluate(cells)
    total = (len(report["passed_cells"]) + len(report["failed_cells"])
             + len(report["insufficient_cells"]))
    assert total == len(cells)  # nothing dropped, nothing cherry-picked
    # Deterministic ordering, not score ordering.
    assert report["passed_cells"] == sorted(report["passed_cells"])


def test_thousand_cells_evaluate_quickly() -> None:
    cells = [cell(f"cell-{index:04d}") for index in range(1000)]
    cells.append(cell("cell-a"))
    cells.append(cell("cell-b"))
    started = time.perf_counter()
    report = evaluate(cells)
    elapsed = time.perf_counter() - started
    assert len(report["passed_cells"]) == 1002
    assert elapsed < 5.0  # generous ceiling; the evaluator is linear


LIMITS = {
    "max_single_signal_episode_fraction": 0.2,
    "max_single_target_cluster_fraction": 0.2,
    "max_single_causal_segment_fraction": 0.5,
    "max_single_day_fraction": 0.3,
    "max_single_component_fraction": 0.6,
    "max_single_regime_fraction": 0.8,
    "max_single_fallback_tier_fraction": 0.9,
    "small_sample_rows_threshold": 200,
}


def concentration(**overrides) -> dict:
    base = {
        "largest_signal_episode_fraction": 0.05,
        "largest_target_cluster_fraction": 0.05,
        "largest_causal_segment_fraction": 0.2,
        "largest_day_fraction": 0.1,
        "largest_component_fraction": 0.4,
        "largest_regime_fraction": 0.5,
        "largest_fallback_tier_fraction": 0.7,
    }
    base.update(overrides)
    return base


def test_concentration_within_limits() -> None:
    report = robustness.evaluate_concentration(
        concentration_limits=LIMITS, concentration_evidence=concentration(),
        accepted_rows=1000)
    assert report["status"] == "within_limits"
    assert report["breach_count"] == 0


def test_one_dominant_episode_is_detected() -> None:
    report = robustness.evaluate_concentration(
        concentration_limits=LIMITS,
        concentration_evidence=concentration(largest_signal_episode_fraction=0.6),
        accepted_rows=1000)
    assert report["status"] == "dangerously_concentrated"
    breached = [check for check in report["checks"] if check["breached"]]
    assert breached[0]["dimension"] == "largest_signal_episode_fraction"


def test_one_dominant_segment_is_detected() -> None:
    report = robustness.evaluate_concentration(
        concentration_limits=LIMITS,
        concentration_evidence=concentration(largest_causal_segment_fraction=0.9),
        accepted_rows=1000)
    assert report["status"] == "dangerously_concentrated"


def test_small_sample_concentration_is_inconclusive_not_damning() -> None:
    report = robustness.evaluate_concentration(
        concentration_limits=LIMITS,
        concentration_evidence=concentration(largest_signal_episode_fraction=0.6),
        accepted_rows=50)
    assert report["status"] == "inconclusive_small_sample"
    assert report["small_sample"] is True

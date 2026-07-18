from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import simulate_integrator as integrator


def synthetic_inputs() -> tuple[np.ndarray, np.ndarray, list[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    times = np.arange(9, dtype=np.int64) * integrator.DT_NOM_NS
    actual_x = np.column_stack((
        0.001 * np.arange(9),
        -0.0005 * np.arange(9),
        0.00025 * np.arange(9),
    ))
    values = np.array([[1.0, 1.0e-7, 0.015, 0.0, 0.0, 20_000.0]])
    dynamics = [{"times": np.array([0], dtype=np.int64), "values": values.copy()} for _ in integrator.PAIRS]
    coupling = {"times": np.array([0], dtype=np.int64), "matrices": np.zeros((1, 3, 3))}
    return times, actual_x, dynamics, coupling


def assert_same_final_state(left: dict[str, object], right: dict[str, object]) -> None:
    assert left["final_state"]["timestamp"] == right["final_state"]["timestamp"]
    np.testing.assert_allclose(left["final_state"]["x_hat"], right["final_state"]["x_hat"], atol=1e-15)
    np.testing.assert_allclose(left["final_state"]["v_hat"], right["final_state"]["v_hat"], atol=1e-15)


def test_checkpoint_resume_matches_uninterrupted_replay(tmp_path: Path) -> None:
    times, actual_x, dynamics, coupling = synthetic_inputs()
    full_checkpoint = tmp_path / "full.json"
    _daily, _gaps, uninterrupted = integrator.replay(
        times, actual_x, dynamics, coupling, max_steps=100, checkpoint=full_checkpoint, resume=False
    )

    checkpoint = tmp_path / "resume.json"
    _daily, _gaps, partial = integrator.replay(
        times, actual_x, dynamics, coupling, max_steps=2, checkpoint=checkpoint, resume=False
    )
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["state_index"] + 1 == payload["next_arrival_index"]
    assert payload["state_timestamp"] == integrator.utc_iso(int(times[payload["state_index"]]))

    _daily, _gaps, resumed = integrator.replay(
        times, actual_x, dynamics, coupling, max_steps=100, checkpoint=checkpoint, resume=True
    )
    assert partial["arrivals_processed"] == 2
    assert resumed["arrivals_processed"] == uninterrupted["arrivals_processed"] - partial["arrivals_processed"]
    assert_same_final_state(uninterrupted, resumed)


def test_legacy_checkpoint_uses_its_next_row_without_skipping(tmp_path: Path) -> None:
    times, actual_x, dynamics, coupling = synthetic_inputs()
    checkpoint = tmp_path / "legacy-source.json"
    _daily, _gaps, _partial = integrator.replay(
        times, actual_x, dynamics, coupling, max_steps=2, checkpoint=checkpoint, resume=False
    )
    current = json.loads(checkpoint.read_text(encoding="utf-8"))
    legacy = {
        "version": "integrator-1.1.0",
        "pair_scope": list(integrator.PAIRS),
        "next_row_index": current["next_arrival_index"],
        "last_timestamp": current["state_timestamp"],
        "x_hat": current["x_hat"],
        "v_hat": current["v_hat"],
    }
    checkpoint.write_text(json.dumps(legacy), encoding="utf-8")
    _daily, _gaps, resumed = integrator.replay(
        times, actual_x, dynamics, coupling, max_steps=100, checkpoint=checkpoint, resume=True
    )
    full_checkpoint = tmp_path / "full.json"
    _daily, _gaps, uninterrupted = integrator.replay(
        times, actual_x, dynamics, coupling, max_steps=100, checkpoint=full_checkpoint, resume=False
    )
    assert_same_final_state(uninterrupted, resumed)


def test_nonnormal_transient_growth_rejects_spectral_radius_only_acceptance() -> None:
    kappa = np.array([3.32340637e-08, 3.61581926e-07, 2.04387235e-07])
    gamma = np.array([0.00104061, 0.00692137, 0.082814])
    coupling = np.array([
        [0.0, 2.10871775e-05, -7.11724231e-05],
        [0.0, 0.0, 4.27134699e-05],
        [0.0, 0.0, 0.0],
    ])
    metrics = integrator.amplification_metrics(kappa, gamma, coupling, integrator.DT_NOM_S)
    assert metrics["rho"] < 1.0
    assert metrics["sigma_max"] > 1.0
    assert metrics["transient_growth_max"] > integrator.TRANSIENT_GROWTH_LIMIT
    assert metrics["eigenvector_condition_number"] > 1.0


def test_sampled_dt_report_is_not_a_monotonic_bisection_claim() -> None:
    kappa = np.array([1.0e-7, 2.0e-7, 1.5e-7])
    gamma = np.array([0.015, 0.016, 0.017])
    report = integrator.sampled_dt_stability(kappa, gamma, np.zeros((3, 3)))
    assert report["dt_grid_samples"] == len(integrator.STABILITY_DT_GRID_S)
    assert report["dt_stable_interval_count"] >= 1
    assert report["dt_60s_spectral_pass"] is True

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import stat_arb


FAST_CONFIG = stat_arb.ArenaConfig(
    warmup_steps=64,
    bootstrap_samples=32,
    bootstrap_block_sensitivity_steps=(8, 16, 32),
    low_regime=stat_arb.RegimeParameters(32.0, 32.0, 8, 0.94, 0.50, 8, 2.0, 2),
    high_regime=stat_arb.RegimeParameters(16.0, 16.0, 4, 0.82, 0.75, 6, 1.35, 1),
)


def test_identity_free_transform_removes_exact_triangle_returns() -> None:
    transform, inverse, labels = stat_arb.identity_free_transform()
    raw_return = np.zeros(len(stat_arb.PAIRS))
    raw_return[stat_arb.PAIRS.index("EURUSD")] = 0.0012
    raw_return[stat_arb.PAIRS.index("GBPUSD")] = -0.0004
    raw_return[stat_arb.PAIRS.index("EURGBP")] = 0.0016
    raw_return[stat_arb.PAIRS.index("USDJPY")] = -0.0007
    raw_return[stat_arb.PAIRS.index("EURJPY")] = 0.0005
    raw_return[stat_arb.PAIRS.index("GBPJPY")] = -0.0011
    transformed = transform @ raw_return
    assert abs(transformed[labels.index("EURGBP_triangle_residual")]) < 1e-15
    assert abs(transformed[labels.index("EURJPY_triangle_residual")]) < 1e-15
    assert abs(transformed[labels.index("GBPJPY_triangle_residual")]) < 1e-15
    np.testing.assert_allclose(inverse @ transformed, raw_return, atol=1e-15)


def test_triangle_signal_dual_mapping_preserves_return_functional() -> None:
    transform, _inverse, labels = stat_arb.identity_free_transform()
    rng = np.random.default_rng(7)
    for label in ("EURGBP_triangle_residual", "EURJPY_triangle_residual", "GBPJPY_triangle_residual"):
        signal = np.zeros(len(stat_arb.PAIRS))
        signal[labels.index(label)] = -1.75
        raw_weights = stat_arb.basis_signal_to_raw_weights(signal, transform)
        for _ in range(20):
            raw_return = rng.normal(0.0, 0.002, len(stat_arb.PAIRS))
            assert np.isclose(signal @ (transform @ raw_return), raw_weights @ raw_return, atol=1e-14)


def test_factor_neutral_weights_are_currency_and_selected_factor_neutral() -> None:
    transform, inverse, labels = stat_arb.identity_free_transform()
    signal = np.zeros(len(stat_arb.PAIRS))
    signal[labels.index("EURGBP_triangle_residual")] = 1.0
    # Factor directions on bridge pairs leave the structural triangle cycle
    # feasible while still exercising the inverse(T) primal-direction mapping.
    raw_loadings = np.zeros((len(stat_arb.PAIRS), 3))
    raw_loadings[3, 0] = 1.0
    raw_loadings[4, 1] = 1.0
    raw_loadings[5, 2] = 1.0
    loadings = transform @ raw_loadings
    incidence, _currencies = stat_arb.currency_incidence()
    weights = stat_arb.factor_neutral_weights(signal, transform, inverse, loadings, incidence, 2)
    mapped_raw_loadings = inverse @ loadings
    np.testing.assert_allclose(incidence.T @ weights, np.zeros(incidence.shape[1]), atol=1e-12)
    np.testing.assert_allclose(mapped_raw_loadings[:, :2].T @ weights, np.zeros(2), atol=1e-12)
    assert np.isclose(np.sum(np.abs(weights)), 1.0)


def test_circular_bootstrap_has_exact_sample_length_and_never_indexes_outside_input() -> None:
    segment_ids = np.asarray([0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
    indices = stat_arb.circular_block_resample_indices(segment_ids, 3, np.random.default_rng(3))
    assert len(indices) == len(segment_ids)
    assert (indices >= 0).all() and (indices < len(segment_ids)).all()


def test_frozen_target_uses_entry_model_and_reports_path_diagnostics() -> None:
    times = np.arange(5, dtype=np.int64) * stat_arb.DT_NS
    log_prices = np.zeros((5, len(stat_arb.PAIRS)))
    transform, _inverse, _labels = stat_arb.identity_free_transform()
    entry = stat_arb.FrozenResidualTarget(
        source_index=1,
        segment_id=0,
        selected=0,
        entry_level=2.0,
        mean_basis=np.zeros(len(stat_arb.PAIRS)),
        loadings_basis=np.zeros((len(stat_arb.PAIRS), 3)),
        residual_scale=np.ones(len(stat_arb.PAIRS)),
        level_ar=0.9,
        horizon_steps=2,
        stop_multiple=2.0,
        raw_basket_weights=np.zeros(len(stat_arb.PAIRS)),
    )
    outcome = stat_arb.evaluate_frozen_residual_target(times, log_prices, transform, entry)
    assert outcome is not None
    assert outcome["convergence_label"] == 1
    assert np.isclose(outcome["frozen_target_level"], 2.0 * 0.9 * 0.9)
    assert outcome["gross_convergence"] > 0.0
    assert outcome["percentage_displacement_removed"] > 0.0
    assert outcome["frozen_basket_cumulative_log_return"] == 0.0


def test_causal_emissions_ignore_future_price_mutations() -> None:
    times, log_prices = stat_arb.synthetic_input()
    baseline = stat_arb.run_arrays(times, log_prices, test_start_index=500, config=FAST_CONFIG)
    changed = log_prices.copy()
    changed[700:] += 0.1
    altered = stat_arb.run_arrays(times, changed, test_start_index=500, config=FAST_CONFIG)
    left = baseline.emissions[baseline.emissions["source_index"] < 680].set_index("source_index")
    right = altered.emissions[altered.emissions["source_index"] < 680].set_index("source_index")
    common = left.index.intersection(right.index)
    np.testing.assert_array_equal(left.loc[common, "p_convergence"], right.loc[common, "p_convergence"])
    np.testing.assert_array_equal(left.loc[common, "breakdown_probability"], right.loc[common, "breakdown_probability"])


def test_arena_uses_frozen_level_targets_and_never_promotes_bid_only_data() -> None:
    times, log_prices = stat_arb.synthetic_input()
    result = stat_arb.run_arrays(times, log_prices, test_start_index=500, config=FAST_CONFIG)
    labelled = result.emissions[result.emissions["convergence_label"].notna()]
    assert not labelled.empty
    assert (labelled["target_time"] > labelled["timestamp"]).all()
    assert labelled["gross_convergence"].notna().all()
    assert labelled["target_path_volatility"].notna().all()
    currency_exposures = result.emissions.filter(regex=r"^currency_incidence_exposure_").to_numpy(dtype=float)
    assert float(np.nanmax(np.abs(currency_exposures))) < 1e-8
    selected_factor_exposures = result.emissions.filter(regex=r"^selected_factor_exposure_").to_numpy(dtype=float)
    assert float(np.nanmax(np.abs(selected_factor_exposures))) < 1e-8
    assert set(result.emissions["active_regime"]) == {"low", "high"}
    assert result.emissions["holding_horizon_steps"].nunique() == 2
    assert (result.emissions["selected_graph_pressure"] > 0.0).any()
    assert result.summary["components"]["execution_cost_status"].startswith("BLOCKED")
    assert result.summary["promotion_status"].startswith("REJECTED")

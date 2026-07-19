from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


from engine.models.statistical import stat_arb


FAST_CONFIG = stat_arb.ArenaConfig(
    warmup_steps=64,
    bootstrap_samples=32,
    bootstrap_block_sensitivity_minutes=(8, 16, 32),
    basket_mode=stat_arb.RELATIVE_VALUE_MODE,
    relative_value_currency_exposure_budget=0.35,
    basket_neutral_zone_z=0.05,
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
    weights = stat_arb.factor_neutral_weights(signal, transform, inverse, loadings, incidence, 2).weights
    mapped_raw_loadings = inverse @ loadings
    np.testing.assert_allclose(incidence.T @ weights, np.zeros(incidence.shape[1]), atol=1e-12)
    np.testing.assert_allclose(mapped_raw_loadings[:, :2].T @ weights, np.zeros(2), atol=1e-12)
    assert np.isclose(np.sum(np.abs(weights)), 1.0)


def test_cycle_and_relative_value_modes_have_distinct_currency_contracts() -> None:
    transform, inverse, _labels = stat_arb.identity_free_transform()
    signal = np.zeros(len(stat_arb.PAIRS))
    signal[stat_arb.PAIRS.index("USDCNH")] = 1.0  # a bridge, not a closed cycle
    loadings = np.zeros((len(stat_arb.PAIRS), 3))
    incidence, _currencies = stat_arb.currency_incidence()
    cycle = stat_arb.factor_neutral_weights(signal, transform, inverse, loadings, incidence, 0,
                                             stat_arb.CYCLE_NEUTRAL_MODE).weights
    relative_result = stat_arb.factor_neutral_weights(signal, transform, inverse, loadings, incidence, 0,
                                                       stat_arb.RELATIVE_VALUE_MODE, 0.25)
    relative = relative_result.weights
    np.testing.assert_allclose(cycle, np.zeros(len(stat_arb.PAIRS)), atol=1e-12)
    assert np.sum(np.abs(relative)) > 0.0
    assert float(np.max(np.abs(incidence.T @ relative))) <= 0.25 + 1e-12
    assert relative_result.converged
    assert relative_result.max_weight_violation <= 1e-8


def test_post_construction_probability_and_gate_reject_erased_signal() -> None:
    parameters = stat_arb.RegimeParameters(16.0, 16.0, 4, 0.95, 0.5, 8, 2.0, 1)
    features = {
        "residual_level": np.asarray([2.0] + [0.0] * (len(stat_arb.PAIRS) - 1)),
        "graph_pressure": np.zeros(len(stat_arb.PAIRS)),
        "graph_cluster_size": np.ones(len(stat_arb.PAIRS)),
        "beta": np.full(len(stat_arb.PAIRS), 0.5),
        "half_life": np.full(len(stat_arb.PAIRS), 10.0),
        "parameters": parameters,
        "regime_high_probability": 0.2,
    }
    selection = stat_arb.selection_from_features(features)
    config = stat_arb.ArenaConfig(maximum_basket_concentration=1.0)
    retained = stat_arb.prediction_from_constructed_basket(
        features, selection, np.asarray([0.5, -0.5] + [0.0] * 8), 0.0, 1.0, config)
    erased = stat_arb.prediction_from_constructed_basket(
        features, selection, np.zeros(len(stat_arb.PAIRS)), 1.0, 0.0, config)
    assert retained["entry_eligible"]
    assert not erased["entry_eligible"]
    assert erased["probability"] < retained["probability"]


def test_observed_minute_bootstrap_samples_sparse_entries_from_real_time_blocks() -> None:
    timeline_times = np.arange(40, dtype=np.int64) * stat_arb.DT_NS
    timeline_segments = np.zeros(len(timeline_times), dtype=np.int64)
    # These are deliberately sparse: 10 entry rows are not ten minutes.
    source_indices = np.asarray([1, 2, 3, 21, 22, 23], dtype=np.int64)
    source_segments = np.zeros(len(source_indices), dtype=np.int64)
    indices = stat_arb.observed_minute_block_resample_indices(
        source_indices, source_segments, timeline_times, timeline_segments, 3, np.random.default_rng(3))
    assert len(indices) == len(source_indices)
    assert (indices >= 0).all() and (indices < len(source_indices)).all()
    # Every selected observation is indexed through the raw observed timeline,
    # so the configured unit is minutes rather than dataframe row positions.
    assert np.all(np.diff(timeline_times) == stat_arb.DT_NS)


def test_frozen_target_uses_entry_model_and_reports_path_diagnostics() -> None:
    times = np.arange(5, dtype=np.int64) * stat_arb.DT_NS
    log_prices = np.zeros((5, len(stat_arb.PAIRS)))
    transform, _inverse, _labels = stat_arb.identity_free_transform()
    entry = stat_arb.FrozenResidualTarget(
        source_index=1,
        segment_id=0,
        selected=0,
        entry_level=2.0,
        entry_levels_by_regime=np.asarray([np.full(len(stat_arb.PAIRS), 2.0), np.zeros(len(stat_arb.PAIRS))]),
        regime_probabilities=np.asarray([1.0, 0.0]),
        means_by_regime=np.zeros((2, len(stat_arb.PAIRS))),
        loadings_by_regime=np.zeros((2, len(stat_arb.PAIRS), 3)),
        residual_scales_by_regime=np.ones((2, len(stat_arb.PAIRS))),
        level_ars=np.asarray([0.9, 0.8]),
        horizon_steps=2,
        stop_multiple=2.0,
        raw_basket_weights=np.zeros(len(stat_arb.PAIRS)),
        basket_one_step_volatility=1.0,
        basket_neutral_zone_z=0.25,
    )
    outcome = stat_arb.evaluate_frozen_residual_target(times, log_prices, transform, entry)
    assert outcome is not None
    assert outcome["convergence_label"] == 1
    assert np.isclose(outcome["frozen_target_level"], 2.0 * 0.9 * 0.9)
    assert outcome["gross_convergence"] > 0.0
    assert outcome["percentage_displacement_removed"] > 0.0
    assert outcome["frozen_basket_cumulative_log_return"] == 0.0
    assert outcome["basket_directional_label"] == 0
    assert np.isnan(outcome["basket_binary_label"])
    assert np.isnan(outcome["basket_residual_convergence_disagreement"])


def test_separate_regime_levels_do_not_jump_on_a_pure_regime_label_flip() -> None:
    same_level_regime = stat_arb.RegimeParameters(16.0, 16.0, 4, 0.95, 0.50, 8, 2.0, 1)
    config = stat_arb.ArenaConfig(low_regime=same_level_regime, high_regime=same_level_regime)
    transform, _inverse, _labels = stat_arb.identity_free_transform()
    state = stat_arb.CausalFactorState(config, transform)
    state.residual_levels = [np.ones(len(stat_arb.PAIRS)), np.ones(len(stat_arb.PAIRS))]
    # Hold the volatility update fixed so only the active-regime label changes.
    state._update_regime_probability = lambda _raw: 0.0  # type: ignore[method-assign]
    state.regime_high_probability = 0.0
    low_features, _ = state.update(np.zeros(len(stat_arb.PAIRS)))
    state.residual_levels = [np.ones(len(stat_arb.PAIRS)), np.ones(len(stat_arb.PAIRS))]
    state.regime_high_probability = 1.0
    high_features, _ = state.update(np.zeros(len(stat_arb.PAIRS)))
    np.testing.assert_allclose(low_features["residual_level"], high_features["residual_level"], atol=1e-12)
    np.testing.assert_allclose(
        high_features["residual_level"],
        high_features["regime_probabilities"] @ high_features["residual_levels_by_regime"],
        atol=1e-12,
    )


def test_regime_mixture_uses_full_innovation_once_and_decomposes_level_change() -> None:
    parameters = stat_arb.RegimeParameters(16.0, 16.0, 99, 0.95, 0.5, 8, 2.0, 1)
    config = stat_arb.ArenaConfig(low_regime=parameters, high_regime=parameters)
    transform, _inverse, _labels = stat_arb.identity_free_transform()
    state = stat_arb.CausalFactorState(config, transform)
    state.steps = 1  # Preserve the deliberately zero factor bases below.
    state.loadings_by_regime = [np.zeros((len(stat_arb.PAIRS), 3)), np.zeros((len(stat_arb.PAIRS), 3))]
    state._update_regime_probability = lambda _raw: 0.0  # type: ignore[method-assign]
    state.regime_high_probability = 0.5
    raw_return = np.zeros(len(stat_arb.PAIRS))
    raw_return[0] = 0.001
    features, _ = state.update(raw_return)
    # With zero initial state, each conditional level receives its full
    # innovation, not half an innovation before the posterior blend.
    np.testing.assert_allclose(features["residual_levels_by_regime"][0], features["residual_innovation"])
    np.testing.assert_allclose(
        features["residual_level_change"],
        features["residual_level_decay_effect"] + features["residual_level_innovation_effect"]
        + features["residual_level_posterior_reweighting_effect"],
        atol=1e-12,
    )


def test_regime_posterior_reweighting_is_explicit_without_new_innovation() -> None:
    parameters = stat_arb.RegimeParameters(16.0, 16.0, 99, 0.999, 0.5, 8, 2.0, 1)
    config = stat_arb.ArenaConfig(low_regime=parameters, high_regime=parameters)
    transform, _inverse, _labels = stat_arb.identity_free_transform()
    state = stat_arb.CausalFactorState(config, transform)
    state.steps = 1
    state.loadings_by_regime = [np.zeros((len(stat_arb.PAIRS), 3)), np.zeros((len(stat_arb.PAIRS), 3))]
    state.residual_levels = [np.full(len(stat_arb.PAIRS), 2.5), np.full(len(stat_arb.PAIRS), -0.8)]
    state.previous_regime_probabilities = np.asarray([0.8, 0.2])
    state.regime_high_probability = 0.8
    state._update_regime_probability = lambda _raw: 0.0  # type: ignore[method-assign]
    features, _ = state.update(np.zeros(len(stat_arb.PAIRS)))
    # There is no raw innovation; the movement is almost entirely the declared
    # posterior reweighting from low (+2.5) to high (-0.8), plus AR decay.
    np.testing.assert_allclose(features["residual_level_innovation_effect"], 0.0, atol=1e-12)
    assert float(features["residual_level_posterior_reweighting_effect"][0]) < -1.9
    np.testing.assert_allclose(
        features["residual_level_change"],
        features["residual_level_decay_effect"] + features["residual_level_innovation_effect"]
        + features["residual_level_posterior_reweighting_effect"],
        atol=1e-12,
    )


def test_frozen_target_standardizes_and_excludes_neutral_returns() -> None:
    times = np.arange(5, dtype=np.int64) * stat_arb.DT_NS
    log_prices = np.zeros((5, len(stat_arb.PAIRS)))
    log_prices[2:, 0] = 0.01
    log_prices[3:, 0] = 0.02
    transform, _inverse, _labels = stat_arb.identity_free_transform()
    entry = stat_arb.FrozenResidualTarget(
        source_index=1,
        segment_id=0,
        selected=0,
        entry_level=1.0,
        entry_levels_by_regime=np.zeros((2, len(stat_arb.PAIRS))),
        regime_probabilities=np.asarray([0.5, 0.5]),
        means_by_regime=np.zeros((2, len(stat_arb.PAIRS))),
        loadings_by_regime=np.zeros((2, len(stat_arb.PAIRS), 3)),
        residual_scales_by_regime=np.ones((2, len(stat_arb.PAIRS))),
        level_ars=np.asarray([0.9, 0.9]),
        horizon_steps=2,
        stop_multiple=2.0,
        raw_basket_weights=np.eye(1, len(stat_arb.PAIRS), 0).ravel(),
        basket_one_step_volatility=0.01,
        basket_neutral_zone_z=0.25,
    )
    outcome = stat_arb.evaluate_frozen_residual_target(times, log_prices, transform, entry)
    assert outcome is not None
    assert outcome["basket_standardized_return"] == pytest.approx(2.0 ** -0.5 * 2.0)
    assert outcome["basket_outcome_class"] == 1
    assert outcome["basket_binary_label"] == 1.0

    negative_prices = np.zeros_like(log_prices)
    negative_prices[2:, 0] = -0.01
    negative_prices[3:, 0] = -0.02
    negative = stat_arb.evaluate_frozen_residual_target(times, negative_prices, transform, entry)
    assert negative is not None
    assert negative["basket_outcome_class"] == -1
    assert negative["basket_binary_label"] == 0.0


def test_causal_emissions_ignore_future_price_mutations() -> None:
    times, log_prices = stat_arb.synthetic_input()
    baseline = stat_arb.run_arrays(times, log_prices, test_start_index=500, config=FAST_CONFIG)
    changed = log_prices.copy()
    changed[700:] += 0.1
    altered = stat_arb.run_arrays(times, changed, test_start_index=500, config=FAST_CONFIG)
    left = baseline.emissions[baseline.emissions["source_index"] < 680].set_index("source_index")
    right = altered.emissions[altered.emissions["source_index"] < 680].set_index("source_index")
    common = left.index.intersection(right.index)
    np.testing.assert_array_equal(left.loc[common, "p_basket_directional"], right.loc[common, "p_basket_directional"])
    np.testing.assert_array_equal(left.loc[common, "breakdown_probability"], right.loc[common, "breakdown_probability"])


def test_arena_uses_frozen_level_targets_and_never_promotes_bid_only_data() -> None:
    times, log_prices = stat_arb.synthetic_input()
    result = stat_arb.run_arrays(times, log_prices, test_start_index=500, config=FAST_CONFIG)
    labelled = result.emissions[result.emissions["convergence_label"].notna()]
    assert not labelled.empty
    assert (labelled["target_time"] > labelled["timestamp"]).all()
    assert labelled["gross_convergence"].notna().all()
    assert labelled["target_path_volatility"].notna().all()
    assert labelled["basket_directional_label"].notna().all()
    assert labelled["basket_residual_convergence_disagreement"].dropna().isin([0, 1]).all()
    assert labelled["projection_distortion_l2"].notna().all()
    currency_exposures = result.emissions.filter(regex=r"^currency_incidence_exposure_").to_numpy(dtype=float)
    assert float(np.nanmax(np.abs(currency_exposures))) <= FAST_CONFIG.relative_value_currency_exposure_budget + 1e-8
    selected_factor_exposures = result.emissions.filter(regex=r"^selected_factor_exposure_").to_numpy(dtype=float)
    assert float(np.nanmax(np.abs(selected_factor_exposures))) < 1e-8
    assert set(result.emissions["active_regime"]) == {"low", "high"}
    assert result.emissions["holding_horizon_steps"].nunique() == 2
    assert (result.emissions["selected_graph_pressure"] > 0.0).any()
    assert result.summary["components"]["execution_cost_status"].startswith("BLOCKED")
    primary_gate = result.summary["evaluation"]["primary_three_class_observed_minute_block_bootstrap"]
    assert primary_gate["exact_replicate_entry_count"] == result.summary["evaluation"]["oos_samples"]
    assert "log_loss_improvement" in primary_gate["sensitivity"]["32_observed_minutes"]
    assert result.summary["promotion_status"].startswith("REJECTED")

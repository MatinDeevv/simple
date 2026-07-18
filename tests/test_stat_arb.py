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
    outcome_horizon_steps=8,
    factor_refresh_steps=8,
    covariance_half_life_steps=32.0,
    residual_half_life_steps=32.0,
    bootstrap_samples=32,
    bootstrap_block_steps=32,
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


def test_factor_neutral_weights_are_neutral_to_current_factor_span() -> None:
    _transform, inverse, _labels = stat_arb.identity_free_transform()
    signal = np.zeros(len(stat_arb.PAIRS))
    signal[5] = 1.0
    loadings = np.eye(len(stat_arb.PAIRS), 3)
    weights = stat_arb.factor_neutral_weights(signal, inverse, loadings)
    raw_loadings = inverse @ loadings
    np.testing.assert_allclose(raw_loadings.T @ weights, np.zeros(3), atol=1e-12)
    assert abs(float(np.sum(weights))) < 1e-12
    assert np.isclose(np.sum(np.abs(weights)), 1.0)


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
    np.testing.assert_array_equal(
        left.loc[common, "breakdown_probability"], right.loc[common, "breakdown_probability"])


def test_arena_requires_contiguous_targets_and_never_promotes_bid_only_data() -> None:
    times, log_prices = stat_arb.synthetic_input()
    result = stat_arb.run_arrays(times, log_prices, test_start_index=500, config=FAST_CONFIG)
    labelled = result.emissions[result.emissions["convergence_label"].notna()]
    assert not labelled.empty
    assert (labelled["target_time"] > labelled["timestamp"]).all()
    assert result.summary["components"]["execution_cost_status"].startswith("BLOCKED")
    assert result.summary["promotion_status"].startswith("REJECTED")

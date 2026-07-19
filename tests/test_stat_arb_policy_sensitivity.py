from __future__ import annotations

import numpy as np
import pytest

from engine.models.statistical import stat_arb


def test_multiclass_bootstrap_rejects_invalid_contracts() -> None:
    matrix = np.tile(np.array([[1.0, 0.0, 0.0]]), (2, 1))
    kwargs = dict(model_probabilities=matrix, baseline_probabilities=matrix, labels=np.array([0, 0]),
                  source_indices=np.array([1, 2]), source_segment_ids=np.array([0, 0]),
                  timeline_times=np.arange(4, dtype=np.int64) * stat_arb.DT_NS,
                  timeline_segment_ids=np.zeros(4, dtype=np.int64), block_minutes=1, samples=2, seed=1)
    with pytest.raises(stat_arb.ContractError, match="samples"):
        stat_arb.observed_minute_block_multiclass_bootstrap_comparison(**(kwargs | {"samples": 0}))
    with pytest.raises(stat_arb.ContractError, match="duplicate"):
        stat_arb.observed_minute_block_multiclass_bootstrap_comparison(**(kwargs | {"source_indices": np.array([1, 1])}))


def test_policy_order_and_boundary_contracts_are_explicit() -> None:
    times, prices = stat_arb.synthetic_input(900)
    result = stat_arb.run_arrays(times, prices, 500, stat_arb._fast_config())
    views = result.summary["evaluation"]["entry_policy_sensitivities"]
    assert list(views) == list(stat_arb.ENTRY_POLICY_SENSITIVITY_ORDER)
    for policy, view in views.items():
        assert view["boundary_contract"] == "reset_at_oos_start"
        assert set(view["boundary_contracts"]) == {"reset_at_oos_start", "carry_chronological_state_into_oos"}
        assert view["conditional_comparator_identity"] == "three_class_conditional_climatology"
        assert view["frozen_training_rows"] == result.summary["evaluation"]["train_samples"]
        assert view["oos_rows"] == result.summary["evaluation"]["oos_samples"]
        assert view["frozen_training_rows"] + view["boundary_crossing_rows"] + view["oos_rows"] == int(result.emissions["basket_class"].notna().sum())


def test_policy_only_relabels_bootstrap_support_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    times, prices = stat_arb.synthetic_input(900)

    def malformed(*_args, **_kwargs):
        raise stat_arb.ContractError("malformed probabilities")

    monkeypatch.setattr(stat_arb, "observed_minute_block_multiclass_bootstrap_comparison", malformed)
    with pytest.raises(stat_arb.ContractError, match="malformed probabilities"):
        stat_arb.run_arrays(times, prices, 500, stat_arb._fast_config())

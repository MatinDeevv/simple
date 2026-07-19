from __future__ import annotations

import time

import pandas as pd
import pytest

from engine.models.statistical import stat_arb


def test_vectorized_validator_catches_invalid_tail_without_iterrows(monkeypatch: pytest.MonkeyPatch) -> None:
    times, prices = stat_arb.synthetic_input(900)
    result = stat_arb.run_arrays(times, prices, 500, stat_arb._fast_config())
    large = pd.concat([result.emissions] * 130, ignore_index=True)
    large["source_index"] = range(len(large))
    large.iloc[-1, large.columns.get_loc("p_basket_positive")] = 1.5
    started = time.perf_counter()
    with pytest.raises(stat_arb.ContractError, match="probability"):
        stat_arb._validate_frame_rows(large, "stat-arb-emission.schema.json")
    assert time.perf_counter() - started < 10.0

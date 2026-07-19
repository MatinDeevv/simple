"""Repository-wide determinism, no-input-mutation, causality, and numerical-
safety property tests.

These are black-box tests against public module APIs only (``run_arrays``,
``identity_free_transform``, ``synthetic_input``, ``load_schema``,
``run_event_study``, ...). They never edit the production modules directly,
and they intentionally use a different testing
strategy (randomized property checks, subprocess/process-boundary
determinism, explicit input-array-identity checks) than those modules' own
test suites, so they add independent verification rather than duplicating
existing assertions.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
from engine.evaluation import evaluation_protocol as ep
from engine.models.events import legal_event
from engine.models.statistical import stat_arb


# --------------------------------------------------------------------------- #
# Part 11: FX identity-free transform algebra (randomized property checks)
# --------------------------------------------------------------------------- #

def test_transform_is_a_true_inverse_pair_for_randomized_returns() -> None:
    transform, inverse, _labels = stat_arb.identity_free_transform()
    rng = np.random.default_rng(2026)
    for _ in range(200):
        raw_return = rng.normal(0.0, 1.0e-3, size=stat_arb.N_PAIRS)
        recovered = inverse @ (transform @ raw_return)
        np.testing.assert_allclose(recovered, raw_return, atol=1e-10)


def test_dual_signal_transform_matches_primal_transform_for_randomized_signals() -> None:
    """s @ (T @ r) == (T.T @ s) @ r -- a portfolio-return identity that must hold
    for a signal vector expressed in the identity-free basis regardless of the
    specific raw return realization."""
    transform, _inverse, _labels = stat_arb.identity_free_transform()
    rng = np.random.default_rng(99)
    for _ in range(200):
        raw_return = rng.normal(0.0, 1.0e-3, size=stat_arb.N_PAIRS)
        signal = rng.normal(0.0, 1.0, size=stat_arb.N_PAIRS)
        lhs = signal @ (transform @ raw_return)
        rhs = (transform.T @ signal) @ raw_return
        assert lhs == pytest.approx(rhs, abs=1e-9)


def test_triangle_residual_channels_are_exactly_zero_for_consistent_triangles() -> None:
    transform, _inverse, labels = stat_arb.identity_free_transform()
    rng = np.random.default_rng(4)
    for _ in range(50):
        raw_return = rng.normal(0.0, 1.0e-3, size=stat_arb.N_PAIRS)
        raw_return[stat_arb.PAIRS.index("EURGBP")] = (
            raw_return[stat_arb.PAIRS.index("EURUSD")] - raw_return[stat_arb.PAIRS.index("GBPUSD")])
        raw_return[stat_arb.PAIRS.index("EURJPY")] = (
            raw_return[stat_arb.PAIRS.index("EURUSD")] + raw_return[stat_arb.PAIRS.index("USDJPY")])
        raw_return[stat_arb.PAIRS.index("GBPJPY")] = (
            raw_return[stat_arb.PAIRS.index("GBPUSD")] + raw_return[stat_arb.PAIRS.index("USDJPY")])
        transformed = transform @ raw_return
        for name in ("EURGBP_triangle_residual", "EURJPY_triangle_residual", "GBPJPY_triangle_residual"):
            assert abs(transformed[labels.index(name)]) < 1e-10


# --------------------------------------------------------------------------- #
# Part 8: determinism across repeated runs, processes, cwd, and hash seed
# --------------------------------------------------------------------------- #

def _run_fast_arena():
    times, log_prices = stat_arb.synthetic_input()
    return stat_arb.run_arrays(times, log_prices, test_start_index=500, config=stat_arb._fast_config())


def test_same_process_repeated_run_is_bitwise_identical() -> None:
    first = _run_fast_arena()
    second = _run_fast_arena()
    pd.testing.assert_frame_equal(first.emissions, second.emissions)
    pd.testing.assert_frame_equal(first.graph, second.graph)
    assert json.dumps(first.summary, sort_keys=True, default=str) == \
           json.dumps(second.summary, sort_keys=True, default=str)


def test_run_arrays_does_not_mutate_caller_owned_input_arrays() -> None:
    times, log_prices = stat_arb.synthetic_input()
    times_before = times.copy()
    prices_before = log_prices.copy()
    stat_arb.run_arrays(times, log_prices, test_start_index=500, config=stat_arb._fast_config())
    np.testing.assert_array_equal(times, times_before)
    np.testing.assert_array_equal(log_prices, prices_before)


def test_input_row_copy_versus_original_array_gives_the_same_result() -> None:
    times, log_prices = stat_arb.synthetic_input()
    config = stat_arb._fast_config()
    original_result = stat_arb.run_arrays(times, log_prices, test_start_index=500, config=config)
    copied_result = stat_arb.run_arrays(times.copy(), log_prices.copy(), test_start_index=500, config=config)
    pd.testing.assert_frame_equal(original_result.emissions, copied_result.emissions)


def test_mutating_prices_after_a_boundary_never_changes_emissions_at_or_before_it() -> None:
    times, log_prices = stat_arb.synthetic_input()
    config = stat_arb._fast_config()
    baseline = stat_arb.run_arrays(times, log_prices, test_start_index=500, config=config)
    boundary = 700
    mutated_prices = log_prices.copy()
    mutated_prices[boundary:] += 0.05
    mutated = stat_arb.run_arrays(times, mutated_prices, test_start_index=500, config=config)
    left = baseline.emissions[baseline.emissions["source_index"] < boundary - 30].set_index("source_index")
    right = mutated.emissions[mutated.emissions["source_index"] < boundary - 30].set_index("source_index")
    common = left.index.intersection(right.index)
    assert len(common) > 50
    np.testing.assert_array_equal(left.loc[common, "p_basket_directional"], right.loc[common, "p_basket_directional"])


_SUBPROCESS_SNIPPET = (
    "import sys, json; sys.path.insert(0, {repo_root!r}); from engine.models.statistical import stat_arb; "
    "times, prices = stat_arb.synthetic_input(); "
    "result = stat_arb.run_arrays(times, prices, test_start_index=500, config=stat_arb._fast_config()); "
    "payload = {{'p': result.emissions['p_basket_directional'].round(12).tolist(), "
    "'selected': result.emissions['selected_component_index'].tolist(), "
    "'summary': result.summary}}; "
    "print(json.dumps(payload, sort_keys=True, default=str))"
)


def _run_subprocess_arena(cwd: Path, env_overrides: dict[str, str] | None = None) -> str:
    import os
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    if env_overrides:
        env.update(env_overrides)
    code = _SUBPROCESS_SNIPPET.format(repo_root=str(ROOT))
    proc = subprocess.run([sys.executable, "-c", code], cwd=cwd, capture_output=True, text=True,
                          env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_fresh_process_repeated_run_is_identical() -> None:
    first = _run_subprocess_arena(ROOT)
    second = _run_subprocess_arena(ROOT)
    assert first == second


def test_run_is_identical_from_a_different_working_directory(tmp_path: Path) -> None:
    reference = _run_subprocess_arena(ROOT)
    from_tmp = _run_subprocess_arena(tmp_path)
    assert reference == from_tmp


def test_run_is_identical_across_different_python_hash_seeds() -> None:
    seed_zero = _run_subprocess_arena(ROOT, {"PYTHONHASHSEED": "0"})
    seed_one = _run_subprocess_arena(ROOT, {"PYTHONHASHSEED": "1"})
    assert seed_zero == seed_one


# --------------------------------------------------------------------------- #
# Part 11: numerical-safety rejection tests
# --------------------------------------------------------------------------- #

def test_run_arrays_rejects_non_increasing_timestamps() -> None:
    times, log_prices = stat_arb.synthetic_input()
    broken_times = times.copy()
    broken_times[10] = broken_times[9]
    with pytest.raises(stat_arb.ContractError):
        stat_arb.run_arrays(broken_times, log_prices, test_start_index=500, config=stat_arb._fast_config())


def test_run_arrays_rejects_non_finite_prices() -> None:
    times, log_prices = stat_arb.synthetic_input()
    broken_prices = log_prices.copy()
    broken_prices[10, 0] = np.nan
    with pytest.raises(stat_arb.ContractError):
        stat_arb.run_arrays(times, broken_prices, test_start_index=500, config=stat_arb._fast_config())


def test_run_arrays_rejects_wrong_shaped_prices() -> None:
    times, log_prices = stat_arb.synthetic_input()
    with pytest.raises(stat_arb.ContractError):
        stat_arb.run_arrays(times, log_prices[:, :-1], test_start_index=500, config=stat_arb._fast_config())


def test_legal_event_load_schema_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    schema = json.loads((ROOT / "engine" / "config" / "legal-event-schema.json").read_text(encoding="utf-8"))
    schema["schema_version"] = "not-a-real-version"
    broken_path = tmp_path / "broken-schema.json"
    broken_path.write_text(json.dumps(schema), encoding="utf-8")
    with pytest.raises(legal_event.ContractError, match="schema_version"):
        legal_event.load_schema(broken_path)


def test_legal_event_load_schema_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(legal_event.ContractError):
        legal_event.load_schema(tmp_path / "does_not_exist.json")


def test_run_event_study_rejects_non_increasing_timestamps() -> None:
    times = np.arange(200, dtype=np.int64) * legal_event.DT_NS
    times[5] = times[4]
    prices = np.cumsum(np.full((200, legal_event.N_PAIRS), 1e-5), axis=0) + 1.0
    with pytest.raises(legal_event.ContractError):
        legal_event.run_event_study(times, prices, [])


def test_run_event_study_rejects_non_finite_prices() -> None:
    times = np.arange(200, dtype=np.int64) * legal_event.DT_NS
    prices = np.cumsum(np.full((200, legal_event.N_PAIRS), 1e-5), axis=0) + 1.0
    prices[5, 0] = np.inf
    with pytest.raises(legal_event.ContractError):
        legal_event.run_event_study(times, prices, [])


# --------------------------------------------------------------------------- #
# evaluation_protocol.py / entry_diagnostics.py: no caller-owned mutation
# --------------------------------------------------------------------------- #

def test_evaluation_protocol_helpers_do_not_mutate_their_inputs() -> None:
    source_index = np.arange(50)
    segment_id = np.zeros(50, dtype=np.int64)
    is_training = source_index < 40
    before_source = source_index.copy()
    before_segment = segment_id.copy()
    before_training = is_training.copy()
    ep.build_evaluation_metadata(source_index, 5, segment_id, is_training, embargo_steps=2)
    np.testing.assert_array_equal(source_index, before_source)
    np.testing.assert_array_equal(segment_id, before_segment)
    np.testing.assert_array_equal(is_training, before_training)

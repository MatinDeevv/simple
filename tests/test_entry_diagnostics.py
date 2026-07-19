from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.evaluation import entry_diagnostics as ed
from engine.evaluation import evaluation_protocol as ep


def _frame(source_index, segment_id, component, eligible, horizon, weights_a=None, weights_b=None) -> pd.DataFrame:
    n = len(source_index)
    data = {
        "source_index": np.asarray(source_index, dtype=np.int64),
        "segment_id": np.asarray(segment_id, dtype=np.int64),
        "selected_component_index": np.asarray(component, dtype=np.int64),
        "entry_eligible": np.asarray(eligible, dtype=bool),
        "holding_horizon_steps": np.full(n, horizon, dtype=np.int64) if np.isscalar(horizon) else np.asarray(horizon),
    }
    if weights_a is not None:
        data["diagnostic_weight_a"] = np.asarray(weights_a, dtype=np.float64)
        data["diagnostic_weight_b"] = np.asarray(weights_b if weights_b is not None else weights_a, dtype=np.float64)
    return pd.DataFrame(data)


def test_missing_required_columns_raises() -> None:
    with pytest.raises(ed.EntryDiagnosticsError):
        ed.annotate_entry_diagnostics(pd.DataFrame({"source_index": [0]}))


def test_unknown_policy_raises() -> None:
    frame = _frame([0], [0], [0], [True], 1)
    with pytest.raises(ed.EntryDiagnosticsError):
        ed.compute_accepted_mask(frame, "not_a_real_policy")


def test_independent_research_entries_accepts_every_eligible_row() -> None:
    frame = _frame([0, 1, 2], [0, 0, 0], [0, 0, 0], [True, True, False], 5)
    accepted = ed.compute_accepted_mask(frame, "independent_research_entries")
    np.testing.assert_array_equal(accepted, [True, True, False])


def test_non_overlapping_global_keeps_only_the_first_of_overlapping_rows() -> None:
    frame = _frame([0, 1, 2], [0, 0, 0], [0, 1, 0], [True, True, True], 10)
    accepted = ed.compute_accepted_mask(frame, "non_overlapping_global")
    np.testing.assert_array_equal(accepted, [True, False, False])


def test_non_overlapping_component_allows_different_components_to_overlap() -> None:
    frame = _frame([0, 1, 2, 3], [0, 0, 0, 0], [0, 1, 0, 1], [True, True, True, True], 3)
    accepted = ed.compute_accepted_mask(frame, "non_overlapping_component")
    np.testing.assert_array_equal(accepted, [True, True, False, False])
    # global policy on the same data is strictly more restrictive
    global_accepted = ed.compute_accepted_mask(frame, "non_overlapping_global")
    np.testing.assert_array_equal(global_accepted, [True, False, False, False])


def test_non_overlapping_basket_allows_distinct_baskets_in_same_component_to_overlap() -> None:
    frame = _frame([0, 1], [0, 0], [0, 0], [True, True], 3,
                   weights_a=[1.0, -1.0], weights_b=[-1.0, 1.0])
    basket_accepted = ed.compute_accepted_mask(frame, "non_overlapping_basket")
    np.testing.assert_array_equal(basket_accepted, [True, True])
    component_accepted = ed.compute_accepted_mask(frame, "non_overlapping_component")
    np.testing.assert_array_equal(component_accepted, [True, False])


def test_active_target_count_and_overlap_minutes_hand_traced() -> None:
    frame = _frame([0, 1, 2], [0, 0, 0], [0, 0, 0], [True, True, True], 2)
    annotated = ed.annotate_entry_diagnostics(frame)
    np.testing.assert_array_equal(annotated["active_target_count"], [0, 1, 2])
    np.testing.assert_array_equal(annotated["same_component_active_count"], [0, 1, 2])
    np.testing.assert_array_equal(annotated["overlaps_existing_target"], [False, True, True])
    np.testing.assert_array_equal(annotated["overlap_minutes"], [0, 1, 1])


def test_episode_assignment_splits_on_component_segment_gap_and_ineligibility() -> None:
    source_index = np.array([0, 1, 2, 3, 4, 5, 6])
    segment_id =   np.array([0, 0, 0, 0, 0, 0, 1])
    component =    np.array([0, 0, 1, 1, 0, 0, 0])
    eligible =     np.array([True, True, True, True, False, True, True])
    episodes = ed.assign_signal_episodes(source_index, segment_id, component, eligible)
    assert episodes[0] == episodes[1]        # same component/segment, contiguous
    assert episodes[2] == episodes[3]        # own episode (component switch)
    assert episodes[2] != episodes[1]
    assert episodes[4] == -1                 # not eligible
    assert episodes[5] != episodes[3]        # eligibility lapsed in between -> new episode
    assert episodes[6] != episodes[5]        # segment changed -> new episode
    assert len(set(episodes[episodes >= 0].tolist())) == 4


def test_annotate_entry_diagnostics_does_not_mutate_input_frame() -> None:
    frame = _frame([0, 1, 2], [0, 0, 0], [0, 0, 0], [True, True, True], 2,
                   weights_a=[1.0, 2.0, 3.0])
    original_columns = list(frame.columns)
    original_copy = frame.copy(deep=True)
    ed.annotate_entry_diagnostics(frame)
    assert list(frame.columns) == original_columns
    pd.testing.assert_frame_equal(frame, original_copy)


def test_candidate_change_tracks_every_minute_while_accepted_change_only_tracks_accepted_rows() -> None:
    frame = _frame([0, 1, 2], [0, 0, 0], [0, 0, 0], [True, True, True], 10,
                   weights_a=[1.0, 2.0, 5.0], weights_b=[0.0, 0.0, 0.0])
    annotated = ed.annotate_entry_diagnostics(frame, policy="non_overlapping_global")
    # candidate change is defined every row (minute-over-minute), regardless of acceptance
    np.testing.assert_allclose(annotated["candidate_weight_change_l1"], [0.0, 1.0, 3.0])
    # only row 0 is accepted under non_overlapping_global here (horizon=10 overlaps all)
    np.testing.assert_array_equal(annotated["accepted"], [True, False, False])
    assert np.isnan(annotated["accepted_entry_weight_change_l1"].iloc[0])
    assert annotated["accepted_entry_weight_change_l1"].iloc[1:].isna().all()


@pytest.mark.parametrize("policy", (
    "independent_research_entries",
    "non_overlapping_global",
    "non_overlapping_component",
    "non_overlapping_basket",
))
def test_accepted_rows_under_non_overlapping_policies_never_overlap_each_other(policy: str) -> None:
    rng = np.random.default_rng(5)
    n = 60
    source_index = np.arange(n)
    segment_id = np.zeros(n, dtype=np.int64)
    component = rng.integers(0, 3, size=n)
    eligible = rng.random(n) > 0.3
    horizon = rng.integers(1, 8, size=n)
    frame = _frame(source_index, segment_id, component, eligible, horizon)
    accepted = ed.compute_accepted_mask(frame, policy)
    if policy == "independent_research_entries":
        np.testing.assert_array_equal(accepted, eligible)
        return
    accepted_index = source_index[accepted]
    accepted_end = accepted_index + horizon[accepted]
    accepted_component = component[accepted]
    if policy == "non_overlapping_global":
        groups = [np.arange(len(accepted_index))]
    else:
        # non_overlapping_component, and non_overlapping_basket falling back to a
        # per-component signature when the fixture has no diagnostic_weight_* columns.
        groups = [np.where(accepted_component == value)[0] for value in np.unique(accepted_component)]
    for group in groups:
        if len(group) < 2:
            continue
        order = np.argsort(accepted_index[group])
        starts = accepted_index[group][order]
        ends = accepted_end[group][order]
        for i in range(len(starts) - 1):
            assert starts[i + 1] > ends[i], f"policy={policy} produced overlapping accepted entries"


def test_summarize_entry_policy_counts_are_internally_consistent() -> None:
    source_index = np.arange(30)
    segment_id = np.zeros(30, dtype=np.int64)
    component = (source_index % 2)
    eligible = np.ones(30, dtype=bool)
    eligible[10] = False  # forces an episode split
    frame = _frame(source_index, segment_id, component, eligible, 4)
    summary = ed.summarize_entry_policy(frame, policy="independent_research_entries")
    assert summary.raw_eligible_entries == 29
    assert summary.unique_signal_episodes >= 2
    assert summary.first_entry_count + summary.repeated_entry_count == summary.raw_eligible_entries
    assert summary.max_concurrent_targets >= 1
    assert 0.0 <= summary.pct_entries_overlapping_another_target <= 100.0


def test_reuses_evaluation_protocol_interval_semantics_for_overlap_checks() -> None:
    # sanity: entry_diagnostics' notion of "overlap" agrees with evaluation_protocol's
    starts, ends = ep.target_interval(np.array([0, 1]), 5)
    assert ep.intervals_overlap(starts[0], ends[0], starts[1], ends[1])


def test_segment_transition_resets_acceptance_active_overlap_and_turnover() -> None:
    # The second row would be blocked by row zero without the causal-gap reset.
    frame = _frame([10, 11, 0], [1, 1, 0], [0, 0, 0], [True, True, True], [30, 30, 30],
                   weights_a=[3.0, 4.0, 1.0], weights_b=[0.0, 0.0, 0.0])
    annotated = ed.annotate_entry_diagnostics(frame, policy="non_overlapping_global")
    # Source order is row 2 (segment 0), then rows 0/1 (segment 1); both first rows are accepted.
    np.testing.assert_array_equal(annotated["accepted"], [True, False, True])
    np.testing.assert_array_equal(annotated["active_target_count"], [0, 1, 0])
    np.testing.assert_allclose(annotated["candidate_weight_change_l1"], [0.0, 1.0, 0.0])
    assert np.isnan(annotated.loc[0, "accepted_entry_weight_change_l1"])
    assert np.isnan(annotated.loc[2, "accepted_entry_weight_change_l1"])
    assert annotated.loc[1, "overlap_minutes"] == 29


def test_episode_first_and_cluster_representative_are_explicit_sensitivities() -> None:
    frame = _frame([0, 1, 2, 3, 20], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
                   [True, True, True, True, True], 10)
    frame["target_cluster_id"] = [0, 0, 0, 0, 1]
    independent = ed.compute_accepted_mask(frame, "independent_research_entries")
    episode_first = ed.compute_accepted_mask(frame, "first_entry_per_signal_episode")
    cluster_first = ed.compute_accepted_mask(frame, "one_representative_per_target_cluster")
    assert independent.sum() == 5
    assert episode_first.sum() == 2
    assert cluster_first.sum() == 2


def test_first_entry_after_gap_has_null_accepted_turnover_and_zero_candidate_turnover() -> None:
    frame = _frame([0, 1, 2], [0, 1, 1], [0, 0, 0], [True, True, True], 30,
                   weights_a=[1.0, 9.0, 10.0], weights_b=[0.0, 0.0, 0.0])
    annotated = ed.annotate_entry_diagnostics(frame, policy="non_overlapping_global")
    assert annotated.loc[1, "candidate_weight_change_l1"] == 0.0
    assert np.isnan(annotated.loc[1, "accepted_entry_weight_change_l1"])

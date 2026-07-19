from __future__ import annotations

import numpy as np
import pytest

from engine.evaluation import evaluation_protocol as ep


def test_target_interval_is_inclusive_and_rejects_negative_horizon() -> None:
    start, end = ep.target_interval(np.array([10, 20]), 5)
    np.testing.assert_array_equal(start, [10, 20])
    np.testing.assert_array_equal(end, [15, 25])
    with pytest.raises(ep.EvaluationProtocolError):
        ep.target_interval(np.array([1]), -1)


def test_intervals_overlap_hand_cases() -> None:
    assert ep.intervals_overlap(0, 10, 10, 20)      # touching endpoints overlap (inclusive)
    assert not ep.intervals_overlap(0, 9, 10, 20)    # adjacent, non-touching
    assert ep.intervals_overlap(5, 15, 0, 20)        # fully nested
    assert not ep.intervals_overlap(0, 5, 100, 105)  # far apart


def test_purge_removes_every_row_that_actually_overlaps_held_out() -> None:
    rng = np.random.default_rng(7)
    n = 500
    train_start = rng.integers(0, 1000, size=n)
    train_end = train_start + rng.integers(0, 30, size=n)
    held_out_start = np.array([400])
    held_out_end = np.array([460])
    purge = ep.purge_overlapping_training_rows(train_start, train_end, held_out_start, held_out_end)
    kept = ~purge
    assert not ep.intervals_overlap(train_start[kept], train_end[kept],
                                    np.full(kept.sum(), held_out_start[0]),
                                    np.full(kept.sum(), held_out_end[0])).any()
    naive = np.array([
        bool(np.any((train_start[i] <= held_out_end) & (held_out_start <= train_end[i])))
        for i in range(n)
    ])
    np.testing.assert_array_equal(purge, naive)


def test_purge_matches_brute_force_reference_for_multiple_held_out_ranges() -> None:
    rng = np.random.default_rng(11)
    n_train, n_held = 300, 6
    train_start = rng.integers(0, 2000, size=n_train)
    train_end = train_start + rng.integers(0, 50, size=n_train)
    held_start = rng.integers(0, 2000, size=n_held)
    held_end = held_start + rng.integers(0, 50, size=n_held)
    purge = ep.purge_overlapping_training_rows(train_start, train_end, held_start, held_end)
    naive = np.array([
        bool(np.any((train_start[i] <= held_end) & (held_start <= train_end[i])))
        for i in range(n_train)
    ])
    np.testing.assert_array_equal(purge, naive)


def test_apply_embargo_widens_symmetrically_and_rejects_negative() -> None:
    start, end = ep.apply_embargo(np.array([100]), np.array([130]), 20)
    assert start[0] == 80
    assert end[0] == 150
    with pytest.raises(ep.EvaluationProtocolError):
        ep.apply_embargo(np.array([0]), np.array([1]), -5)


def test_embargo_purges_at_least_as_many_rows_as_plain_purge() -> None:
    rng = np.random.default_rng(3)
    n = 400
    train_start = rng.integers(0, 1000, size=n)
    train_end = train_start + rng.integers(0, 20, size=n)
    held_start, held_end = np.array([500]), np.array([520])
    plain = ep.purge_overlapping_training_rows(train_start, train_end, held_start, held_end)
    wide_start, wide_end = ep.apply_embargo(held_start, held_end, 30)
    widened = ep.purge_overlapping_training_rows(train_start, train_end, wide_start, wide_end)
    assert widened.sum() >= plain.sum()
    assert np.all(widened[plain])  # embargo is a strict superset of plain purge


def test_assign_target_clusters_groups_overlapping_and_splits_on_segment_or_gap() -> None:
    # rows 0,1,2 chain-overlap (each starts before the previous one's end + horizon)
    # row 3 is far away -> its own cluster
    # row 4 overlaps row 3's window but is in a different segment -> forced new cluster
    source_index = np.array([0, 2, 4, 100, 101])
    segment_id =   np.array([0, 0, 0,   0,   1])
    horizon = 5
    clusters = ep.assign_target_clusters(source_index, horizon, segment_id)
    assert clusters[0] == clusters[1] == clusters[2]
    assert clusters[3] != clusters[2]
    assert clusters[4] != clusters[3]
    assert len(set(clusters.tolist())) == 3


def test_build_evaluation_metadata_end_to_end() -> None:
    source_index = np.arange(0, 200)
    segment_id = np.zeros(200, dtype=np.int64)
    horizon_steps = 10
    is_training = source_index < 150
    metadata, summary = ep.build_evaluation_metadata(source_index, horizon_steps, segment_id,
                                                      is_training, embargo_steps=5)
    assert len(metadata) == 200
    assert set(metadata.columns) == {"target_start_index", "target_end_index",
                                     "target_cluster_id", "purged_from_training", "embargoed"}
    # training rows whose target window reaches into [150, 209] must be purged
    train_view = metadata[is_training]
    reaches_eval = train_view["target_end_index"] >= 150
    assert (train_view.loc[reaches_eval, "purged_from_training"]).all()
    assert (~train_view.loc[~reaches_eval, "purged_from_training"]).all()
    # embargo widens the exclusion zone, so it purges a superset
    assert summary.embargoed_extra_purged_rows >= 0
    assert summary.remaining_train_rows == summary.train_rows_total - (
        summary.purged_rows + summary.embargoed_extra_purged_rows)
    # evaluation rows are never marked purged/embargoed (that label only applies to training rows)
    assert not metadata.loc[~is_training, "purged_from_training"].any()
    assert not metadata.loc[~is_training, "embargoed"].any()


def test_build_evaluation_metadata_rejects_mismatched_lengths() -> None:
    with pytest.raises(ep.EvaluationProtocolError):
        ep.build_evaluation_metadata(np.array([1, 2]), 5, np.array([0]), np.array([True, False]))


def test_row_specific_horizons_drive_endpoints_and_clusters() -> None:
    source = np.array([30, 0, 15, 16])  # deliberately unsorted
    horizon = np.array([15, 30, 15, 30])
    segments = np.array([1, 0, 0, 0])
    start, end = ep.target_interval(source, horizon)
    np.testing.assert_array_equal(start, source)
    np.testing.assert_array_equal(end, [45, 30, 30, 46])
    clusters = ep.assign_target_clusters(source, horizon, segments)
    assert clusters[1] == clusters[2] == clusters[3]
    assert clusters[0] != clusters[1]


@pytest.mark.parametrize("invalid", [True, -1, np.array([15.5, 30.0]), np.array([15, 30, 45]),
                                     np.array([[15, 30]]), np.array([np.inf, 30.0])])
def test_target_interval_rejects_invalid_scalar_or_vector_horizons(invalid: object) -> None:
    with pytest.raises(ep.EvaluationProtocolError):
        ep.target_interval(np.array([0, 1]), invalid)  # type: ignore[arg-type]


def test_purge_and_embargo_are_scoped_to_causal_segments() -> None:
    train_start = np.array([95, 95, 130])
    train_end = np.array([110, 110, 145])
    train_segment = np.array([0, 1, 1])
    held_start = np.array([100])
    held_end = np.array([120])
    held_segment = np.array([1])
    purge = ep.purge_overlapping_training_rows(train_start, train_end, held_start, held_end,
                                               train_segment, held_segment)
    np.testing.assert_array_equal(purge, [False, True, False])
    wide_start, wide_end = ep.apply_embargo(held_start, held_end, 15)
    embargo = ep.purge_overlapping_training_rows(train_start, train_end, wide_start, wide_end,
                                                 train_segment, held_segment)
    np.testing.assert_array_equal(embargo, [False, True, True])


def test_metadata_excludes_split_crossers_without_treating_them_as_oos() -> None:
    source = np.array([80, 90, 100, 110])
    horizon = np.array([15, 15, 15, 15])
    segments = np.zeros(4, dtype=np.int64)
    # source=90 crosses the split at 100 and is neither frozen train nor OOS.
    metadata, _summary = ep.build_evaluation_metadata(
        source, horizon, segments, np.array([True, False, False, False]),
        embargo_steps=5, is_evaluation=np.array([False, False, True, True]),
    )
    assert not metadata.loc[0, "purged_from_training"]
    assert not metadata.loc[1, "purged_from_training"]
    assert metadata.loc[1, "target_end_index"] == 105

"""Post-hoc entry-policy and turnover-semantics diagnostics for arena emissions.

``pipeline/stat_arb.py`` emits one candidate research row per eligible
one-minute close; it does not itself decide whether concurrent/overlapping
entries in the same or a different component should be treated as
independent research observations or as a single held position. That is a
real ambiguity (see docs/evaluation-protocol.md): "previous weights" in the
raw emissions is the previous *candidate* basket, computed every eligible
minute, not necessarily a previously *accepted* entry or an open portfolio
position.

This module resolves the ambiguity by making it explicit rather than
picking a silent default deep inside the arena's per-minute loop. It is a
read-only, post-hoc annotation layer: it consumes an already-produced
emissions DataFrame (or anything with the same handful of columns) and
never re-derives or overrides the arena's own entry-eligibility, factor, or
regime logic.

Four entry-policy contracts are supported:

* ``independent_research_entries`` -- every eligible row is its own research
  observation, regardless of overlap. This is the FROZEN DEFAULT: it is the
  interpretation implicitly used by the archived v0.1/v0.2 evaluation code
  in ``pipeline/stat_arb.py`` today (no dedup is applied there), so treating
  it as the default here does not silently change any already-archived
  result's meaning.
* ``non_overlapping_global`` -- at most one accepted entry open at a time,
  across all components and baskets.
* ``non_overlapping_component`` -- at most one accepted entry open at a time
  per selected residual component; different components may overlap.
* ``non_overlapping_basket`` -- at most one accepted entry open at a time
  per distinct basket (component + rounded diagnostic-weight signature);
  the same component with a materially different basket may overlap.

For each policy this module can compute, per row: ``entry_episode_id``,
``active_target_count``, ``same_component_active_count``,
``same_basket_active_count``, ``overlaps_existing_target``,
``overlap_minutes``, and an ``accepted`` flag, plus a summary comparing raw
eligible-entry counts to unique signal-episode counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

ENTRY_POLICIES = (
    "independent_research_entries",
    "non_overlapping_global",
    "non_overlapping_component",
    "non_overlapping_basket",
)
FROZEN_DEFAULT_ENTRY_POLICY = "independent_research_entries"

REQUIRED_COLUMNS = ("source_index", "segment_id", "selected_component_index",
                   "entry_eligible", "holding_horizon_steps")


class EntryDiagnosticsError(RuntimeError):
    """Raised when the input emissions frame does not have the required shape."""


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise EntryDiagnosticsError(f"emissions frame is missing required columns: {missing}")


def assign_signal_episodes(source_index: np.ndarray, segment_id: np.ndarray,
                          selected_component_index: np.ndarray, entry_eligible: np.ndarray,
                          cooldown_steps: int = 0) -> np.ndarray:
    """Group consecutive eligible rows for the same component into one episode.

    An episode ends (the next eligible row starts a new one) when: the
    component changes, the segment changes (a causal gap reset already
    occurred), eligibility itself lapses for one or more rows, or the gap
    between consecutive eligible source indices exceeds
    ``1 + cooldown_steps``. Ineligible rows get episode id ``-1``.
    """
    if cooldown_steps < 0:
        raise EntryDiagnosticsError("cooldown_steps must be non-negative")
    n = len(source_index)
    episode_id = np.full(n, -1, dtype=np.int64)
    order = np.argsort(source_index, kind="stable")
    current_episode = -1
    previous_component: int | None = None
    previous_segment: int | None = None
    previous_index: int | None = None
    for position in order:
        if not entry_eligible[position]:
            previous_index = None
            continue
        starts_new = (
            previous_index is None
            or previous_component != int(selected_component_index[position])
            or previous_segment != int(segment_id[position])
            or (int(source_index[position]) - previous_index) > 1 + cooldown_steps
        )
        if starts_new:
            current_episode += 1
        episode_id[position] = current_episode
        previous_component = int(selected_component_index[position])
        previous_segment = int(segment_id[position])
        previous_index = int(source_index[position])
    return episode_id


def _basket_signature(frame: pd.DataFrame, decimals: int = 6) -> np.ndarray:
    weight_columns = [column for column in frame.columns if column.startswith("diagnostic_weight_")]
    if not weight_columns:
        return frame["selected_component_index"].astype(str).to_numpy()
    rounded = frame[weight_columns].to_numpy(dtype=np.float64).round(decimals)
    return np.array([
        f"{int(component)}:" + ",".join(f"{value:.{decimals}f}" for value in row)
        for component, row in zip(frame["selected_component_index"], rounded)
    ])


def _greedy_non_overlapping_accept(source_index: np.ndarray, target_end_index: np.ndarray,
                                   group_key: np.ndarray) -> np.ndarray:
    accepted = np.zeros(len(source_index), dtype=bool)
    open_until: dict[Any, int] = {}
    order = np.argsort(source_index, kind="stable")
    for position in order:
        key = group_key[position]
        current_end = open_until.get(key, -1)
        if int(source_index[position]) > current_end:
            accepted[position] = True
            open_until[key] = int(target_end_index[position])
    return accepted


def compute_accepted_mask(frame: pd.DataFrame, policy: str) -> np.ndarray:
    """Boolean mask of which eligible rows are "accepted" entries under ``policy``."""
    _require_columns(frame)
    if policy not in ENTRY_POLICIES:
        raise EntryDiagnosticsError(f"unknown entry_policy {policy!r}; expected one of {ENTRY_POLICIES}")
    eligible = frame["entry_eligible"].to_numpy(dtype=bool)
    if policy == "independent_research_entries":
        return eligible.copy()
    source_index = frame["source_index"].to_numpy(dtype=np.int64)
    target_end_index = source_index + frame["holding_horizon_steps"].to_numpy(dtype=np.int64)
    if policy == "non_overlapping_global":
        group_key = np.zeros(len(frame), dtype=np.int64)
    elif policy == "non_overlapping_component":
        group_key = frame["selected_component_index"].to_numpy()
    else:  # non_overlapping_basket
        group_key = _basket_signature(frame)
    accepted = np.zeros(len(frame), dtype=bool)
    if eligible.any():
        accepted[eligible] = _greedy_non_overlapping_accept(
            source_index[eligible], target_end_index[eligible], group_key[eligible])
    return accepted


def annotate_entry_diagnostics(frame: pd.DataFrame, policy: str = FROZEN_DEFAULT_ENTRY_POLICY,
                               cooldown_steps: int = 0) -> pd.DataFrame:
    """Return a copy of ``frame`` with entry-policy/overlap diagnostic columns added.

    Never mutates the input frame. Adds: ``entry_policy``,
    ``entry_episode_id``, ``accepted``, ``active_target_count``,
    ``same_component_active_count``, ``same_basket_active_count``,
    ``overlaps_existing_target``, ``overlap_minutes``,
    ``candidate_weight_change_l1``, and ``accepted_entry_weight_change_l1``.
    """
    _require_columns(frame)
    result = frame.copy()
    source_index = result["source_index"].to_numpy(dtype=np.int64)
    segment_id = result["segment_id"].to_numpy(dtype=np.int64)
    component = result["selected_component_index"].to_numpy()
    eligible = result["entry_eligible"].to_numpy(dtype=bool)
    horizon = result["holding_horizon_steps"].to_numpy(dtype=np.int64)
    target_end = source_index + horizon
    basket_key = _basket_signature(result)

    result["entry_policy"] = policy
    result["entry_episode_id"] = assign_signal_episodes(source_index, segment_id, component,
                                                        eligible, cooldown_steps)
    result["accepted"] = compute_accepted_mask(result, policy)

    active_total = np.zeros(len(result), dtype=np.int64)
    active_component = np.zeros(len(result), dtype=np.int64)
    active_basket = np.zeros(len(result), dtype=np.int64)
    overlap_minutes = np.zeros(len(result), dtype=np.int64)
    order = np.argsort(source_index, kind="stable")
    open_entries: list[tuple[int, int, Any]] = []  # (end_index, component, basket_key)
    for position in order:
        if not eligible[position]:
            continue
        current = int(source_index[position])
        open_entries = [entry for entry in open_entries if entry[0] >= current]
        active_total[position] = len(open_entries)
        active_component[position] = sum(1 for _, comp, _ in open_entries if comp == component[position])
        active_basket[position] = sum(1 for _, _, key in open_entries if key == basket_key[position])
        if open_entries:
            overlap_minutes[position] = max(0, max(end for end, _, _ in open_entries) - current)
        open_entries.append((int(target_end[position]), component[position], basket_key[position]))

    result["active_target_count"] = active_total
    result["same_component_active_count"] = active_component
    result["same_basket_active_count"] = active_basket
    result["overlaps_existing_target"] = active_total > 0
    result["overlap_minutes"] = overlap_minutes

    weight_columns = [column for column in result.columns if column.startswith("diagnostic_weight_")]
    if weight_columns:
        weights = result[weight_columns].to_numpy(dtype=np.float64)
        candidate_change = np.zeros(len(result))
        candidate_change[1:] = np.sum(np.abs(np.diff(weights, axis=0)), axis=1)
        result["candidate_weight_change_l1"] = candidate_change

        accepted = result["accepted"].to_numpy(dtype=bool)
        accepted_change = np.full(len(result), np.nan)
        last_accepted_weights: np.ndarray | None = None
        for position in order:
            if not accepted[position]:
                continue
            if last_accepted_weights is not None:
                accepted_change[position] = float(np.sum(np.abs(weights[position] - last_accepted_weights)))
            last_accepted_weights = weights[position]
        result["accepted_entry_weight_change_l1"] = accepted_change

    return result


@dataclass(frozen=True)
class EntryPolicySummary:
    policy: str
    raw_eligible_entries: int
    unique_signal_episodes: int
    mean_entries_per_episode: float
    max_concurrent_targets: int
    pct_entries_overlapping_another_target: float
    first_entry_count: int
    repeated_entry_count: int


def summarize_entry_policy(frame: pd.DataFrame, policy: str = FROZEN_DEFAULT_ENTRY_POLICY,
                           cooldown_steps: int = 0) -> EntryPolicySummary:
    annotated = annotate_entry_diagnostics(frame, policy=policy, cooldown_steps=cooldown_steps)
    eligible_view = annotated[annotated["entry_eligible"]]
    raw_entries = int(len(eligible_view))
    episode_counts = eligible_view.loc[eligible_view["entry_episode_id"] >= 0, "entry_episode_id"].value_counts()
    unique_episodes = int(episode_counts.shape[0])
    mean_per_episode = float(episode_counts.mean()) if unique_episodes else 0.0
    max_concurrent = int(eligible_view["active_target_count"].max()) + 1 if raw_entries else 0
    pct_overlapping = (float(eligible_view["overlaps_existing_target"].mean()) * 100.0) if raw_entries else 0.0
    is_first_in_episode = ~eligible_view.duplicated(subset="entry_episode_id", keep="first")
    first_count = int(is_first_in_episode.sum())
    repeated_count = raw_entries - first_count
    return EntryPolicySummary(
        policy=policy,
        raw_eligible_entries=raw_entries,
        unique_signal_episodes=unique_episodes,
        mean_entries_per_episode=mean_per_episode,
        max_concurrent_targets=max_concurrent,
        pct_entries_overlapping_another_target=pct_overlapping,
        first_entry_count=first_count,
        repeated_entry_count=repeated_count,
    )

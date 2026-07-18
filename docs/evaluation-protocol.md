# Evaluation Protocol: Overlap, Purging, Embargo, and Entry Policy

This document covers two reusable, standalone modules added on
`agent2/research-reliability-hardening`:

- `pipeline/evaluation_protocol.py` — purge/embargo/cluster helpers for any
  chronological train/OOS split.
- `pipeline/entry_diagnostics.py` — post-hoc entry-policy and overlap
  annotation for arena emissions.

Neither module edits `pipeline/stat_arb.py` or `pipeline/legal_event.py`.
Both operate on plain arrays / an already-produced emissions `DataFrame`, so
they apply to any research module with the same shape of problem: a row at
`source_index` makes a claim about an outcome observed over
`[source_index, source_index + horizon_steps]`.

## The problem

A chronological train/OOS split (as both arenas already do) prevents *test
data* from leaking into training by timestamp. It does not, by itself,
prevent a *training* row's own outcome window from reaching past the split
boundary into the OOS period — nor does it say anything about how many
training/OOS rows have outcome windows that overlap *each other*, which
inflates the effective sample size used for any significance claim.
Separately, an arena that emits one candidate research row every eligible
minute has no built-in notion of "how many of these rows represent
genuinely independent bets" versus "the same dislocation observed 40 times
in a row while its horizon is still open."

## `evaluation_protocol.py`: purge and embargo

```python
target_interval(source_index, horizon_steps) -> (start, end)          # inclusive
intervals_overlap(a_start, a_end, b_start, b_end) -> bool | ndarray
apply_embargo(held_out_start, held_out_end, embargo_steps) -> (start, end)
purge_overlapping_training_rows(train_start, train_end, held_out_start, held_out_end) -> bool ndarray
assign_target_clusters(source_index, horizon_steps, segment_id) -> cluster_id ndarray
build_evaluation_metadata(source_index, horizon_steps, segment_id, is_training, embargo_steps=0) -> (DataFrame, PurgeSummary)
```

`build_evaluation_metadata` never changes which rows are training vs.
evaluation — `is_training` is supplied by the caller exactly as already
computed by the frozen chronological split. It only *annotates*:

- `target_start_index`, `target_end_index` — the row's outcome window.
- `target_cluster_id` — connected-component id: rows whose outcome windows
  chain together via overlap, within the same causal segment, share a
  cluster. Use a cluster count instead of a raw row count when a
  significance test needs a count of independent observations.
- `purged_from_training` — True for a training row whose outcome window
  overlaps *any* evaluation-partition row's outcome window.
- `embargoed` — the same, but against evaluation windows widened by
  `embargo_steps` on both sides (a superset of `purged_from_training`).

**This module produces a sensitivity view, not a new frozen evaluation.**
Nothing currently calls it from inside either arena's canonical run. To use
it: run the arena, take its emissions frame's `source_index`/`segment_id`
and the same `is_training` boolean the arena's own chronological split uses,
call `build_evaluation_metadata`, and report the purged/embargoed metric
alongside — never in place of — the frozen contract result. Recommended
embargo sensitivities: 0, 30, 240, and 1440 minutes (see
`tests/test_evaluation_protocol.py` for the purge-correctness and embargo-is-
a-superset property tests backing this).

## `entry_diagnostics.py`: entry-policy and overlap annotation

`pipeline/stat_arb.py` emits one research row per eligible minute. Whether
concurrent/overlapping eligible rows should count as independent research
observations, or as a single held position re-observed every minute while
its horizon is open, is a real modeling choice — not something this branch
should decide silently inside the arena's per-minute loop. Four contracts:

| Policy | Rule |
|---|---|
| `independent_research_entries` (**frozen default**) | every eligible row counts, regardless of overlap — this is what the archived v0.1/v0.2 evaluation already does today, so naming it the default does not silently change any archived result's meaning |
| `non_overlapping_global` | at most one accepted entry open at a time, across all components and baskets |
| `non_overlapping_component` | at most one open entry per selected residual component; different components may overlap |
| `non_overlapping_basket` | at most one open entry per distinct basket (component + rounded diagnostic-weight signature); the same component with a materially different basket may overlap |

```python
annotate_entry_diagnostics(frame, policy=FROZEN_DEFAULT_ENTRY_POLICY, cooldown_steps=0) -> DataFrame
summarize_entry_policy(frame, policy=..., cooldown_steps=0) -> EntryPolicySummary
```

`annotate_entry_diagnostics` adds, without mutating the input: `entry_policy`,
`entry_episode_id` (consecutive eligible rows for the same component, split
on component change / segment change / eligibility lapse / gap beyond
`cooldown_steps`), `accepted`, `active_target_count`,
`same_component_active_count`, `same_basket_active_count`,
`overlaps_existing_target`, `overlap_minutes`, `candidate_weight_change_l1`
(minute-over-minute change in the raw candidate basket — this is what the
arena's own `turnover_l1_diagnostic` column already measures), and
`accepted_entry_weight_change_l1` (change only between consecutive
*accepted* entries under the chosen policy — `NaN` everywhere else). Keeping
both columns side by side, rather than renaming the arena's own column,
resolves the "previous weights: candidate or position?" ambiguity by making
both readings available and explicitly labeled instead of picking one
silently.

`summarize_entry_policy` reports raw eligible-entry count, unique
signal-episode count, mean entries per episode, max concurrent targets, the
percentage of entries overlapping another target, and a first-entry vs.
repeated-entry split.

## What this does not do

- It does not change `pipeline/stat_arb.py`'s frozen v0.1/v0.2 evaluation,
  bootstrap, or promotion logic.
- It does not pick a "correct" entry policy — that remains a modeling
  decision for whoever owns the arena's trading-policy contract.
- `non_overlapping_basket`'s signature is a rounded (6-decimal) hash of the
  `diagnostic_weight_*` columns; it is an approximation of "materially
  different basket," not a semantic comparison of basket construction logic.

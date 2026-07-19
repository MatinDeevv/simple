# Causal FX Residual-Level Research Arena

`engine/models/statistical/stat_arb.py` is a causal forecast-evaluation
arena, not an execution system. It uses one-minute BID closes only and makes
no PnL, tradability, spread, commission, fill, latency, impact, capacity,
leverage, contract-notional, or market-making claim.

## Entry-policy sensitivity and publication provenance

The frozen primary population remains independent research entries. Secondary
policy sensitivities are descriptive only: no policy is selected by score.
For each fixed policy, the three-class conditional comparator is refit using
only that policy's accepted training rows, then scored on accepted OOS rows.
`reset_at_oos_start` is the main comparability sensitivity; the separate
`carry_chronological_state_into_oos` view reports episodes or open targets
crossing the split. Sparse secondary views report an explicit insufficiency
status rather than changing the frozen primary result.

Publication writes staging bytes but records hashes under deterministic final
logical paths. After atomic publication, every output is rehashed against the
manifest; failed verification removes the unverified run. Frame contracts are
checked vectorially with strict serialization reserved for representative rows.
Duplicate source indices are rejected for bootstrap evaluation: one physical
minute is one primary evaluation observation.

## Corrective frozen contract: v0.2.1

`stat-arb-arena-0.2.1-corrective-frozen` corrects evaluation semantics that
were documented for v0.2.0 but not implemented. The correction was made before
any new post-2024 data inspection and is not a signal, threshold, regime,
horizon, factor-count, or profitability change.

- v0.2.0's primary three-class gate compared the model only with a global
  train prior, despite documentation specifying a conditional climatology.
- v0.2.1 uses that documented train-only conditional three-class comparator.
- Existing v0.1 archive paths and any possible v0.2.0 artifacts are never
  overwritten. New publications use a distinct `stat_arb_v0_2_1_runs/<uuid>/`
  directory.

All currently available canonical observations include the already inspected
2024 holdout. The data CLI therefore remains guarded. `--self-check` is the
supported synthetic check; `--allow-burned-holdout-research` only permits a
non-promotable forensic run and must never be used for tuning, selection, or
advertising. Post-2024 untouched data and execution-quality inputs remain
required before any empirical promotion decision.

## Causal target and row-specific horizon

The primary target is the entry-frozen standardized basket return with class
order `neutral`, `positive`, `negative`; neutral outcomes remain in every
primary score. The selected component, both-regime factor basis, residual
scales, regime probabilities, diagnostic basket, stop, and each row's actual
`holding_horizon_steps` are frozen at entry. A target is absent unless every
arrival from entry through its individual horizon is an observed contiguous
minute.

Each labelled row records its truthful inclusive target interval,
`target_start_index` through `target_end_index`, and connected
`target_cluster_id`. Training means the complete target ends before the split;
a row crossing the split is excluded from frozen training. It is not silently
given the maximum or average horizon.

## Primary comparator and uncertainty

The primary comparator is a three-class conditional climatology fitted only to
the permitted training labels. It returns an `N x 3` probability matrix in
`neutral`, `positive`, `negative` order. Its fixed sparse-cell hierarchy is:

1. absolute residual-level bin + selected component + UTC session bucket + active regime;
2. absolute residual-level bin + selected component + active regime;
3. absolute residual-level bin + active regime;
4. global three-class train prior.

Every tier uses symmetric Dirichlet/Laplace smoothing with fixed `alpha=1.0`
and a fixed minimum cell count of 20. Each OOS row persists the selected tier
and its three comparator probabilities. The global train prior remains a
separately reported simple comparator, but is no longer the sole primary gate.

The directional binary climatology remains a secondary diagnostic only. Its
name explicitly states that neutral outcomes are excluded.

The primary gate compares model probabilities with the three-class conditional
comparator using contiguous observed-minute block bootstrap sensitivity
(`30`, `240`, and `1440` minutes by default). The predeclared decision bound is
`lower_one_sided_95 = q0.05`. Fields are unambiguous:

- `lower_one_sided_95 = q0.05`, `upper_one_sided_95 = q0.95`;
- `two_sided_95_lower = q0.025`, `two_sided_95_upper = q0.975`;
- `lower_95_deprecated` and `upper_95_deprecated` are temporary aliases only.

## Segment, purge, and overlap sensitivities

A causal segment transition is a hard boundary. Target clusters, purging,
embargo, accepted-entry state, active-overlap counts, overlap minutes, and
candidate/accepted turnover never cross it. The first candidate in a segment
has candidate turnover zero; the first accepted entry in a segment has
accepted-entry turnover null.

The frozen chronological view, purged view, and purged-plus-embargoed view
score the same OOS population and the unchanged model probabilities. Each
refits the real three-class conditional comparator on only its allowed train
labels, then reports train/OOS counts, fallback-tier counts, Brier and log-loss
scores, model-minus-baseline improvements, and observed-minute uncertainty.

The named primary population remains `independent_research_entries` for
compatibility. The summary reports every view in this fixed order, without
choosing the best result: independent entries, first entry per signal episode,
non-overlapping global, component, basket, and one representative per target
cluster. Each view exposes eligible/accepted counts, unique episode and target
cluster counts, class counts, scores, and either uncertainty or an explicit
insufficient-sample status.

## Transactional artifacts and validation

Each publication creates a UUID and writes only to a sibling staging directory.
It validates serialized emission rows, graph rows, daily rows, the complete
summary, and the existing research-run manifest before one directory move
publishes the run. The summary already contains final relative artifact paths
and source hashes before it is validated and written. JSON is serialized with
`allow_nan=False`; NaN/infinity are rejected at the summary boundary, while
nullable emission fields are explicitly serialized as JSON `null`.

The published directory contains exactly `minute.parquet`, `graph.parquet`,
`daily.parquet`, `summary.json`, and `manifest.json`. Failed staging is removed;
an existing final UUID directory is refused rather than overwritten. Returned,
persisted, and printed summaries are the same JSON structure. The manifest has
`required_tests_passed=false` for a normal run unless real repository-defined
evidence is supplied, and the burned-holdout blocker remains in force.

`stat-arb-summary.schema.json` requires the evaluation contract, comparator,
purge/embargo views, entry-policy sensitivities, confidence convention,
optimizer diagnostics, paths, hashes, and contract version.
`stat-arb-emission.schema.json` validates integration-critical interval,
segment, entry-policy, overlap, turnover, optimizer, and probability fields.
The graph and daily files have dedicated row schemas.

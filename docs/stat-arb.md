# Causal FX Residual-Level Research Arena

`engine/models/statistical/stat_arb.py` is a classical, causality-first research arena. It is
not a strategy, backtest, OMS, execution model, market maker, or portfolio
authorization system. The source is one-minute BID bars only; it has no ask,
spread, fill, queue, borrow, impact, capacity, contract-notional, or conversion
price data.

## Frozen v0.2 contract

Version `stat-arb-arena-0.2.0-frozen` is fixed before obtaining new data. It is
synthetically tested only. Its normal data CLI is guarded because all current
canonical observations end in the already inspected 2024 holdout.

```powershell
python engine\models\statistical\stat_arb.py --self-check
```

`--allow-burned-holdout-research` exists only for an explicit, non-promotable
forensic run. It must not be used to select, tune, or report a v0.2 result.
New post-2024 data and a predeclared chronological split are required before a
v0.2 empirical evaluation.

The old `stat_arb_*` artifacts are preserved as v0.1 archive files. A later
authorized v0.2 run writes only `stat_arb_v0_2_*` files, so it cannot overwrite
the archive.

## Correct basis and portfolio mapping

Let `z = T r` be the identity-free return transform. A residual signal `s` is a
dual vector, therefore its raw-return functional is:

```text
s' z = s' T r = (T' s)' r
```

The basket signal is consequently `T.T @ s`, not `inverse(T) @ s`.
Factor loading columns are primal directions; their raw map remains
`inverse(T) @ B`. Tests assert the equality above for each EURGBP, EURJPY, and
GBPJPY triangle-residual channel over randomized returns.

The basket constraints are also explicit. `D` is the pair-by-currency incidence
matrix: base currency `+1`, quote currency `-1`. Diagnostic weights satisfy
`D.T @ w = 0` plus a predeclared number of factor-direction constraints. This
is currency-incidence neutrality in pair-coefficient units, not dollar-risk
neutrality: contract notionals and conversion prices are absent, so risk-unit
sizing and any executable neutrality claim remain blocked.

## Causal target

v0.1 selected the largest current return residual and labelled whether a later,
re-estimated standardized return was smaller. That was an invalid convergence
target: it was dominated by order statistics/regression to the mean and changed
factor loadings, scales, and basis between entry and evaluation.

v0.2 maintains two identity-free standardized residual levels, one for each
regime definition:

```text
S_t^j = rho_j S_(t-1)^j + e_t^j / sigma_t^j
S_t = P(z_t=low) S_t^low + P(z_t=high) S_t^high
```

Each branch is updated only in its own mean/loading/scale definition; a regime
switch cannot inject one model's innovation into the other level. The posterior
is applied once, at the mixture stage; it is not also multiplied into a branch
innovation. Emissions decompose each blended level change into branch-AR decay,
full conditional innovation, and posterior reweighting. At each entry
the selected component, both factor means/loadings/scales, regime probabilities,
both level AR coefficients, basket, holding horizon, and stop are frozen.
Future returns are reprojected through exactly those frozen definitions. The
residual explanatory diagnostic is:

```text
gross_convergence = -sign(S_t) * (S_(t+h,frozen) - S_t)
```

Residual convergence requires positive gross convergence and a smaller absolute
frozen level. It remains explanatory. The **primary predictive target** is the
frozen basket return standardized at entry:

```text
y = sum_h(w' r) / (sigma_basket,entry * sqrt(h))
```

The fixed neutral zone is `|y| <= 0.25`. Primary evaluation is three-class
(`negative`, `neutral`, `positive`) with multiclass log loss and Brier score;
neutral observations are never removed from the primary score. This is still a
no-cost diagnostic return, not PnL or an executable performance claim.

Every entry records projection distortion, fraction of the original signal
preserved, correlation between basket returns and selected residual returns,
basket/residual-label disagreement, basket directional hit, cumulative gross
log return, and basket maximum adverse excursion. Emissions also retain
residual MAE, time-to-zero, displacement removed, breakdown, frozen-path
volatility, and turnover. A target is absent when even one arrival from entry
through horizon is not an observed contiguous minute.

## Basket-space modes and post-construction gate

`cycle_neutral` is the frozen default: it enforces `D.T @ w = 0` in
pair-coefficient space and therefore admits only closed currency loops, along
with the selected factor-neutrality constraints. `relative_value` is a separate
diagnostic contract: it solves a constrained projected utility problem with
selected-factor neutrality, L1 gross normalization, per-pair caps, and a
`max(abs(D.T @ w))` currency-incidence budget. The budget changes composition,
rather than merely shrinking the same vector.
It does not claim dollar or risk neutrality because contract notionals and FX
conversion prices are not available.

Selection is provisional until the actual basket is constructed. Probability
and entry gating consume the post-projection signal-preservation fraction,
projection distortion, geometric signal alignment, and concentration. Gross
exposure is only a nonzero eligibility check. The hard preservation gate is fixed at `0.35`; an erased or
over-concentrated projection cannot enter merely because its pre-projection
residual was large.

## Actual regime and graph effects

This is regime switching, not merely regime-aware scoring. A causal two-state
posterior maintains distinct low/high covariance, factor loading, residual
variance, and residual-AR state. The active state also changes factor refresh
cadence, level persistence, entry threshold, holding horizon, stop multiple,
number of neutralized factors, and diagnostic position scale.

The sparse partial-correlation graph is not decorative. Its incident pressure
and connected-cluster size penalize residual selection, enter breakdown risk,
and reduce diagnostic position scale. The graph edge artifact records the
active regime at each refresh.

## Evaluation baselines and uncertainty

All baselines are frozen from the train partition. The primary comparator is a
conditional **basket-directional** climatology indexed by absolute
residual-level bin, component, UTC session bucket, and regime, with
predeclared sparse-cell fallbacks. A deterministic time-shuffled-label placebo
is reported separately.

The statistical interval is an exact-entry-count bootstrap whose dependence
blocks are actual contiguous observed one-minute ranges. A 1,440-minute block
is therefore one observed trading day of raw bars, not 1,440 sparse eligible
entries. Each sampled time block contributes the eligible entries inside it;
the final replicate is truncated to the original entry count. It uses at least
2,000 replicates by default, reports Brier, log-loss, and calibration-improvement
intervals, and records 30-minute, 4-hour, and one-day sensitivities. The
conditional-climatology gate requires positive one-day lower-95% Brier
improvement; it still cannot promote anything without executable data and a
new untouched holdout.

IID-residual simulation, AR(1), static-vs-dynamic PCA, and no-regime ablations
remain required matched comparators before any empirical model comparison. They
are not claimed as executed v0.2 results.

## Archived v0.1 evidence

The 50,000-row v0.1 bounded diagnostic covered 48,653 synchronous rows from
2024-11-12T07:30Z through 2024-12-31T21:59Z. It had 483 gap resets and failed:
Brier `0.397654` versus frozen-prior `0.150843`, with lower-95% moving-block
Brier improvement `-0.278320`.

The v0.1 2024 outer fold processed 1,071,797 synchronous rows. Its
2022-2023 history served both to warm causal state and to establish frozen
training baseline-label estimates; 2024 was the scored outer partition. It
also failed: Brier `0.462652` versus `0.150255`, lower-95% improvement
`-0.337775`. These are archived rejection evidence, not a v0.2 baseline and
not a license to tune on 2024.

## Artifacts

- `data/derived/stat_arb_*`: immutable v0.1 archive artifacts.
- `data/derived/stat_arb_v0_2_*_minute.parquet`: causal v0.2 emissions,
  frozen-target outcomes, weights, exposures, and diagnostics.
- `data/derived/stat_arb_v0_2_*_graph.parquet`: active-regime graph edges.
- `data/derived/stat_arb_v0_2_*_daily.parquet`: diagnostic aggregates only.
- `data/derived/stat_arb_v0_2_*_summary.json`: target definition, data hashes,
  baselines, bootstrap sensitivity, and non-promotion status.

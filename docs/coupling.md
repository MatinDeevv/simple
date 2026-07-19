# Causal Coupling Field (OQ-6, OQ-7)

| Item | v1 definition |
|---|---|
| Owner | sim-coupling |
| Input | `data/canonical/<PAIR>.parquet` only; raw CSV is never read |
| Instrument order | `[EURUSD, USDJPY, GBPUSD, AUDUSD, USDCAD, USDCNH, USDCHF, EURGBP, EURJPY, GBPJPY]` |
| Default command | `python engine/models/classical/estimate_coupling.py` |
| Output | `data/derived/coupling_estimates.parquet`, `coupling_diagnostics.parquet`, `coupling_summary.json`, `coupling_validation.json` |

This is a classical, time-varying dependence field. “Correlation” has no quantum-mechanics meaning.

## Computable field

At an observed, fully synchronous minute `t`, let `x_i(t)=ln(close_bid_i,t)` and `tau0=60 s`.

```
g_j(t) = tau0 * v_j(t) = x_j(t) - x_j(t-60 s)                 [nats]
a_i(t) = [x_i(t) - 2x_i(t-60 s) + x_i(t-120 s)] / tau0^2     [nats s^-2]
```

The simulator consumes the delivered raw-coordinate matrix exactly as:

```
coupling_force_i(t) = sum_j C_ij(t) * g_j(t)                 [nats s^-2]
```

`C_ij` is therefore in `s^-2`; row `i` is affected instrument, column `j` is source instrument. `C_ii=0` exactly. The matrix is not symmetrized: each target row has its own regression and `C_ij != C_ji` is permitted.

## OQ-7: structural identity-free basis

The estimator never treats the three cross arithmetic relations as discovered dependence. Define `z=Tg`, `b=Ta`, where channels 0–6 remain the raw returns/accelerations and:

```
z_7 = g_EURGBP - g_EURUSD + g_GBPUSD
z_8 = g_EURJPY - g_EURUSD - g_USDJPY
z_9 = g_GBPJPY - g_GBPUSD - g_USDJPY
```

and identically for `b`. Thus channels 7–9 are cross-quote residual movements. `T` is invertible, so the fit is mapped back to the schema’s raw-instrument convention:

```
M(t) = [sum b(s) z(s)^T] * [sum z(s) z(s)^T + lambda diag(diag(sum z(s) z(s)^T))]^-1
C_raw(t) = offdiag(T^-1 M(t) T)
```

`lambda=0.001` is dimensionless; the diagonal ridge has the same units as `sum z z^T`. `offdiag` sets the raw diagonal to exactly zero, because self-dynamics are assigned to `m,k,c`, not coupling. The removed unconstrained diagonal norm is emitted as a diagnostic so that this convention is visible.

This construction retains independent major and non-triangle pair channels while representing EURGBP/EURJPY/GBPJPY only through deviations from their stipulated arithmetic relationships during estimation. It is not “zeroing the pairs.”

## Causality, cadence, and alignment

- Snapshot cadence: the last valid sample in every UTC calendar day.
- Lookback: exactly the last 28,800 valid two-step synchronous samples ending at the snapshot (`--lookback-samples` changes this explicitly). This is approximately twenty 24-hour-equivalent minute blocks, not a promise of 20 contiguous market days.
- Every fit statistic, including ridge scale and triangle diagnostic, is recomputed from that trailing window only. No full-sample centering, normalizer, shuffle, centered difference, or future row is used.
- Alignment: timestamps are the strict intersection of all ten canonical timestamp columns. There is no as-of join, interpolation, forward fill, or dense-grid construction.
- A usable sample needs all ten pairs at `t`, `t-60 s`, and `t-120 s` in that intersection. Any weekend, outage, or asynchronous/missing bar invalidates the sample; it is excluded rather than bridged.

The estimator does not decide integrator behavior across a gap (OQ-8). It merely refuses to estimate a local 60-second coupling across one.

## Low-data and regime-break behavior

Before 28,800 usable samples, a daily diagnostics row is recorded with `INSUFFICIENT_HISTORY`; no matrix is invented. A non-finite fit or condition number over `1e12` records `REJECTED_NUMERICAL_CONDITION`; no zero, identity, or stale substitute matrix is written for that time. A consumer must hold its previously accepted field or explicitly decouple/re-initialize under the integrator’s policy, while preserving the rejection event.

The emitted diagnostic contains the regularized condition number, maximum absolute coefficient, removed self-component norm, and Frobenius change from the last accepted matrix. `regime_break_flag=true` when the relative Frobenius change is at least 0.50 or the condition number is at least `1e8`. A numerical rejection is an explicit decoupling-required event: no substitute field is emitted. v1 does not silently damp or clamp the matrix; policy action belongs to the simulator/integrator.

## Triangle enforcement diagnostic

For each accepted window the diagnostics report, for each identity, mean absolute cross-parent correlation before and after arithmetic removal:

| Identity | Before | After |
|---|---|---|
| T-1 | mean `abs(corr(g_EURGBP,g_EURUSD))`, `abs(corr(g_EURGBP,g_GBPUSD))` | same correlations with `z_7` replacing `g_EURGBP` |
| T-2 | EURJPY versus EURUSD/USDJPY | `z_8` versus EURUSD/USDJPY |
| T-3 | GBPJPY versus GBPUSD/USDJPY | `z_9` versus GBPUSD/USDJPY |

The intended result is material reduction from pre to post. It proves the estimator removed the stipulated arithmetic component from the estimated channels; it does not prove an arbitrage relationship or causal trading signal. The actual latest values from an executed run are persisted in `coupling_summary.json` and every window’s values in `coupling_diagnostics.parquet`.

## Output schema

`coupling_estimates.parquet` is a long 100-row-per-accepted-snapshot table:

| Column | Type | Meaning |
|---|---|---|
| `update_time`, `window_start_time` | `timestamp[us, UTC]` | causal end and first included usable sample |
| `effective_samples` | int32 | exactly the configured lookback on accepted rows |
| `affected_index`, `source_index` | int8 | frozen index order above |
| `affected_symbol`, `source_symbol` | string | redundant, human-checkable labels |
| `c_ij_s_minus_2` | float64 | raw-coordinate `C_ij`, units `s^-2` |

`coupling_diagnostics.parquet` has one row per eligible UTC day, including accepted and non-accepted states, causal window endpoints, condition metrics, and `t1/t2/t3_*_abs_corr`. `coupling_validation.json` records zero-diagonal, finite-matrix, causality-boundary, triangle-diagnostic, and static-transform self-checks. `coupling_summary.json` contains input counts, timestamp span, run time, and latest triangle values.

## BID-only limits

All prices are BID prices. No ask or spread is observed, so triangle residuals include bid-side quote construction, vendor timing, and microstructure noise. They are neither executable cross-arbitrage residuals nor a friction estimate. Contemporaneous regression at a bar close is a conditional dependence estimate, not proof that source `j` causes target `i`; directional rows are a model convention for the simulator.

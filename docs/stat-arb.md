# Causal FX Regime and Residual Research Arena

`pipeline/stat_arb.py` is the first implementation of OQ-14. It is a
classical, forecast-evaluation research arena for the only architecture in the
proposal that matches the available data: regime-aware cross-sectional FX
residual analysis.

It is not a live strategy, backtest, OMS, smart order router, market maker,
options engine, macro nowcaster, or legal-event system. The canonical source
contains one-minute BID bars only. It has no ask, spread, fill, queue, order
book, borrow, impact, options-chain, macro-vintage, or legal-document data.

## Causal flow

```text
canonical synchronous BID closes through t
  -> valid one-minute returns, reset at each observed gap
  -> identity-free FX transform
  -> causal EW factor covariance and sparse partial-correlation graph
  -> causal two-state volatility-regime filter and residual AR diagnostics
  -> residual ranking, breakdown probability, factor/net-neutral diagnostic weights
  -> fixed-horizon convergence probability at t
  -> target observed only after t + 30 contiguous minutes
  -> frozen train-prior Brier comparison and block-bootstrap interval
```

EURGBP, EURJPY, and GBPJPY are converted to triangle-residual channels before
factor and graph estimation. Their arithmetic relationships therefore cannot be
misreported as discovered cross-asset economic factors.

## Predeclared target and gate

At each eligible close, the model selects the largest absolute standardized
identity-free residual. The binary target is whether that same residual has a
smaller absolute standardized magnitude after 30 observed contiguous minutes.
The target never enters feature state, regime inference, factor updates, or
allocation construction.

The primary score is Brier score against the convergence frequency frozen from
the training partition. The reported lower 95% confidence bound is a
deterministic moving-block bootstrap of the Brier improvement, with one-day
blocks. A candidate needs a lower bound above zero before it has passed even
the prediction gate. It still cannot be promoted without all predeclared 2022,
2023, and 2024 outer folds and an independently specified execution-data
contract.

The model emits L1-normalized diagnostic weights that jointly remove current
factor exposure and net exposure. They are not orders. With BID-only data,
execution costs, capacity, fill probability, impact, and PnL are explicitly
blocked rather than assumed to be zero.

## Commands

```powershell
python pipeline\stat_arb.py --self-check
python pipeline\stat_arb.py --max-rows 50000
python pipeline\stat_arb.py --test-year 2022
python pipeline\stat_arb.py --test-year 2023
python pipeline\stat_arb.py --test-year 2024
```

The bounded command is a development diagnostic; it is not one of the required
calendar outer folds. A `--test-year` run loads two prior calendar years solely
to warm causal state and scores the selected calendar year.

## First bounded result

The 50,000-row request yielded 48,653 synchronous rows from
2024-11-12T07:30Z through 2024-12-31T21:59Z. It had 483 observed gap resets;
the longest contiguous segment was 1,394 steps, so post-gap warmup is a
predeclared 60 observed contiguous minutes rather than an invented session
bridge.

The fixed candidate failed its diagnostic gate:

| Predictor | OOS Brier | OOS log loss | Accuracy |
|---|---:|---:|---:|
| Factor/regime residual candidate | 0.397654 | 1.075137 | 44.06% |
| Frozen training convergence prior | 0.150843 | 0.479053 | 81.49% |

The lower 95% moving-block-bootstrap bound for Brier improvement was
`-0.278320`. This is a rejection result. It provides no trading evidence and
does not authorize tuning, PnL construction, or another model family.

## 2024 outer-fold result

The first required calendar fold used 2022-2023 only for causal state warmup
and scored 2024. It processed 1,071,797 synchronized rows, with 532,814
training-target observations and 236,358 OOS targets. The fixed candidate had
Brier `0.462652` versus `0.150255` for the frozen training prior; its lower 95%
moving-block-bootstrap Brier-improvement bound was `-0.337775`.

This is another rejection. The remaining 2022 and 2023 commands are preserved
for protocol completeness, but this 2024 outcome does not authorize tuning the
candidate on any outer-fold result.

## Artifacts

- `data_derived/stat_arb_*_minute.parquet`: causal emissions, residual
  diagnostics, weights, labels, and timing.
- `data_derived/stat_arb_*_graph.parquet`: thresholded partial-correlation
  edges at factor refreshes.
- `data_derived/stat_arb_*_daily.parquet`: diagnostic aggregates only.
- `data_derived/stat_arb_*_summary.json`: target, split, baseline, bootstrap,
  data hashes, and non-promotion status.

The next authorised work is to execute and inspect all three outer folds, then
define matched classical alternatives without looking at those test years.

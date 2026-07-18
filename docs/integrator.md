# Classical Integrator: Numerical Safety, Not Harmonic-Model Acceptance

| Item | Current contract |
|---|---|
| Owner | sim-integrator |
| Implementation | `pipeline/simulate_integrator.py`, `integrator-1.2.0` |
| Scope | EURUSD, USDJPY, USDCNH only; indices `[0,1,5]` |
| Status | Gap handling closed; harmonic-curvature identification and non-normal stability acceptance reopened |
| Non-claim | This is neither a ten-pair simulation nor a validated restoring-force model. |

## Discrete update

At a contiguous 60-second arrival, parameters and coupling are zero-order-held
from the previous timestamp `t`:

```text
g(t) = 60 * v_hat(t)
A_c(t) = C(t) * g(t)                    # specific acceleration
kappa_raw(t) = k(t) / m(t)
gamma(t) = c(t) / m(t)
kappa_sim(t) = max(kappa_raw(t), 0)     # model-changing projection

v_hat(t+60) = [v_hat(t) + 60 * (-kappa_sim(t)*(x_hat(t)-x_eq(t)) + A_c(t))]
              / [1 + 60*gamma(t)]
x_hat(t+60) = x_hat(t) + 60*v_hat(t+60)
```

`C` maps `g` to specific acceleration and is therefore not divided by mass.
Forcing remains zero: adding residual forcing before jointly conditioning it on
coupling would double-count cross-pair effects.

## Gap and checkpoint contracts

When an arriving bar reveals `dt != 60 s`, the replay takes no large step,
sets `x_hat` to the observed position, sets `v_hat=0`, and logs the reset. The
first later contiguous arrival is eligible again; no session-crossing velocity
is fabricated.

A checkpoint now stores two unambiguous indices:

```text
state_index          state is x_hat/v_hat at times[state_index]
next_arrival_index   exactly the next row to process; state_index + 1
```

Resume begins at `next_arrival_index`, not the following row. The test suite
proves a split/resumed replay reaches the same final state as an uninterrupted
replay, including the legacy `integrator-1.1.0` checkpoint payload.

## Non-normal stability diagnostics

The directional coupling field makes the six-state amplification matrix
potentially non-normal, so `rho(A) <= 1` alone is insufficient. For every
causal configuration, the script reports:

- spectral radius `rho(A)`;
- largest singular value `sigma_max(A)`;
- eigenvector condition number;
- `max_{1<=n<=60} ||A^n||_2`;
- an estimated unit-circle pseudospectral distance from sampled resolvents.

Norm-based quantities use the dimensionally balanced state `[x_hat, 60*v_hat]`
rather than `[x_hat, v_hat]`. The timestep report is a sampled grid from 1 to
3,600 seconds; it records stable segments and makes no monotonic-bisection
claim. A configuration is integrated only if the guarded 60-second spectral
and 60-step transient-growth checks pass.

## Negative curvature is a model failure signal

The projection `kappa_sim=max(kappa_raw,0)` is not a solver-only guard. It
replaces a negative local quadratic potential with a flat one. It is logged as
`MODEL_PROJECT_NEGATIVE_KAPPA_TO_ZERO`, and any replay using it has
`model_status=REJECTED_HARMONIC_IDENTIFICATION_NEGATIVE_CURVATURE_PROJECTED`.

The next research question is not another guard. It is how to identify a
bounded nonlinear potential, regime model, or estimation-noise explanation
without selecting it on the evaluation data. A quartic potential is only a
candidate specification; it has not been implemented or accepted.

## Executed canonical replay

Command:

```powershell
python pipeline\simulate_integrator.py --max-steps 250000
```

| Metric | Result |
|---|---:|
| Stability configurations | 3,850 |
| Raw maximum 60-s spectral radius | 1.000590684 |
| Negative-curvature configurations | 2,565 |
| Guarded maximum 60-s spectral radius | 1.000000000 |
| Guarded maximum singular value | 1.449185 |
| Guarded maximum 60-step transient growth | 2.625709 |
| Transient-growth limit | 10.0 |
| Minimum sampled stable-dt maximum | 150 s |
| Arrivals processed / contiguous updates | 250,000 / 246,297 |
| Market-gap resets | 3,703 |
| Negative-curvature projected replay steps | 153,863 |
| Numerical safety | Pass: finite state, no guarded spectral/transient rejection |
| Harmonic-model identification | Rejected: projection was required in most replay steps |

The pseudo-energy output remains diagnostic only; damping, moving equilibrium,
coupling, and resets make this a forced/dissipative system. Outputs are
`data_derived/integrator_stability.parquet`,
`integrator_replay_daily.parquet`, `integrator_gap_events.parquet`,
`integrator_summary.json`, and `integrator_checkpoint.json`.

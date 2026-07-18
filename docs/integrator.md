# Stability-Checked Integrator (OQ-8, OQ-9)

| Item | Accepted v1 contract |
|---|---|
| Owner | sim-integrator |
| Implementation | pipeline/simulate_integrator.py, integrator-1.1.0 |
| Executed command | python pipeline/simulate_integrator.py --max-steps 250000 |
| Scope | EURUSD, USDJPY, USDCNH only; indices [0,1,5] |
| Inputs | Canonical BID parquet, the three accepted dynamics streams, and the corresponding 3x3 submatrix of accepted daily C |
| Non-claim | This is not a 10-pair simulation: the other seven dynamics streams have not been estimated. |

## Discrete state update

The script uses semi-implicit Euler: damping is implicit and position uses the
new velocity. At a contiguous 60-second bar, use causal zero-order-held
parameters at time t:

    g(t) = 60 * v_hat(t)
    A_c(t) = C(t) * g(t)                         # specific acceleration
    kappa_raw(t) = k(t) / m(t)
    gamma(t) = c(t) / m(t)
    kappa_sim(t) = max(kappa_raw(t), 0)          # explicit stability guard

    v_hat(t+60) =
     [v_hat(t) + 60 * (-kappa_sim(t)*(x_hat(t)-x_eq(t)) + A_c(t))]
     / [1 + 60*gamma(t)]

    x_hat(t+60) = x_hat(t) + 60*v_hat(t+60)

C is not divided by mass. It was fitted as the map from g to specific
acceleration. This matches the source-schema physical equation because the
physical coupling force is m*A_c.

The replay uses epsilon=0: dynamics residual scale was fitted before coupling
was conditioned into the residual, so adding sampled forcing here would risk
double counting cross-pair effects. This is a deterministic, conservative
stability test rather than a jointly calibrated stochastic simulator.

## OQ-8 gap policy

At arrival of a canonical bar, calculate dt_observed=t_arrival-t_previous.

| Condition | Action |
|---|---|
| dt_observed = 60 s and guarded state is stable | Take the update above. |
| dt_observed != 60 s | Do not take a large step. Set x_hat=x_observed, v_hat=0, record RESET_TO_OBSERVED_POSITION_ZERO_VELOCITY, and resume only when a later contiguous bar arrives. |
| Guarded amplification unstable | Do not clip coupling or integrate. Hold to the arriving observed state and record REJECT_UNSTABLE_CONFIG_HOLD_OBSERVED. |

This policy is causal: the reset is made when the next bar reveals the gap,
not from knowledge of a future missing bar. No velocity is fabricated across
weekend, session, or intra-session gaps.

## OQ-9 stability test and signed-curvature guard

For each input-configuration change, the script builds the 6x6 linear update
matrix for [x_hat, v_hat] and computes rho(A), its spectral radius. The
numerical acceptance criterion is rho(A) <= 1 + 1e-10. It additionally
brackets a stable timestep by bisection, capped at 3600 seconds.

Raw fitted kappa can be negative. That is retained in
dynamics_params_<PAIR>.parquet and the raw spectral result is reported. A
negative-curvature state is not a stable spring, so the integrator's explicit
simulation guard is kappa_sim=max(kappa_raw,0). Every applied guard is
recorded as PROJECT_NEGATIVE_KAPPA_TO_ZERO; coupling is never suppressed.
If the guarded update were still unstable, it would be rejected rather than
integrated.

## Checkpoint and resume

data_derived/integrator_checkpoint.json stores pair scope, next canonical row
index, last timestamp, x_hat, and v_hat. --resume validates that contract
against the canonical timestamp, restores state, then reloads parameter and
coupling values by causal timestamp. It never infers a gap-crossing velocity.

## Executed result

| Metric | Result |
|---|---:|
| Stability configurations | 3,850 |
| Raw maximum rho(A) at 60 s | 1.000590684 |
| Raw 60-s pass | No; signed-curvature configurations require the explicit guard |
| Negative-curvature configurations projected | 2,565 |
| Guarded maximum rho(A) at 60 s | 1.000000000 |
| Guarded 60-s pass | Yes |
| Minimum guarded stable-dt lower bound | 179.088 s |
| Real canonical rows replayed | 250,001 |
| Continuous 60-s updates | 246,297 |
| Market-gap resets | 3,703 |
| Logged negative-curvature projections in replay | 155 configurations / 153,863 steps |
| Integrated unstable steps | 0 |
| Finite-state check | Pass |
| Replay span | 2015-01-30 21:51 UTC to 2015-10-20 08:14 UTC |

The pseudo-energy diagnostic is recorded per day, but is not treated as
conserved: moving equilibrium, damping, coupling, and causal resets make this
a forced/dissipative system. Stability is instead evidenced by the guarded
spectral-radius bound, finite-state check, explicit guard events, and bounded
state norm.

Outputs are data_derived/integrator_stability.parquet,
integrator_replay_daily.parquet, integrator_gap_events.parquet,
integrator_summary.json, and the checkpoint above.

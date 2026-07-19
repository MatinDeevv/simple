# Causal Single-Particle Dynamics (OQ-1 to OQ-5a)

| Item | Accepted v1 definition |
|---|---|
| Owner | sim-dynamics |
| Canonical input | `data/canonical/<PAIR>.parquet` only; `timestamp, close` |
| Estimator | `engine/models/classical/estimate_dynamics.py`, `dynamics-est-1.2.0` |
| Accepted run | `python engine/models/classical/estimate_dynamics.py --pairs EURUSD USDJPY USDCNH` |
| Outputs | `data/derived/dynamics_params_<PAIR>.parquet`, `data/derived/dynamics_summary.json` |
| Time convention | UTC bar close; nominal `dt=60 s`; log BID price in nats |

The fitted per-pair equation excludes the coupling term, which is owned by `docs/coupling.md`:

```text
a_i(t) = -kappa_i(t)d_i(t-1) - gamma_i(t)v_i(t-1) + eps_i(t)
k_i(t) = m_i(t)kappa_i(t);  c_i(t) = m_i(t)gamma_i(t);  F_i(t) = m_i(t)eps_i(t)
```

## Computable definitions

| OQ | Field | Definition |
|---|---|---|
| OQ-1 | Mass `m_i(t)` | Define eligible `r(t)=x(t)-x(t-60 s)` only when the observed predecessor is exactly 60 s earlier. `sigma_hat^2(t)` is the wall-clock EWMA of eligible `r(t)^2` through `t`, half-life 432000 s. `m(t)=clip((1e-4/sigma_hat(t))^2, 0.1, 10.0)`. It is a dimensionless inverse-realized-volatility proxy; it uses no volume or OQ-10 normalization. |
| OQ-2 | Equilibrium `x_eq,i(t)` | Wall-clock EWMA of `x(t)=ln(close_bid(t))` over all observations through `t`, half-life 86400 s. `d(t)=x(t)-x_eq(t)`. |
| OQ-3 | Spring `k_i(t)` | On trailing 43200 rows, retain valid two-step observations and require at least 20000. Fit no-intercept OLS `a(t)=b1*d(t-1)+b2*v(t-1)+eps(t)`. Set `kappa=-b1`; `k=m*kappa`. |
| OQ-4 | Damping `c_i(t)` | From the same regression set `gamma=-b2`; `c=m*gamma`. This is a conditional acceleration/return-decay coefficient, **not true spread or microstructure friction**: BID-only data have neither ask prices nor spread. |
| OQ-5a | Structural forcing `F_i(t)` | Estimation: `F=m[a+kappa*d(t-1)+gamma*v(t-1)]`. Simulation: `F=m*sigma_eps(t)*sqrt(s(how(t)))*eta`. `sigma_eps^2` is causal 5-day EWMA residual variance; `s(how)` is a strictly-prior hour-of-week variance ratio clipped to [0.1,10]. Fit `nu_fit` to standardized residuals; use `nu_sim=max(nu_fit,2.1)` and `eta=sqrt((nu_sim-2)/nu_sim)*t_nu_sim` for unit finite variance. The floor flag is emitted; it prevents undefined variance when `nu_fit<=2`. |

## Causality and gap scope

`v(t)=[x(t)-x(t-60 s)]/60` and `a(t)=[v(t)-v(t-60 s)]/60` are backward differences. A regression/residual observation is eligible only when both backward intervals are observed 60-second intervals. Gap-crossing rows are excluded from the regression, volatility estimator, residual fit, and seasonal fit; no velocity or acceleration is invented across a gap.

Holding the most recent valid volatility estimate is only an estimator-state convention. It does not select OQ-8 integrator behavior for session or weekend gaps. Every time-t regression window ends at t; regressors are `d(t-1)` and `v(t-1)`. Equilibrium, volatility, and residual scale use no time later than t. The seasonal numerator and denominator use strictly prior observations in the same hour-of-week bin. No centered window, future normalization, or full-sample fit feeds a time-t output.

## Accepted empirical run

Run completed `2026-07-18T03:28:58Z`. All daily parameter rows are finite, chronologically ordered, and have `m in [0.1,10]`. Values are `p5 / p50 / p95`; `k*dt^2` and `c*dt` use `dt=60 s`.

| Pair | Bars / daily rows | `k` (s^-2) | `c` (s^-1) | `k*dt^2` | `c*dt` |
|---|---:|---|---|---|---|
| EURUSD | 3,727,366 / 3,110 | `-2.657e-8 / 1.552e-8 / 1.090e-7` | `3.522e-3 / 1.185e-2 / 2.816e-2` | `-9.564e-5 / 5.586e-5 / 3.924e-4` | `0.211 / 0.711 / 1.689` |
| USDJPY | 3,726,492 / 3,110 | `-3.109e-8 / 9.473e-9 / 1.070e-7` | `2.683e-3 / 1.042e-2 / 2.786e-2` | `-1.119e-4 / 3.410e-5 / 3.851e-4` | `0.161 / 0.625 / 1.672` |
| USDCNH | 3,606,202 / 3,092 | `-1.112e-7 / 5.859e-8 / 6.925e-7` | `8.716e-3 / 3.265e-2 / 1.093e-1` | `-4.002e-4 / 2.109e-4 / 2.493e-3` | `0.523 / 1.959 / 6.561` |

Negative `k` estimates are retained as empirical output, not called stable springs. OQ-9 must use upper absolute stiffness and damping ranges, handle signed curvature, and reject unstable updates.

### Residual force `F` distribution (nats s^-2)

Kurtosis is excess kurtosis. These are estimation-residual diagnostics, not a claim of independent or Gaussian forcing.

| Pair | Residual n | `F` p5 / p50 / p95 | Mean / std | Skew / excess kurtosis | `nu_fit -> nu_sim` |
|---|---:|---|---|---|---|
| EURUSD | 3,688,524 | `-3.517e-8 / 3.076e-12 / 3.502e-8` | `-9.124e-12 / 2.440e-8` | `0.115 / 71.972` | `3.420 -> 3.420` |
| USDJPY | 3,686,586 | `-3.384e-8 / 7.914e-12 / 3.383e-8` | `1.746e-11 / 2.376e-8` | `-0.319 / 61.343` | `3.084 -> 3.084` |
| USDCNH | 3,537,448 | `-5.942e-8 / -9.266e-13 / 5.922e-8` | `-2.597e-11 / 4.312e-8` | `0.330 / 101.019` | `1.856 -> 2.100`; floor applied |

The spring term adds little in-sample incremental uncentered R2 versus the damping-only regression: EURUSD `4.302e-5`, USDJPY `7.953e-5`, USDCNH `8.894e-5`. This is not predictive evidence; it is a strictly causal structural residualization convention.

## Integrator handoff

The daily streams provide `m, k, c, k*dt^2, c*dt, sigma_eps`, and valid-window count. This range table plus `data/derived/dynamics_summary.json` are OQ-9 inputs. No integrator gap policy, stability conclusion, or learned residual/controller is decided here.

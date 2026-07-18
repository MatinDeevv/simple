# State-Vector Schema — Source of Truth

| | |
|---|---|
| **Version** | 1.5 |
| **Status** | DRAFT — open items assigned, see §7 |
| **Owner** | Chief Systems Architect |
| **Date** | 2026-07-18 |
| **Scope** | Canonical per-instrument state, update order, coupling block, integration contract |
| **Non-goals (v1)** | Ask-side reconstruction, synthetic spreads, tick/order-flow modeling, execution/portfolio layer, intraday session modeling beyond gap flag |

Rule of this document: every physics term is paired with a literal computable definition, or is marked **OPEN** with an owning agent. No metaphor survives without math.

---

## 1. Data ground truth (validated — do not re-verify)

| Item | Value |
|---|---|
| Instruments | 10 FX pairs (see §2 index table) |
| Source | Dukascopy 1-minute **BID** bars, one CSV per pair in `dukascopy_data/` |
| Columns | `timestamp,open,high,low,close,volume,symbol` |
| Timezone | UTC (e.g. `2015-02-01 22:00:00+00:00`) |
| Range | 2015-01-01 22:00 → 2024-12-31 21:59 |
| Rows | ~3.6–3.73M per pair (~99% minute coverage in market hours) |
| Quality | Sorted chronologically, zero duplicate timestamps, zero NaN closes (post-validation) |
| Gaps | Missing minutes exist (no ticks → no bar); weekend gaps ~48 h guaranteed |
| Volume | Dukascopy tick volume; float, broker-relative units, **not** real market volume |

**Hard constraints (binding on all agents):**

| ID | Constraint | Consequence |
|---|---|---|
| HC-1 | BID only — no ask, no spread column | Any spread-based quantity (friction/damping proxy, cross-arb residual floor) is unavailable from raw data |
| HC-2 | No tick/quote-level order flow | Forcing `F` cannot be defined from flow; must be proxy or learned residual |
| HC-3 | `dukascopy_multi.py` re-runs append chunks **UNSORTED** | Pipeline must NEVER assume raw CSVs sorted. Only the sim-datapipe canonical output stream is sorted/deduped. Raw CSVs are not a valid input to any downstream agent. |

---

## 2. Instrument index (canonical, frozen)

Index order is load-bearing for the coupling matrix `C`. Do not reorder.
`config/instruments.json` is the tracked source of this order. The generated
`data_canonical/manifest.json` must record the identical order and per-pair
indices, but it is data provenance rather than application configuration.

| i | Symbol | | i | Symbol |
|---|---|---|---|---|
| 0 | EURUSD | | 5 | USDCNH |
| 1 | USDJPY | | 6 | USDCHF |
| 2 | GBPUSD | | 7 | EURGBP |
| 3 | AUDUSD | | 8 | EURJPY |
| 4 | USDCAD | | 9 | GBPJPY |

---

## 3. Units convention (frozen)

| Quantity | Unit |
|---|---|
| Position (log-price) | nats (dimensionless; natural log of bid quote) |
| Time | seconds |
| Mass | dimensionless, cross-sectionally O(1) by convention |
| Velocity | nats·s⁻¹ |
| Momentum | nats·s⁻¹ (mass dimensionless) |
| Force | nats·s⁻² |
| Spring constant k | s⁻² |
| Damping c | s⁻¹ |
| Coupling C_ij | s⁻²; `g_j=60 s·v_j` is in nats |

---

## 4. Canonical state vector

Per instrument `i ∈ {0..9}` per timestep `t` (bar close time). Full state:
`S(t) = { core_i(t) : i=0..9 } ∪ C(t) ∈ ℝ^{10×10} ∪ bookkeeping(t)`

Governing structural equation (definitions below; estimation vs. simulation modes in §5):

```
m_i ẍ_i = −k_i (x_i − x_eq,i) − c_i ẋ_i + m_i Σ_j C_ij · g_j + F_i
```

### 4.1 Core fields (per instrument)

| Field | Symbol | Units | Computable definition | Status | Owner |
|---|---|---|---|---|---|
| Position | `x_i(t)` | nats | `ln(close_bid_i,t)` | **CLOSED** | — |
| Velocity | `v_i(t)` | nats·s⁻¹ | `(x_i(t) − x_i(t−1)) / Δt(t)` for an observed 60-s predecessor. When an arriving bar reveals `Δt≠60 s`, simulator state resets to observed `x` and `v=0`; no gap-crossing velocity is invented. | **CLOSED** (OQ-8) | sim-integrator |
| Mass | `m_i(t)` | dimensionless | `clip((1e-4/sigma_hat_i(t))^2, 0.1, 10)`, where `sigma_hat²` is the 5-day wall-clock EWMA of eligible 60-s squared log returns through `t`; no volume input. | **CLOSED** (OQ-1) | sim-dynamics |
| Momentum | `p_i(t)` | nats·s⁻¹ | `m_i(t) · v_i(t)` — derived, never independently stored/estimated | **CLOSED** formula; blocked by OQ-1 | — |
| Equilibrium | `x_eq,i(t)` | nats | Wall-clock EWMA of `x_i(τ≤t)=ln(close_bid_i,τ)`, half-life 86400 s. | **CLOSED** (OQ-2) | sim-dynamics |
| Spring constant | `k_i(t)` | s⁻² | `k=m*kappa`; `kappa=-b1` from causal no-intercept OLS of `a(t)=b1*d(t-1)+b2*v(t-1)` over trailing 43200 valid two-step rows, min 20000. | **ESTIMATOR DEFINED; identification OPEN** (OQ-3) | sim-dynamics |
| Damping | `c_i(t)` | s⁻¹ | `c=m*gamma`; `gamma=-b2` from the OQ-3 causal regression. It is a return-decay proxy, not spread or microstructure friction (HC-1). | **CLOSED** (OQ-4) | sim-dynamics |
| Forcing | `F_i(t)` | nats·s⁻² | Estimation: `F=m[a+kappa*d(t-1)+gamma*v(t-1)]`. Simulation structural spec: `F=m*sigma_eps*sqrt(s(how))*eta`, causal 5-day `sigma_eps²`, strictly-prior seasonal `s`, and finite-variance Student-t `eta` with `nu_sim=max(nu_fit,2.1)`. Learned residual remains OQ-5b. | **CLOSED** (OQ-5a); **OPEN** (OQ-5b) | sim-dynamics (structural), sim-neural (learned) |
| Coupling row | `C_i·(t)` | s⁻² | Row `i` of daily causal specific-acceleration `C(t)`; `C_ii=0`, directional rows allowed. `g_j(t)=x_j(t)-x_j(t-60 s)=60s·v_j(t)`. Physical coupling force is `m_i Σ_j C_ij g_j`. See §6. | **CLOSED** (OQ-6, OQ-7) | sim-coupling |

### 4.2 Bookkeeping fields (per timestep, per instrument)

| Field | Symbol | Units | Computable definition | Status |
|---|---|---|---|---|
| Bar timestamp | `t` | s (UTC epoch) | bar close time from canonical stream | CLOSED |
| Step size | `Δt(t)` | s | `t − t_prev` for this instrument; 60 nominal | CLOSED |
| Gap flag | `g_gap(t)` | bool | `Δt(t) ≠ 60` | CLOSED |

Derived quantities used only in estimation (never part of handed-off state): acceleration `a_i(t) = (v_i(t) − v_i(t−1)) / Δt(t)` — backward difference, needs `x` at `t, t−1, t−2`.

---

## 5. Update order (per timestep)

**Lookahead discipline (blanket rule):** every field indexed `t` must be computable from canonical bars at times `≤ t` only. Backward differences only — no centered differences. No full-sample statistics (z-scores, normalizations, fitted parameters) may touch data after `t`; estimators use rolling/expanding windows ending at `t`. sim-redteam owns leak detection (OQ-12).

| # | Step | Computed from | Lookahead note |
|---|---|---|---|
| 0 | (offline, once) sim-datapipe: sort + dedupe + validate raw CSVs → canonical bar stream | raw CSVs | HC-3: raw never assumed sorted |
| 1 | Read bar `t` for all instruments present at `t` | canonical stream | close known only at bar close |
| 2 | `Δt_i(t)`, `g_gap,i(t)` | timestamps ≤ t | — |
| 3 | `x_i(t) = ln(close_bid_i,t)` | bar t | — |
| 4 | `v_i(t)` backward difference | `x(t), x(t−1), Δt(t)` | for `Δt≠60`, causal reset to observed position and zero velocity; no large step |
| 5 | Parameter updates: `m_i(t), x_eq,i(t), k_i(t), c_i(t)` | history ≤ t | closed causal definitions in `docs/dynamics.md` |
| 6 | `p_i(t) = m_i(t) · v_i(t)` | steps 4, 5 | — |
| 7 | `C(t)` update | history ≤ t, all instruments | daily causal estimate in identity-free basis; §6 constraints apply |
| 8 | `F_i(t)`: **estimation mode** = `m_i[a_i + kappa_i d_i + gamma_i v_i − Σ_j C_ij g_j]` with backward `a_i`; **simulation mode** = structural OQ-5a or learned OQ-5b | steps 3–7 | residual uses only ≤ t |
| 9 | Assemble `S(t)`; hand to integrator → `x̂(t+Δt), v̂(t+Δt)` | full state | §7 contract |

Parameter hold policy between bars: zero-order hold — `m, x_eq, k, c, C` frozen at their `t` values for the step `t → t+Δt` (frozen decision D-006).

---

## 6. Coupling block C(t)

| Property | Convention |
|---|---|
| Shape | `C(t) ∈ ℝ^{10×10}`, row = affected instrument `i`, column = source instrument `j`, index order per §2 |
| Diagonal | `C_ii = 0` — self-dynamics live exclusively in `m, k, c`; no self-coupling |
| Symmetry | **NOT assumed.** Lead–lag is directional; `C_ij ≠ C_ji` allowed |
| Functional form | `g_j(t)=x_j(t)-x_j(t-60 s)=60s·v_j(t)` [nats]. `A_c,i=sum_j C_ij*g_j` is specific coupling acceleration [nats s⁻²], so physical coupling force is `m_i A_c,i`; `C_ij` is s⁻². |
| Estimation method | Daily, trailing 28800 valid synchronous two-step samples: fit in an identity-free transform basis, map to raw coordinates, and set `C_ii=0`; see `docs/coupling.md`. |

**Structural triangle constraint (binding on the estimator — OQ-7).** In log space the crosses are near-arithmetic identities of the majors:

| Identity | Relation (log bid, approximate) |
|---|---|
| T-1 | `x_EURGBP ≈ x_EURUSD − x_GBPUSD` |
| T-2 | `x_EURJPY ≈ x_EURUSD + x_USDJPY` |
| T-3 | `x_GBPJPY ≈ x_GBPUSD + x_USDJPY` |

These hold to within spread/microstructure noise (exact only for arb-free mid; we have bid only, HC-1). Consequence: near-deterministic correlations inside {0,2,7} and {0,1,2,8,9} are **arithmetic, not dynamics**. The coupling estimator must encode these as structural constraints (or estimate on an identity-free basis) and must not report them as discovered coupling. Any estimator whose headline output is T-1/T-2/T-3 has found nothing. sim-redteam validates (OQ-12).

---

## 7. Integration contract

| Item | Contract |
|---|---|
| Input | `S(t)` per §4: `{x, v, m, x_eq, k, c, F}` × 10, `C(t)` 10×10, bookkeeping `{t, Δt, g_gap}` × 10 |
| Parameters during step | Zero-order hold at time-t values |
| Output | `x̂_i(t+Δt), v̂_i(t+Δt)` for all i |
| dt | Nominal **60 s** |
| Gap handling | **CLOSED** (OQ-8): on arrival of a bar with observed `Δt≠60 s`, take no large step; reset `x_hat=x_observed`, `v_hat=0`, log the event, and resume only on a later contiguous 60-s bar. |
| Scheme + stability | **REOPENED** (OQ-9/D-023): semi-implicit Euler with damping implicit. Directional coupling is non-normal, so require spectral radius, singular value, finite-horizon power growth, eigenvector conditioning, and sampled timestep segments. `kappa_sim=max(kappa,0)` is a logged model projection, not proof that the signed harmonic model is accepted. See `docs/integrator.md`. |
| Live-mode caveat | At time `t` the arrival time of the next bar is unknown (missing minutes are not forecastable). Integrator must not require future `Δt` knowledge beyond the nominal 60 s step. |

---

## 8. Open questions

| ID | Question | Owner | Blocking |
|---|---|---|---|
| OQ-1 | **RESOLVED 2026-07-18** (D-008): inverse causal realized-volatility mass; see `docs/dynamics.md` | sim-dynamics | unblocks `p`, `F` magnitude, OQ-9 range analysis |
| OQ-2 | **RESOLVED 2026-07-18** (D-009): 24-h wall-clock causal EWMA; see `docs/dynamics.md` | sim-dynamics | unblocks spring displacement and `F` residual |
| OQ-3 | **REOPENED 2026-07-18** (D-022): signed-curvature estimator exists, but frequent negative `kappa` rejects harmonic restoring-force identification in those regimes; define and validate bounded nonlinear/regime/uncertainty alternatives without holdout selection | sim-dynamics | blocks physical interpretation and acceptance of the classical restoring-force simulator |
| OQ-4 | **RESOLVED 2026-07-18** (D-011): causal return-decay proxy `c=m*gamma`, explicitly not spread friction; see `docs/dynamics.md` | sim-dynamics | provides `F` residual and OQ-9 range input |
| OQ-5a | **RESOLVED 2026-07-18** (D-012): residual-derived, heteroskedastic finite-variance Student-t structural forcing; see `docs/dynamics.md` | sim-dynamics | structural simulation runs; OQ-5b remains open |
| OQ-5b | Learned residual/controller contract for `F̂` (interface, train/test split honoring §5 lookahead rule) | sim-neural | simulation runs, backtests |
| OQ-6 | **RESOLVED 2026-07-18** (D-013): daily causal identity-free coupling field; see `docs/coupling.md` | sim-coupling | unblocks step 7–8 and cross-asset simulation |
| OQ-7 | **RESOLVED 2026-07-18** (D-014): invertible residual basis enforces all three arithmetic identities; see `docs/coupling.md` | sim-coupling | validates coupling claims against arithmetic leakage |
| OQ-8 | **RESOLVED 2026-07-18** (D-016): causal reset-on-arrival for every non-60-s interval; see `docs/integrator.md` | sim-integrator | closed for the numerically safe three-pair replay |
| OQ-9 | **REOPENED 2026-07-18** (D-023): integrator now records non-normal transient diagnostics and avoids monotonic timestep bisection, but only numerical safety is established; see `docs/integrator.md` | sim-integrator | blocks acceptance of the three-pair harmonic simulator until transient/OOS/model-identification gates are defined |
| OQ-10 | Volume normalization across pairs/years (broker-relative units) — canonical normalized volume field spec | sim-datapipe | optional future feature; no longer blocks OQ-1 |
| OQ-11 | ~~Ingestion spec~~ **RESOLVED 2026-07-17** (D-007): see `docs/datapipe.md` — pipeline v1.0.0, canonical zstd parquet in `data_canonical/` + generated `manifest.json` content hashes; its instrument order is validated against tracked `config/instruments.json` | sim-datapipe | — |
| OQ-12 | Adversarial validation plan: lookahead-leak tests (§5 rule), triangle-identity leak test (§6), residual-`F` sanity bounds | sim-redteam | v1 sign-off |
| OQ-13 | **FROZEN** experimental quantum-software representations: density filter, qutrit trajectories/tomography, ten-qutrit MPS/TEBD, ten-qubit kernel/reservoir, and Aer synthetic-noise calibration; never a claim that FX is physically quantum; see `docs/quantum-frontier.md` and `docs/quantum-redteam.md` | Chief Systems Architect | noncanonical negative-results archive: every executed predictive representation loses to a required baseline or fails convergence; no new quantum model until shared target/classical/OOS gates pass |
| OQ-14 | **PARTIALLY IMPLEMENTED 2026-07-18** (D-026): causal FX residual-convergence arena with fixed target, gap policy, frozen train-prior Brier baseline, 2022/2023/2024 fold commands, and moving-block bootstrap; matched alternatives and all outer-fold evidence remain required | Chief Systems Architect | blocks any predictive model promotion, including a quantum-branch revisit |

---

## 9. Decision log

| # | Date | Decision |
|---|---|---|
| D-001 | 2026-07-17 | Schema v1 created. Data validated and sorted: 10 pairs, 1-min Dukascopy BID bars, UTC, 2015-01-01 22:00 → 2024-12-31 21:59, ~3.6–3.73M rows/pair, no duplicate timestamps, no NaN closes. Constraints recorded: **bid-only (no spread)** and **no order-flow data** — all damping/forcing proxies are assigned to owning agents (OQ-4, OQ-5), not invented here. |
| D-002 | 2026-07-17 | HC-3 recorded: downloader appends unsorted; raw CSVs are never a valid downstream input — only the sim-datapipe canonical stream is (OQ-11). |
| D-003 | 2026-07-17 | Units frozen (§3): log-price in nats, time in seconds, mass dimensionless O(1). |
| D-004 | 2026-07-17 | Instrument index order frozen (§2); load-bearing for `C`. |
| D-005 | 2026-07-17 | Coupling conventions frozen: `C ∈ ℝ^{10×10}`, zero diagonal, asymmetry allowed; triangle identities T-1..T-3 declared structural, not discoverable (§6). |
| D-006 | 2026-07-17 | Lookahead rule frozen (§5): all fields t-measurable, backward differences only, causal estimators only; zero-order hold on parameters within a step. |
| D-007 | 2026-07-17 | OQ-11 resolved: `pipeline/ingest.py` v1.0.0 (spec `docs/datapipe.md`). Canonical stream = per-pair zstd parquet in `data_canonical/` + `manifest.json` (source SHA256, `fxsim-canonical-v1` data content hash, counts). Idempotency verified across two runs; 37.1M rows, 0 dups, 0 OHLC violations. Step-0 input for all agents is now `data_canonical/`, never raw CSVs. Note for OQ-8/OQ-10 owners: wall-clock minute coverage ~70.9% (weekends ~28.6%; intra-session ~99%); USDCNH outlier 68.6%. |
| D-008 | 2026-07-18 | OQ-1 resolved from `docs/dynamics.md`: mass is clipped inverse causal realized volatility (`1e-4` reference 1-min scale, 5-day wall-clock EWMA), not raw or normalized broker volume. |
| D-009 | 2026-07-18 | OQ-2 resolved from `docs/dynamics.md`: equilibrium is the 24-hour wall-clock causal EWMA of BID log price. |
| D-010 | 2026-07-18 | OQ-3 estimator definition recorded from `docs/dynamics.md`: `k=m*kappa` from daily-sampled causal rolling no-intercept OLS; EURUSD/USDJPY/USDCNH range outputs recorded. D-022 later reopened model identification. |
| D-011 | 2026-07-18 | OQ-4 resolved from `docs/dynamics.md`: `c=m*gamma` is a causal return-decay proxy only. HC-1 remains binding: no spread/microstructure damping claim is made. |
| D-012 | 2026-07-18 | OQ-5a resolved from `docs/dynamics.md`: residual forcing has causal scale and seasonal modulation, with an explicit `nu>=2.1` simulation floor for finite Student-t variance. OQ-5b is not resolved. |
| D-013 | 2026-07-18 | OQ-6 resolved from `docs/coupling.md`: `g_j=x_j(t)-x_j(t-60s)` and daily causal 10x10 directional C matrices are produced from trailing valid synchronous samples. |
| D-014 | 2026-07-18 | OQ-7 resolved from `docs/coupling.md`: EURGBP, EURJPY, and GBPJPY are transformed to arithmetic-residual channels before fitting. Executed diagnostics show the expected pre-to-post reduction. |
| D-015 | 2026-07-18 | Coupling-unit contract reconciled: the fitted `C` maps `g` to specific acceleration. The governing equation and residual now multiply the coupling acceleration by `m`; this matches the executed estimator units. |
| D-016 | 2026-07-18 | OQ-8 resolved from `docs/integrator.md`: when a newly arriving bar exposes a non-60-s interval, no giant step is taken. The simulator resets to observed position with zero velocity and logs the event. |
| D-017 | 2026-07-18 | OQ-9 resolved from `docs/integrator.md`: semi-implicit Euler is accepted only after causal per-configuration amplification checks. Raw negative curvature remains reported; `kappa_sim=max(kappa,0)` is explicit and logged. The verified scope is EURUSD/USDJPY/USDCNH only. |
| D-018 | 2026-07-18 | OQ-13 added as an isolated research branch: `docs/quantum-experiment.md` uses a computable qutrit density-matrix filter with unitary, measurement, and Lindblad operations. It is explicitly not evidence of physical quantum FX behavior and does not alter canonical simulation state. |
| D-019 | 2026-07-18 | Quantum red-team result recorded. The density filter now has complete-instrument, gap-safe, minute-replay controls; the independent trajectory unraveling is numerically valid. Both fail their baseline/OOS diagnostic gates, so OQ-13 remains isolated and is not authorised to scale, control the simulator, or enter a trading path. |
| D-020 | 2026-07-18 | Explicit complexity stress test recorded. Ten-qutrit MPS/TEBD, exact ten-qubit fidelity-kernel, ten-qubit data-reupload reservoir, and qutrit process tomography were added as isolated artifacts. Tomography/numerical checks validate software mathematics only; all predictive branches remain rejected, with MPS additionally failing its truncation-convergence threshold. No canonical state, controller, or trading permission changes. |
| D-021 | 2026-07-18 | Isolated Qiskit Aer environment added for a declared synthetic ten-qubit density-matrix noise calibration. It reports ideal-versus-noisy observables only; no provider credentials, backend calibration, hardware job, forecasting target, or promotion permission was added. |
| D-022 | 2026-07-18 | OQ-3 reopened. The executed replay projects negative curvature in 2,565 of 3,850 configurations and 153,863 of 250,000 arrivals. This is a harmonic-model identification failure signal, not a numerical fix; no nonlinear potential has been selected. |
| D-023 | 2026-07-18 | OQ-9 reopened. Resume now has explicit `state_index`/`next_arrival_index` semantics and split-vs-resumed regression coverage. Directional-coupling stability now logs balanced-coordinate singular value, 60-step transient growth, eigenvector conditioning, sampled unit-circle resolvent sensitivity, and non-monotonic sampled dt segments; spectral radius alone is no longer an acceptance proof. |
| D-024 | 2026-07-18 | Pair order and 60-second gap semantics centralized in `pipeline/contracts.py`. Ten-pair kernel, MPS, reservoir, and Aer branches use the shared order contract; the quantum archive is frozen and OQ-14 is the required shared scoring gate before any predictive escalation. |
| D-025 | 2026-07-18 | CI bootstrap repair: tracked `config/instruments.json` is now the sole import-time source for the ten-pair order. Ingestion copies it into the generated manifest, and contract tests validate generated manifest order/index agreement without requiring the data lake. |
| D-026 | 2026-07-18 | OQ-14 causal FX regime/residual arena added in `pipeline/stat_arb.py`. It applies identity-free factor/graph estimation, a fixed causal regime filter, breakdown diagnostics, factor/net-neutral diagnostic weights, and a predeclared 30-minute residual-convergence Brier target. BID-only data blocks execution/PnL/capacity claims; the first bounded candidate failed its frozen-prior diagnostic and is not promotable. |

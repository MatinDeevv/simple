# Quantum-Trajectory Open-System Experiment

## Boundary and purpose

This isolated experiment uses real open-quantum-system mathematics to encode
the causal EURUSD, USDJPY, and USDCNH canonical observations. It does **not**
assert that FX markets are physical quantum systems, that market actors are
quantum objects, or that its output is a trading signal. It does not change
the canonical state schema, classical simulator, existing density-matrix
experiment, controller, or trading logic.

The quantum numerical core is NumPy only. Pandas is used solely to read the
existing local Parquet input and write result artifacts; no quantum framework
(Qiskit, PennyLane, or similar) is used.

## Qutrit and causal encoding

The computational basis is one basis state per pair:

    |0> = EURUSD, |1> = USDJPY, |2> = USDCNH.

At contiguous minute `t`, `z_t` is the return from `t-1` to `t`, divided by a
causal 1,440-step EWMA volatility. `C_t` is the most recent accepted coupling
matrix with timestamp less than or equal to `t`. Neither uses the next bar.
The trace-free Hermitian Hamiltonian is

    H_t = 0.08 [diag(clip(z_t,-5,5)) + 0.5 * 3600 * (C_t + C_t^T)]
    H_t <- (H_t + H_t-dagger)/2 - I trace(H_t)/3,
    U_t = exp(-i H_t).

The real symmetric construction makes `H_t=H_t-dagger`; `U_t` is obtained by
a Hermitian eigendecomposition and its unitarity is checked on every update.

## Lindblad bath and Monte Carlo wavefunctions

The environmental dephasing model has explicit Lindblad jump operators

    L_j = sqrt(gamma) |j><j|,   j in {0,1,2}
    gamma = -log(0.98)/60 per second.

Thus off-diagonal density terms contract by `q=exp(-gamma*60)=0.98` each
minute. The exact finite-step Kraus map after the unitary is

    K_0 = sqrt(q) U_t
    K_j = sqrt(1-q) |j><j| U_t.

`sum_a K_a-dagger K_a=I`. Each of the 96 seeded trajectories follows this
unravelling: first `|psi'>=U_t|psi>`, then no jump occurs with probability
`q`; otherwise it collapses to `|j>` with Born probability
`|<j|psi'>|^2`. Every resulting state vector is normalized. The reconstructed
density is

    rho_t = (1/N) sum_n |psi_t^(n)><psi_t^(n)|.

There is no conditional market-data measurement here. Returns only parameterize
the causal Hamiltonian. The random jump is an unobserved environmental bath
sample in a Lindblad unraveling; it is not an observed FX-market measurement,
an execution event, or an assertion about market ontology.

At initialization and after a gap, equally allocating normalized trajectories
to the three basis states reconstructs exactly `rho=I/3`.

## Gap policy and diagnostics

Any non-60-second interval is a boundary. The experiment does not update or
score a prediction across it. At the first bar after the boundary it resets
the ensemble to `I/3` and restarts the causal volatility EWMA. This is a
statistical representation reset, not a measurement of market decoherence.

The reported next-bar result is a fixed chronological final-30% diagnostic:
at time `t`, the Born diagonal of `rho_t` predicts which pair will have the
largest absolute valid return from `t` to `t+1`. The target is never used in
the state update. This is out-of-sample relative to the preceding 70% of the
run but is still non-trading and non-promoted: it is not cost adjusted, not a
claim of causal efficacy, and not evidence of market quantum ontology.

Before the holdout begins, the runner freezes three baselines using only the
first 70% target labels: uniform probabilities, a one-hot constant predictor
for the pre-holdout modal class, and the pre-holdout empirical class-prior
probabilities. The final 30% report includes all three alongside the model.

## Checks and execution

The self-check verifies Hermiticity, unitarity, Kraus completeness, zero trace
of the Lindblad generator, MC reconstruction against the exact density map,
positive reconstructed density, normalized vectors, and same-seed bitwise
reproducibility.

    python engine/quantum/quantum_trajectories.py --self-check
    python engine/quantum/quantum_trajectories.py --max-steps 250000 --trajectories 96 --seed 20260718

The second command writes the executed results to:

- `data/derived/quantum_trajectories_daily.parquet`
- `data/derived/quantum_trajectories_minute.parquet`
- `data/derived/quantum_trajectories_summary.json`
- `data/derived/quantum_trajectories_validation.json`

The minute artifact is the reproducibility surface: every processed source row
has timestamps, gap/skip reason, reset flag, raw and standardized causal
returns, causal coupling timestamp/age/full 3x3 submatrix, Born predictions,
next-bar target, jump count, and density purity. Rows skipped at a gap retain
their reason code and do not have a fabricated return or prediction.

## Executed canonical run

The documented command was executed with the fixed seed on the three-pair
canonical slice from 2015-01-30 21:51 UTC through 2015-10-20 08:14 UTC.

| Metric | Result |
|---|---:|
| Source rows / valid updates | 250,000 / 243,208 |
| Gap ensemble resets | 3,703 |
| Trajectories / seed | 96 / 20260718 |
| Observed jump probability | 1.9985% (configured 2.0000%) |
| Maximum density trace error | 1.998e-15 |
| Maximum density Hermiticity error | 0 |
| Minimum reconstructed density eigenvalue | 0.186212 |
| Maximum normalized-state error | 1.110e-15 |
| OOS samples (final 30%) | 72,016 |
| Trajectory top-1 / Brier | 37.8166% / 0.668522 |
| Uniform expected top-1 / Brier | 33.3333% / 0.666667 |
| Frozen EURUSD-constant top-1 / Brier | 50.0611% / 0.998778 |
| Frozen empirical-prior top-1 / Brier | 50.0611% / 0.602831 |

The experiment is **not promoted**: although its top-1 figure exceeds the
uniform expectation, it loses to the frozen modal-class baseline on top-1 and
is worse than both uniform and frozen-prior probabilities on Brier score.
This is an experimental diagnostic result, not a trading result.

## Complexity trade-off

For qutrit dimension `d=3`, the density-matrix filter stores one `d x d`
matrix and costs `O(d^3)` per update. The trajectory method stores `N` pure
vectors and costs `O(N d^2 + d^3)` per update, with stochastic Monte Carlo
error roughly `O(N^-1/2)`. The 96-trajectory run is therefore intentionally
more expensive and noisier than the direct density filter at this tiny
dimension. Its value is architectural: it is a valid path to larger
open-system models or measurement records where direct density matrices scale
as `d^2` storage and `d^3` operations. It is not currently a performance or
trading upgrade.

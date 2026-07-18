# Open-Quantum-System FX Experiment

## Boundary

This uses quantum-mechanical numerical objects: a density matrix, Hermitian
Hamiltonian, unitary evolution, a complete Kraus instrument, and a pure
dephasing channel. It is a data-conditioned statistical representation of
classical BID bars. It is not evidence that currency pairs are physical
quantum systems, that markets exhibit entanglement, or that the output is a
trading signal.

The experiment is isolated from the canonical simulator: it cannot change the
accepted state vector, forcing process, coupling matrix, controller, or any
trading path.

## Computable qutrit filter

The qutrit basis is a forced categorical encoding, not three mutually
exclusive market states:

    |0> = EURUSD, |1> = USDJPY, |2> = USDCNH.

At a valid contiguous minute `t`, `z_t` is the causal EWMA-standardized
three-pair return and `C_t` is the latest causal three-by-three coupling
submatrix. The trace-free Hermitian Hamiltonian is

    H_t = 0.08 * [diag(clip(z_t,-5,5)) + 0.5 * 3600 * (C_t + C_t^T)]
    H_t <- (H_t + H_t-dagger)/2 - I*trace(H_t)/3
    U_t = exp(-i H_t)
    rho <- U_t rho U_t-dagger

The conditional evidence branch is embedded in a complete two-outcome
instrument. With `r=clip(z_t,-5,5)`,

    e_i = exp(0.20 * (r_i - max(r)))
    K0 = diag(sqrt(e_i));  K1 = diag(sqrt(1-e_i))
    K0-dagger K0 + K1-dagger K1 = I
    rho <- K0 rho K0-dagger / trace(K0 rho K0-dagger)

`K0` is the branch used by this likelihood filter. The contemporaneous
classical return selects its effect, so this is mathematically a valid
instrument but not a market measurement correspondence. Pure dephasing then
applies the exact finite-step channel

    rho <- 0.98*rho + 0.02*diag(diag(rho)).

Born probabilities are `diag(rho)`. The use of a symmetric coupling is
necessary for a Hermitian Hamiltonian and deliberately discards directional
lead-lag semantics; those semantics remain only in the classical coupling
branch.

## Causality, gaps, and artifacts

The state at `t` uses only the return from `t-1` to `t` and a coupling update
whose timestamp is no later than `t`. A non-60-second arrival resets both
`rho` and the EWMA state; the first post-gap row is logged and skipped before
a return can be formed. A row whose next bar is noncontiguous is also skipped.
Coupling older than 36 hours causes a reset and skip rather than a silent stale
state update. The runner derives `cross_gap_state_updates` from the persisted
minute rows (`reason=scored_contiguous` with a false predecessor-contiguity
flag) and rejects any nonzero count.

`quantum_lindblad_minutes.parquet` is the replay surface: all source rows have
their timestamp, reason code, coupling timestamp/age, full three-by-three
matrix, causal standardized inputs, predictions when scored, next-bar target,
and density/instrument checks. Source and implementation SHA-256 hashes are
in the JSON summary.

The only diagnostic target is the pair with the largest absolute *next valid*
60-second log return. It is calculated after state update and never feeds back.
This target is a forced categorical classifier and has no identified physical
observable.

## Executed result

Command:

    python pipeline/quantum_lindblad.py --self-check
    python pipeline/quantum_lindblad.py --max-steps 250000

| Metric | Result |
|---|---:|
| Scope / span | EURUSD, USDJPY, USDCNH / 2015-01-30 21:51 to 2015-10-20 08:14 UTC |
| Source rows / scored contiguous updates | 250,000 / 234,825 |
| Gap resets / cross-gap state updates | 3,703 / 0 |
| Next-bar-gap skips / stale-coupling skips | 3,089 / 8,383 |
| Maximum coupling age observed | 270,660 s (logged; updates above 129,600 s skipped) |
| Maximum trace / Hermiticity / instrument-completeness error | 2.220e-16 / 0 / 2.220e-16 |
| Minimum density eigenvalue | 2.498e-7 |
| Model top-1 / Brier | 37.842% / 0.787543 |
| Uniform Brier | 0.666667 |
| Causal Laplace-prior top-1 / Brier | 57.792% / 0.540073 |
| Causal last-magnitude top-1 / Brier | 51.432% / 0.971362 |

The model fails its causal baseline comparison and is not promoted. It also is
only a first-segment diagnostic, not a frozen chronological OOS study. Its
result must not motivate a larger Hilbert space, controller, neural layer, or
trading claim.

## Promotion gate

Before any escalation, freeze a chronological train/validation/untouched-OOS
protocol; persist per-minute emissions and hashes; compare against a matched
classical state filter, constant/prior, and causal volatility baselines; and
pass gap/regime, leakage, and placebo red-team tests. The shared complexity
and OOS requirements are defined in `docs/quantum-frontier.md`.

Outputs: `data_derived/quantum_lindblad_daily.parquet`,
`data_derived/quantum_lindblad_minutes.parquet`, and
`data_derived/quantum_lindblad_summary.json`.

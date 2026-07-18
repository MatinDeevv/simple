# Quantum Branch Red-Team Decision

## Decision: reject promotion; permit only isolated numerical research

The quantum branch is not a model of physical market quantum mechanics. It is
software that applies open-quantum-system mathematics to a chosen encoding of
classical price data. Neither numerical validity nor a predictive score can
establish physical quantumness, entanglement of currency pairs, or a quantum
trading advantage.

## Findings and disposition

| Severity | Finding | Disposition |
|---|---|---|
| Critical | The original Lindblad run formed a multi-hour/weekend log return at the first post-gap bar. | Fixed in `quantum-lindblad-1.1.0`: reset density and EWMA, skip, and assert zero cross-gap state updates. Fresh replay: 3,703 reset/skip rows, all with null `z` and prediction values. |
| Critical | The original diagonal update was normalized but did not state a complete generalized-measurement instrument. | Fixed as an explicit `K0`, `K1` instrument with `K0-dagger K0 + K1-dagger K1 = I`; maximum completeness error is `2.220e-16`. It remains a data-conditioned likelihood choice, not a market measurement model. |
| Critical | The original 35.556% top-1 diagnostic was compared only with uniform even though EURUSD is the dominant raw-return winner. | Fixed as a reported failure, not optimized away. Updated Lindblad is 37.842% top-1 / 0.787543 Brier, behind a causal class prior at 57.792% / 0.540073. |
| High | Daily snapshots could not reproduce minute-level targets, predictions, coupling age, or gap handling. | Fixed: `quantum_lindblad_minutes.parquet` contains 250,000 source rows, reason codes, inputs, emissions, targets, coupling time/age/full submatrix, and numerical checks. |
| High | Qutrit basis probabilities were at risk of being read as market states or observed physical outcomes. | Permanently constrained in the documentation: the basis is a forced categorical encoding, and the next-bar winner is an unpromoted classifier target without a physical observable interpretation. |
| Medium | Coupling was symmetrized and could be held stale without telemetry. | Symmetrization is documented as discarding directional lead-lag semantics. Coupling age is now logged; values older than 36 hours reset and skip rather than updating the state. |
| High | A higher-order trajectory method could appear to justify escalation merely by using more formal machinery. | The 96-trajectory result is valid numerically but fails its fixed final-30% diagnostic: 37.817% / 0.668522 versus frozen prior 50.061% / 0.602831. It is explicitly not promoted. |

## Current evidence

Both numerical implementations pass the relevant trace, Hermiticity,
positivity, and channel/instrument checks. That is necessary numerical hygiene;
it is not empirical support for forecasting or market ontology.

- Direct qutrit filter: `pipeline/quantum_lindblad.py`; its 250,000-row replay
  has zero cross-gap state updates but fails causal baseline comparison.
- Monte Carlo wavefunction qutrit: `pipeline/quantum_trajectories.py`; its
  seed replay and exact-channel checks pass, but it loses OOS to frozen
  constant/prior probability baselines.

## Non-negotiable gate before any Tier-1+ work

1. Precommit target, pair order, feature map, hyperparameters, split dates,
   score, and a matched classical comparator before opening the test fold.
2. Use three untouched chronological outer folds (2022, 2023, 2024), with
   tuning confined to the preceding data and valid contiguous next-bar rows
   scored separately from reset-adjacent behavior.
3. Persist every emission, target, input/coupling timestamp, artifact hash,
   and RNG seed. Prove that changing data after a cutoff cannot change earlier
   emissions.
4. Beat constant, causal prior, causal volatility, and equal-capacity classical
   state filters on the declared proper score in every outer fold, with a
   positive 95% moving-block-bootstrap lower bound for improvement.
5. Beat at least 100 seeded circular-block feature-stream placebos; otherwise
   mark the branch `experimental rejected`.

No higher Hilbert-space dimension, multi-pair entanglement representation,
neural quantum controller, package installation, hardware account, or trading
integration is authorised until those gates pass. The escalation map and
resource ceilings are in `docs/quantum-frontier.md`.

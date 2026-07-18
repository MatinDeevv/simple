# Quantum Research Complexity Frontier

## Decision

The quantum branch is **frozen** as a negative-results archive. It contains
qutrit density/trajectory filters, ten-qutrit MPS/TEBD dynamics, an exact
ten-qubit fidelity kernel, a ten-qubit re-uploading reservoir, process
tomography, and Aer noise calibration. These artifacts remain reproducible but
no new quantum representation may be added until the shared target, classical
comparators, and untouched-fold protocol in OQ-14 are satisfied.

This document uses *quantum* in a narrow, literal sense: software evolves
complex state vectors or density operators under quantum-mechanical maps. It
does **not** claim that the FX market is a quantum physical system. The inputs
are classical Dukascopy BID bars, not prepared quantum states; the observation
map is chosen by the modeller; and forecasting lift cannot identify a physical
ontology. In particular, correlation, coherence in a fitted `rho`, or a
better score is not entanglement, interference in the market, or quantum
advantage. Establishing a physical claim would require a physical subsystem,
state-preparation and measurement correspondence, controlled interventions,
and nonclassical tests. Those data do not exist here and that claim is out of
scope permanently for this project.

## Current evidence and hard boundary

| Item | Observed state on 2026-07-18 | Consequence |
|---|---|---|
| Implemented branch | Direct qutrit filter, 96-seed qutrit unraveling, ten-qutrit MPS/TEBD trajectory, exact ten-qubit fidelity kernel, fixed ten-qubit reservoir, qutrit channel tomography, and ten-qubit Aer noise calibration | All are noncanonical, numerically constrained research branches |
| Executed artifacts | Lindblad: 250,000 rows / 234,825 scored / zero cross-gap updates; trajectories: 243,208 valid updates; MPS: 500-row `chi=16` replay; kernel: 250,000-row audit with 384/192 bounded train/OOS samples; reservoir: 50,000 rows with 34,448/14,657 train/OOS samples; tomography: 481 sampled CPTP channels; Aer: 24 causal circuits x 4,096 shots | Numerical checks pass where applicable, but every predictive branch fails its promotion gate |
| Scientific stack | Core: NumPy/pandas/pyarrow. Isolated `.venv-quantum`: Qiskit 2.4.2 and Aer 0.17.2 | Exact statevectors, bounded MPS, and declared noisy density-matrix circuit simulation are possible locally |
| Absent packages | PennyLane, Cirq, quimb, TeNPy, Qiskit Runtime, Braket, Azure Quantum, and all provider credentials/backends are absent | Alternative framework and real-hardware tiers remain blocked; Aer's synthetic noise model is not a backend calibration |
| Hardware access | No credential/configuration artifact for IBM, Braket, Azure, or another quantum provider was found | Hardware execution is blocked |
| Integration status | OQ-13 is frozen and noncanonical; the state schema bars it from changing the classical simulator, controller, or trading logic | Preserve the archive; advance OQ-14/core dynamics instead |

The updated Lindblad result is 37.842% top-1 with Brier 0.787543, versus a
causal class-prior's 57.792% / 0.540073 on the same first segment. The
trajectory final-30% diagnostic is 37.817% / 0.668522, versus its frozen class
prior's 50.061% / 0.602831. The ten-qutrit MPS reaches 6.040% / 0.904375
against uniform 10% / 0.900000 on its bounded OOS slice; the exact kernel is
18.750% / 0.889459 against frozen prior 32.292% / 0.849528; and the reservoir
is 23.313% / 0.893500 against frozen prior 23.395% / 0.871358. None is
evidence against a matched classical model and none may select a larger model.

## Non-negotiable data and validation contract

Every tier must implement the same contract before model-specific work:

```text
input at bar close t:
  tracked instrument config plus canonical manifest/data hash; fixed pair-order; UTC timestamp
  x_t, valid 60-second return z_t, gap flags, and only C(u) with u <= t
  all normalizers and fitted parameters trained strictly before the scored fold

state transition:
  reset on a non-60-second arrival exactly as the current experiment does
  update the quantum state with values available at t only

emission:
  prediction_time=t; target_time=t+60 s; probability vector / forecast;
  trace, Hermiticity, positivity proxy, reset flag, model/config hash, RNG seed
```

The target, loss, pair order, feature map, gap treatment, and split dates must
be committed before the outer holdout is read. A valid comparison is a
chronological expanding-window study with three untouched outer folds (test
years 2022, 2023, and 2024), with all tuning done only before each fold. Score
only valid contiguous next bars, report reset-adjacent bars separately, and
report each of: the full test year, high/low realized-volatility halves defined
from past-only data, and each pair or class. The matched classical comparator
uses the identical causal inputs and target, has comparable output dimension,
and is selected only from the corresponding training/validation period.

A tier is **statistically accepted** only when all three outer folds have both
of the following on the predeclared primary proper score (Brier for a
categorical probability output; log score if that is the predeclared output):

1. `Delta = score_classical - score_quantum` has a 95% moving-block-bootstrap
   lower confidence bound greater than zero. Blocks are one trading day and
   are resampled within the scored contiguous segments.
2. The direction remains non-negative in every required regime slice and the
   median slice delta is positive.

Before this gate, run two red-team checks: (i) change all data after a cutoff
and prove byte-identical emissions at or before that cutoff; (ii) run at least
100 seeded circular-block placebos of the causal feature stream and show that
the real improvement exceeds the 95th percentile placebo improvement. Failing
physics, causality, placebo, or OOS gates means `experimental rejected`, not
"promising." No tier may use PnL, costs, or execution claims as a substitute
for the probability-forecast gate.

## Complexity map

`d` is Hilbert-space dimension. A dense complex128 density matrix costs
`16*d^2` bytes before workspace; dense diagonalisation/exponentiation has
`O(d^3)` work. Numbers below exclude dataframe and library overhead.

| Tier | Literal model and practical scale | Complexity / concrete cost | Current state and acceptance gate |
|---|---|---|---|
| 0 | Current qutrit open filter (`d=3`) | 144 B state; dense work is negligible | Implemented. First build the matched classical 3-state comparator and pass the shared OOS gate; otherwise do not enlarge it. |
| 1 | Single ten-level open filter (`d=10`), one basis state per frozen FX-pair index | 1,600 B density; `O(10^3)` dense update | Implementable now with the existing scientific stack. It may model categorical occupancy/coherence across pairs, but it is **not** a ten-subsystem model. Promote only after Tier 0's protocol is frozen and Tier 1 beats a matched 10-class classical filter. |
| 2 | Ten locally encoded subsystems with quantum trajectories. Binary encoding: `d=2^10=1,024`; ternary down/flat/up encoding: `d=3^10=59,049` | Exact binary density = 16 MiB but dense exponentiation is about `1.07e9` cubic-scale operations/update. Exact ternary density = 51.96 GiB and is not a practical dense minute-bar model. A binary pure trajectory is 16 KiB; ternary is 0.90 MiB, per trajectory. | The qutrit unraveling passes numerical/gap checks but fails its frozen final-30% baselines. No untruncated ten-body trajectory is authorised. Any revisit must validate the qutrit unraveling against exact density and pass `R` versus `2R` emission convergence before OOS scoring. |
| 3 | MPS / locally purified tensor-network state for ten qubits or qutrits, with fixed pair ordering and a sparse interaction graph | Pure MPS storage is `O(n*q*chi^2)`; density/MPO storage is `O(n*q^2*chi^2)`. For 10 qutrits and `chi=32`, the latter is about 1.41 MiB of complex128 coefficients before overhead. TEBD-style updates scale roughly `O(n*q^3*chi^3)` per local layer; long-range all-to-all coupling removes this advantage. | Implemented from first principles: a ten-qutrit spin-1, Strang-TEBD Lindblad trajectory at `chi=16`. It is numerically norm-safe, but its bounded 500-row run has total discarded norm 0.636266 (far above the `1e-5` gate) and loses OOS. It is rejected; require `chi={8,16,32,64}` convergence before any longer run. |
| 4 | Quantum kernels and variational quantum circuits (VQCs) | A float64 kernel Gram matrix costs 0.745 GiB at 10,000 rows and 74.5 GiB at 100,000. A symmetric 10,000-row kernel at 1,024 shots needs about 51.2 billion shots before OOS scoring. A parameter-shift VQC needs roughly `epochs * batches * 2*parameters * shots`; with 50 epochs, 128 batches, 128 parameters, and 1,024 shots that is about 1.68 billion shots. | An exact 10-qubit `H`, `RY/RZ`, `RZZ/RZX` fidelity-kernel simulator with train-only 64-landmark Nyström KRR is implemented. It fails OOS frozen-prior Brier. Trainable VQC and hardware are still blocked by missing SDK/provider access and must first beat equal-feature classical logistic, RBF, and random-Fourier baselines. |
| 5 | Quantum reservoir: fixed random/local circuit, measured observables, classical ridge/logistic readout | Statevector simulation is `O(2^q)` memory and at least `O(depth*2^q)` work/sample. Hardware sampling is `N * measurement_circuits * shots`: even one circuit at 1,024 shots over 3.7M minute bars is about 3.8 billion shots. | Implemented: exact 10-qubit, three-layer data-reupload reservoir with 120 declared gates/minute and 21 observables; ridge readout trains in-fold only. It fails its 14,657-row OOS frozen-prior Brier/top-1 comparison. Five-seed and equal-width echo-state/random-feature tests are required before any revisit. |
| 6 | Real quantum processor execution (sampler/estimator primitives) | Queue delay, transpilation, calibration drift, shot noise, and readout mitigation become part of the experiment. The hardware cost is the Tier-4/5 shot count plus calibration circuits and retries. | Local precursor complete: Qiskit Aer density-matrix simulation ran 24 causal ten-qubit circuits at 4,096 shots with declared synthetic depolarizing/readout noise; mean ideal-to-noisy observable error was 0.02704. Real hardware is still blocked: no provider package, credentials, backend selection, or approved account. A QPU may reproduce a locked small experiment only after immutable circuit/config hashes, backend/job IDs, calibration metadata, and a precommitted simulator-to-hardware protocol exist. |

`chi` is the tensor-network bond dimension, `q` is local dimension (2 or 3),
and `R` is the number of independent trajectories. The resource figures show
why a ten-level *single* system is easy while a tensor product of ten qutrits
is not: `10` states and `3^10` states express entirely different hypotheses.

## Safe implementation sequence

1. **Record the stress test honestly.** Every currently implemented predictive
   quantum branch is rejected for promotion; retain artifacts as negative
   baselines, not targets to tune against.
2. **Close Tier 0 with a matched classical protocol.** Add the comparator,
   three chronological outer folds, and placebos before changing any feature
   map or hyperparameter.
3. **Repair tensor-network convergence before scale.** The observed discarded
   MPS weight rejects `chi=16`; use the required bond-dimension convergence
   sequence, or terminate that representation.
4. **Use hardware only as reproduction research.** A hardware account and SDK are
   necessary but never sufficient; a noisy, queued, shot-limited processor
   cannot satisfy this project's causal 60-second operational contract.

## Explicit exclusions

- Do not alter `docs/state-schema.md`, the classical integrator, forcing,
  coupling, controller, or any trading path from a quantum experiment.
- Do not use future normalisation, daily coupling estimates emitted after the
  bar, cross-fold hyperparameter selection, cherry-picked seeds, or gap-filled
  returns.
- Do not claim physical quantumness, entanglement of currency pairs, quantum
  causation, speedup, or trading profitability from software simulations.
- Do not run dense ten-qutrit density matrices, all-to-all unvalidated tensor
  networks, full-history kernel matrices, or hardware jobs without a passed
  preceding tier and explicit dependency/access approval.

The branch therefore has a coherent scientific ceiling: increasingly capable
open-system *representations* may be tested, but every increase in Hilbert
space, approximation, or hardware realism must first clear numerical,
causality, placebo, and untouched-OOS gates. A failed gate narrows the branch;
it never upgrades the market claim.

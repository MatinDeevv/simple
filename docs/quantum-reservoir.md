# Ten-qubit quantum reservoir experiment

pipeline/quantum_reservoir.py is an isolated **software** experiment. It uses
valid complex state-vector circuit operations, but it does **not** assert that
FX prices are a physical quantum system. It does not write canonical state,
does not generate a trading signal, and is not eligible for promotion from
this result.

## Circuit

The reservoir has one qubit per canonical pair:

AUDUSD, EURGBP, EURJPY, EURUSD, GBPJPY, GBPUSD, USDCAD, USDCHF, USDCNH, USDJPY.

It maintains a normalized complex state vector in a 1,024-dimensional Hilbert
space. For every valid minute t, its current input is the ten-vector

\[
z_{i,t} =
 \frac{\log P_{i,t}-\log P_{i,t-1}}
      {\sqrt{\operatorname{EWMA}_{1440}[(\log P_{i,t}-\log P_{i,t-1})^2]}}.
\]

The EWMA is causal and bounded standardized returns are re-uploaded into each
of three fixed circuit layers. Each layer contains:

- ten seeded, input-dependent RY rotations and ten seeded,
  input-dependent RZ rotations;
- ten local RZZ gates and a directed nearest-neighbour CNOT ring.

That is 120 declared single-/two-qubit gates per valid minute. All circuit
parameters are sampled from the fixed seed before targets are examined and
recorded by SHA-256. The implementation compiles the commuting RZZ ring and
the complete CNOT permutation exactly; the self-check compares the compiled
operation to the declared gate-by-gate operation.

The readout feature is deliberately modest: ten <Z_q> expectations, ten
nearest-neighbour <Z_q Z_(q+1)> correlations, and global Z-parity (21 features
total). This avoids silently feeding classical raw returns directly into the
learned readout.

## Causal and gap contract

The target is the pair with the largest absolute valid one-minute return at
t+1. It is not encoded into the state or used in the circuit parameters. At a
missing/non-60-second interval:

1. The pre-gap row is not encoded or scored because its target is invalid.
2. The first post-gap row resets the state to the all-zero basis state and
   resets the EWMA before a return is formed.
3. No circuit state can bridge the gap.

After the causal feature run, the final 30% of raw timestamps is fixed as
chronological OOS. A one-hot multiclass ridge readout with softmax is fitted
only on pre-OOS rows. Feature mean/scale, ridge weights, class prior, and
modal-class baseline are all frozen before the OOS rows are scored.

## Commands

    python pipeline/quantum_reservoir.py --self-check
    python pipeline/quantum_reservoir.py --max-steps 50000

--max-steps selects the latest bounded canonical window. The default is
50,000 raw minute rows, keeping the full state-vector experiment practical
without pretending it is a full-history production fit.

## Validated run

The committed run used 50,000 latest raw rows from 2024-11-11T08:40:00Z through
2024-12-31T21:59:00Z. OOS starts at 2024-12-16T03:40:00Z.

| Check | Result |
|---|---:|
| Valid reservoir updates | 49,105 |
| Train / OOS updates | 34,448 / 14,657 |
| Gap state resets | 499 |
| Cross-gap state updates | 0 |
| Maximum state-norm error | 3.280e-13 |
| Gate-by-gate vs compiled ring error | 2.861e-17 |
| Fixed-seed replay / gate checks | pass |

The frozen OOS diagnostic is not promotable:

| Predictor | Top-1 accuracy | Brier | Log loss |
|---|---:|---:|---:|
| Quantum-reservoir + ridge readout | 0.233267 | 0.893486 | 2.270701 |
| Uniform | 0.233950 | 0.900000 | 2.302585 |
| Frozen train class prior | 0.233950 | 0.871358 | 2.170091 |
| Frozen train modal class | 0.233950 | 1.532101 | 26.458442 |

It improves Brier over uniform but loses both top-1 and Brier to the frozen
train prior, so passes_all_fixed_baselines is false. This is a rejection
result, not evidence for market quantum ontology or for tradability.

## Audit artifacts

- data_derived/quantum_reservoir_minute.parquet — every raw row, gap reason,
  causal inputs, 21 observables, target, and OOS-only model probabilities.
- data_derived/quantum_reservoir_daily.parquet — day-level update and OOS
  diagnostic aggregates.
- data_derived/quantum_reservoir_summary.json — split, model configuration,
  complete readout weights, frozen baselines, hashes, and result.
- data_derived/quantum_reservoir_validation.json — state norm, observable,
  gate-orientation, compiled-ring equivalence, and deterministic-seed checks.

The minute artifact intentionally leaves pre-OOS model probabilities empty:
only OOS predictions are scored. This prevents in-sample output from being
mistaken for a holdout metric.

## Scaling boundary

Ten qubits require 1,024 complex amplitudes (16 KiB in complex128), while a
dense 10-qubit density matrix would require 1024 squared complex values
(16 MiB) before workspace and repeated-channel overhead. The state-vector
model remains tractable because it is unitary and uses a small observable
readout. Extending to open-system noise, density matrices, or materially more
qubits needs a separate evidence gate and usually tensor-network methods; it
must not be framed as greater physical fidelity of the market.

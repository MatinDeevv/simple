# Ten-Qubit Quantum-Fidelity Kernel Experiment

## Purpose and hard boundary

`pipeline/quantum_kernel.py` is an isolated, exact classical statevector
simulation of a 10-qubit quantum feature map plus a quantum-fidelity kernel.
It uses real quantum-circuit linear algebra, but it is **not** evidence that
FX markets are physical quantum systems, and it is not a trading model or a
canonical simulator input. The output remains non-promoted unless it passes
the stated holdout gates.

The implementation deliberately uses only NumPy for quantum and kernel math
(and Pandas solely for local Parquet I/O). No quantum SDK, quantum provider,
or hardware result is implied.

## Causal universe and target

The ten qubits have a fixed canonical order:

    0 EURUSD, 1 USDJPY, 2 GBPUSD, 3 AUDUSD, 4 USDCAD,
    5 USDCNH, 6 USDCHF, 7 EURGBP, 8 EURJPY, 9 GBPJPY.

At a valid contiguous minute `t`, the input is the 10-vector of return from
`t-1` to `t`, standardized by a causal 1,440-step EWMA variance. The latest
coupling estimate with update timestamp no later than `t` is the only coupling
input. A coupling older than 36 hours is rejected. At a gap, no return,
target, state, or score is emitted across the boundary; the EWMA restarts on
the first post-gap bar.

The diagnostic target is the index of the pair with largest absolute log
return from `t` to the next valid minute `t+1`. The target never enters the
feature map, normalizer, kernel, landmark selection, or state construction.

## Fixed circuit and kernel

The state begins at `|0...0>` and executes this fixed topology:

    H on all ten qubits
    RY(0.75 tanh(z_q)) then RZ(0.55 z_q) on each q
    RZZ(theta_zz) then RZX(theta_zx) on nearest-neighbor edges q,q+1
    for q = 0...8

`z` is the training-normalized causal standardized return, clipped to
`[-3,3]`. The edge angles combine causal neighboring `z` values with bounded
contemporaneous coupling entries. `RY` and `RZ` do not commute on a qubit;
the `RZZ` and `RZX` edge operations also give a noncommuting nearest-neighbor
feature-map layer. The resulting exact statevector has `2^10 = 1,024` complex
amplitudes and is norm-checked for every encoded selected observation.

For samples `x,y`, the kernel is the quantum fidelity kernel:

    K(x,y) = | <psi(x) | psi(y)> |^2.

It is formed exactly from the simulated statevectors. A train-only Nyström
approximation chooses 64 seeded landmarks from training states, eigendecomposes
their kernel, floors only eigenvalues below `1e-10`, and maps into the low-rank
feature surface. A regularized multiclass kernel-ridge classifier (`ridge=0.01`)
is fitted on those training features. The raw KRR scores are transformed by a
fixed softmax solely to report normalized probabilities and Brier/log-loss
diagnostics.

## Split, bounds, and baselines

The replay has a predeclared chronological 70%/30% source-row split. It uses
bounded, evenly spaced, label-agnostic samples: at most 384 training and 192
final-holdout observations. The normalizer, landmarks, class prior, and
constant-class baseline are fit/frozen using only the selected training
partition. The holdout includes:

- Uniform ten-class probabilities.
- Frozen training empirical class-prior probabilities.
- Frozen training modal-class constant prediction.
- A simple causal last-magnitude baseline: the largest absolute current
  return predicts the next largest-return pair.

Promotion requires both a lower Brier score than frozen training prior and a
higher top-1 score than the causal last-magnitude baseline. Passing it would
only permit a separate validation decision; it would not establish market
quantum ontology or authorize trading use.

## Audit and reproducibility

Run:

```text
python pipeline/quantum_kernel.py --self-check
python pipeline/quantum_kernel.py
```

The self-check verifies a 1,024-amplitude normalized state, same-input
bitwise determinism, fidelity-kernel unit diagonal, PSD tolerance,
finite Nyström features, and classifier probability normalization.

The executed runner writes:

- `data_derived/quantum_kernel_minute.parquet` — one row for every processed
  source minute, including gap/stale/unselected reasons; selected holdout rows
  additionally carry model and frozen-baseline probabilities and Brier values.
- `data_derived/quantum_kernel_daily.parquet` — daily aggregation of the
  selected holdout diagnostics.
- `data_derived/quantum_kernel_summary.json` — configuration, causal contract,
  hashes, provenance, numerical diagnostics, scores, and promotion decision.
- `data_derived/quantum_kernel_validation.json` — independent self-check data.

The summary contains the fixed seed, config hash, selected-row hashes, encoded
state hash, canonical manifest/input hashes, and artifact hashes. These expose
the exact bounded run without silently treating a simulation as hardware
execution.

## Executed bounded run

The default command was executed with seed `20260718`, 250,000 source rows,
384 train samples, 192 frozen final-holdout samples, and 64 train-only
landmarks. It found 232,826 contiguous causal candidates. The state-vector
maximum norm error was `3.109e-15`; the 64-landmark kernel minimum eigenvalue
was `0.008791`, so no eigenvalue required flooring.

| Final-30% holdout metric | Result |
|---|---:|
| Quantum-kernel top-1 / Brier / log loss | 18.7500% / 0.889459 / 2.253690 |
| Uniform top-1 / Brier | 8.3333% / 0.900000 |
| Frozen train-prior top-1 / Brier | 32.2917% / 0.849528 |
| Frozen AUDUSD constant top-1 / Brier | 32.2917% / 1.354167 |
| Causal last-magnitude top-1 / Brier | 18.2292% / 1.635417 |

The experiment is **not promoted**. It modestly exceeds uniform and the simple
last-magnitude top-1 figure, but loses decisively to the frozen train-prior on
both top-1 and the proper Brier score. This is an honest rejection result, not
a trading claim.

## Scaling boundary

An exact `n`-qubit statevector has `2^n` amplitudes and a density matrix has
`4^n` entries. The current 10-qubit pure-state feature map is deliberately
within exact classical reach. The next genuinely more complex step would be
to retain more qubits only if a frozen classical comparison first clears its
holdout gate; beyond roughly a few dozen entangled qubits, exact statevectors
become exponential and a tensor-network / MPS approximation needs explicit
bond-dimension convergence tests. Hardware runs would add shots, device noise,
transpilation, and provider calibration constraints; they cannot validate a
physical-quantum-market hypothesis.

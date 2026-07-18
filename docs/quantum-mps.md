# 10-Qutrit MPS / TEBD Quantum-Trajectory Experiment

## Status and boundary

`pipeline/quantum_mps.py` is an isolated, **experimental non-promotion**
layer.  It applies genuine finite-dimensional quantum mechanics--unitary
evolution, a Lindblad-jump unraveling, MPS tensor factorization, and SVD bond
truncation--to a causal statistical encoding of ten FX series.  It does not
assert that currency markets are physical quantum systems, that a price is a
quantum measurement, or that the output is a trading signal.  It neither
reads nor changes canonical state/simulation paths and is not a controller
input.

The experiment exists to find the computational boundary beyond the three
qutrit density/trajectory prototypes.  A failed forecast diagnostic remains a
failure even when the quantum numerical contracts pass.

## Fixed system and causal inputs

There are ten qutrit sites in the frozen canonical coupling order:

```text
EURUSD -- USDJPY -- GBPUSD -- AUDUSD -- USDCAD -- USDCNH -- USDCHF
  -- EURGBP -- EURJPY -- GBPJPY
```

At a contiguous minute `t`, each site receives only

```text
r_i(t) = log(close_i(t)) - log(close_i(t-60s))
z_i(t) = r_i(t) / sqrt(EWMA_1440(r_i^2))
```

where the volatility EWMA is initialized to `1e-8` and updated after the
return is observed.  The full canonical input is a strict ten-way timestamp
intersection--there is no resampling, fill, or cross-gap return.  The
coupling snapshot is the newest row in `coupling_estimates.parquet` whose
`update_time <= t`.  Thus neither the Hamiltonian nor the next-bar diagnostic
can use future input.

Each qutrit uses the purely computational basis

```text
|0> = down, |1> = neutral, |2> = up.
```

Those labels are basis coordinates, not a claim about an FX market state.
The MPS chain graph is a fixed numerical ordering, not an assertion that
adjacent entries have a special economic relationship.

## Hermitian local and coupling generators

Let `Sx`, `Sy`, and `Sz` be the standard spin-one matrices.  The local
dimensionless Hamiltonian is

```text
h_i(t) = clip(z_i(t), -5, 5) * (0.085 Sz + 0.115 Sx).
```

For each adjacent MPS bond `(i,i+1)`, the causal directed coupling field is
made compatible with a Hermitian exchange generator by the explicit symmetric
projection

```text
J_i(t) = clip(0.85 * 60^2 * [C_i,i+1(t) + C_i+1,i(t)] / 2, -0.35, 0.35)
B_i(t) = J_i(t) * [Sx x Sx + Sy x Sy + 0.35 Sz x Sz].
```

The antisymmetric/directed part of `C` is not mislabelled as a Hamiltonian; it
is outside this representation.  All local and two-site gates are calculated
from cached Hermitian eigendecompositions, so `U=exp(-iH)` is checked for
unitarity without using a dense `3^10 x 3^10` matrix.

The update is a second-order Strang TEBD split:

```text
local half -> even bonds half -> odd bonds full -> even bonds half -> local half.
```

After every two-site gate, the combined MPS tensor is split by SVD.  Singular
values are retained subject to `chi` and a relative cutoff, and the discarded
norm-squared is emitted.  The current run uses `chi=16` and cutoff `1e-10`.
This makes its approximation error visible; it never hides truncation behind
an implicit renormalization.

## Open-system trajectory layer

After the TEBD unitary layer, each site independently samples the pure
dephasing instrument

```text
K_0 = sqrt(0.996) I
K_q = sqrt(0.004) |q><q|,  q in {0,1,2}.
```

Equivalently, `gamma=-log(0.996)/60` per second and the local Lindblad
operators are `L_q=sqrt(gamma)|q><q|`.  On a jump, the site is projected using
its MPS-computed Born probabilities and the global state is normalized.  A
fixed NumPy RNG seed makes the complete wavefunction trajectory repeatable.
This is an algorithmic bath sample; it is not an observed market trade,
execution, or physical decoherence event.

Any non-60-second interval is a hard boundary.  The pre-gap row receives no
prediction; the first post-gap row resets the MPS to an exactly normalized
uniform qutrit product state and resets every EWMA.  No state, return, or
volatility is carried over the gap.

## Artifacts and numerical checks

```text
python pipeline/quantum_mps.py --self-check
python pipeline/quantum_mps.py --max-steps 500 --chi 16 --seed 20260718
```

The executed replay writes:

- `data_derived/quantum_mps_minute.parquet`
- `data_derived/quantum_mps_daily.parquet`
- `data_derived/quantum_mps_summary.json`
- `data_derived/quantum_mps_validation.json`

Every source row in the minute artifact has a reason code, prior/next
contiguity flags, reset flag, causal coupling timestamp and age, the entire
10x10 coupling snapshot, raw and causal standardized returns, each local
three-outcome Born distribution, normalized activity output, next-bar target,
jump count, truncation metrics, norm error, and exact Schmidt diagnostics.

The self-check verifies Hermiticity, local and bond unitarity, exact untruncated
MPS splitting, intentionally forced `chi=1` discarded amplitude, nonzero
Schmidt rank/entropy, norm preservation, and bitwise same-seed replay.  The
executed-run validator also requires finite metrics and norm error below
`1e-10`.

## Executed canonical replay

The bounded canonical run used 500 strict-common source rows from
2015-01-30 21:51 UTC through 2015-02-02 07:33 UTC.  It is deliberately a
numerical integration smoke/diagnostic, not a sufficient research holdout.

| Metric | Result |
|---|---:|
| Strict ten-pair source intersection | 3,541,367 rows |
| Requested / valid updates | 500 / 474 |
| Gap resets | 14 |
| `chi` / max observed bond dimension | 16 / 16 |
| Lindblad jump events | 19 |
| Truncation events / total discarded weight | 3,747 / 0.636266 |
| Maximum discarded weight in one full minute | 0.00410519 |
| Maximum norm error | 1.110e-15 |
| Maximum gate unitarity error | 1.110e-15 |
| Maximum Schmidt entropy | 2.259236 |
| OOS rows (chronological final 30%) | 149 |
| MPS activity top-1 / Brier | 6.0403% / 0.904375 |
| Uniform expected top-1 / Brier | 10.0000% / 0.900000 |
| Frozen AUDUSD-constant top-1 / Brier | 36.9128% / 1.261745 |
| Frozen empirical-prior top-1 / Brier | 36.9128% / 0.789751 |

The fixed diagnostic target is the pair with the largest absolute next valid
one-minute log return.  MPS activity is normalized
`p_i(down)+p_i(up)`.  All baselines freeze from the chronological pre-OOS
labels.  The model loses to uniform and frozen empirical-prior Brier score and
to the frozen modal-class top-1 baseline.  It is therefore rejected for
promotion, regardless of its valid quantum numerical behavior.

## Why MPS is the correct next computational boundary

Ten qutrits have Hilbert dimension `3^10 = 59,049`.  A single complex128 pure
state is only about 0.90 MiB, but a dense complex128 density matrix is about
51.96 GiB before the operations needed to evolve it.  The Lindblad trajectory
unravelling avoids that density matrix; the MPS representation further stores
a weakly entangled path in `O(n*d*chi^2)` complex values rather than `O(3^n)`.
For `n=10`, `d=3`, and `chi=16`, the loose storage bound is 120 KiB.

Generic TEBD work per minute is approximately `O(n*d^3*chi^3)` due to
two-site SVDs.  The current exact Schmidt-audit routine is intentionally more
expensive than production MPS measurement and is appropriate only to this
small fixed ten-site experiment; a larger research run must use canonical
form/environment sweeps rather than expanding block bases.  MPS is also only
faithful when the required entanglement remains representable at the selected
bond dimension.  The emitted discarded weight, bond rank, and entropy are
therefore first-class acceptance gates, not decoration.

The next defensible computational progression would be a pre-registered
`chi` convergence study on an untouched sufficiently long holdout, then a
purified matrix-product density operator / process-tensor variant only if the
converged pure-trajectory representation passes every frozen classical
baseline.  Real quantum hardware, variational circuits, and larger tensor
networks cannot turn a failed causal diagnostic into evidence of physical
market quantum behavior.

# Quantum Channel Tomography

## Scope

`engine/quantum/quantum_process_tomography.py` is a numerical-physics audit of the
qutrit branch. It constructs the **nonselective** channel associated with the
current causal inputs. This is deliberately different from the conditional,
normalized likelihood-filter update used for `rho` in
`quantum_lindblad.py`: conditioning and renormalizing makes that filter
nonlinear, so it is not honestly presented as one standalone CPTP channel.

For each sampled causal input, the audited channel is

    E(rho) = D( sum_a K_a U rho U-dagger K_a-dagger )

where `U` is the input-conditioned unitary, `K0,K1` are the complete diagonal
instrument, and `D` is the finite-step qutrit pure-dephasing channel. Its
eight composed Kraus operators are explicitly assembled and tested.

## What is checked

- Kraus completeness: `sum_r A_r-dagger A_r = I`.
- Choi matrix Hermiticity and positivity, so the map is completely positive.
- Choi output partial trace, so the map is trace preserving.
- A nine-by-nine identity-plus-Gell-Mann process-transfer matrix, including
  the spectral radius on the eight-dimensional traceless sector.

The sampled artifact stores causal timestamp/coupling age, input `z`,
Hamiltonian diagonal/norm, Choi checks, and transfer contraction statistic.
The runner excludes first-post-gap, pre-gap, and stale-coupling bars exactly
as the repaired density-filter replay does.

## Execution

    python engine/quantum/quantum_process_tomography.py --self-check
    python engine/quantum/quantum_process_tomography.py --max-steps 250000 --max-samples 512

Outputs are `data/derived/quantum_process_tomography.parquet` and
`data/derived/quantum_process_tomography_summary.json`.

## Interpretation boundary

Passing tomography means the constructed software channel is a valid CPTP
quantum channel for every sampled classical input. It says nothing about
whether price formation is a quantum channel, whether its coherence is a
market observable, whether any pair is entangled, or whether it forecasts or
trades successfully. It is numerical validation only and cannot promote the
experimental branch.

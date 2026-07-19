# Qiskit Aer Noise-Calibration Experiment

## Scope

This experiment is the hardware-simulation tier of the isolated quantum
research branch. It uses Qiskit Aer locally, with a declared synthetic noise
model, to execute a ten-qubit data-reuploading circuit and compare sampled
noisy observables with ideal statevector observables.

It is **not** a real quantum-processor run. The error rates are not copied from
a QPU calibration, no provider account or credentials are present, and the
result does not establish hardware performance, quantum advantage, market
ontology, or a trading signal.

## Circuit and simulation

For each label-free, gap-safe causal ten-pair input, the circuit prepares ten
qubits with a Hadamard layer and then three re-uploading layers:

    10 RY + 10 RZ + 10 RZZ + 10 CX ring gates.

Inputs are only valid `t-1` to `t` returns standardized by a resettable causal
EWMA. The runner measures ten Z expectations, ten nearest-neighbour ZZ
correlations, and global Z parity. Aer uses the density-matrix method plus:

- one-qubit depolarizing probability `0.001` after H/RY/RZ;
- two-qubit depolarizing probability `0.012` after RZZ/CX;
- asymmetric independent readout assignment matrix
  `[[0.992, 0.008], [0.012, 0.988]]`.

The noise model is deliberately declared in the output artifact. It must not be
called a backend calibration.

## Execution

Create the isolated environment once:

    python -m venv .venv-quantum
    .venv-quantum\Scripts\python.exe -m pip install -r requirements-quantum.txt

Then run:

    .venv-quantum\Scripts\python.exe engine\quantum\quantum_aer_noise.py --self-check
    .venv-quantum\Scripts\python.exe engine\quantum\quantum_aer_noise.py --max-rows 50000 --max-samples 24 --shots 4096

The resulting Parquet artifact records causal input, circuit size/depth,
ideal/noisy Z and ZZ observables, parity, and absolute observable errors.

## Promotion boundary

This is a noise-calibration diagnostic only. It makes no forecast, has no
target/readout, cannot choose a predictive model, and cannot promote OQ-13.
Real-hardware work additionally needs explicit provider authority, a selected
backend, immutable circuit/config hashes, job IDs, calibration metadata, and a
precommitted simulator-to-hardware comparison protocol.

"""Qiskit Aer hardware-noise calibration for the isolated 10-qubit branch.

Run with .venv-quantum\\Scripts\\python.exe.  This is local noisy simulation
with a declared synthetic error model, not a real-QPU submission or calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from engine.core.contracts import canonical_pair_order, contiguous_60s

try:
    import qiskit
    import qiskit_aer
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
except ImportError as exc:  # pragma: no cover - depends on optional environment
    raise SystemExit(
        "Qiskit Aer is required. Use .venv-quantum\\Scripts\\python.exe "
        "engine\\quantum\\quantum_aer_noise.py after installing requirements-quantum.txt."
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DIR = ROOT / "data" / "canonical"
DERIVED_DIR = ROOT / "data" / "derived"
MANIFEST_PATH = CANONICAL_DIR / "manifest.json"
N_QUBITS = 10
DT_NS = 60_000_000_000
HALFLIFE_STEPS = 1_440.0
EWMA_ALPHA = 1.0 - math.exp(-math.log(2.0) / HALFLIFE_STEPS)
DEFAULT_MAX_ROWS = 50_000
DEFAULT_SAMPLES = 24
DEFAULT_SHOTS = 4_096
SEED = 20_260_718
VERSION = "quantum-aer-noise-1.0.0"


class ContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def epoch_ns(values: object) -> np.ndarray:
    return pd.DatetimeIndex(values).as_unit("ns").asi8


def pair_order() -> tuple[str, ...]:
    pairs = canonical_pair_order(ROOT)
    if len(pairs) != N_QUBITS:
        raise ContractError("canonical manifest does not define exactly ten unique qubits")
    return pairs


def latest_common_prices(pairs: tuple[str, ...], max_rows: int) -> tuple[np.ndarray, np.ndarray]:
    if max_rows < 32:
        raise ContractError("max_rows must be at least 32")
    reference = pd.read_parquet(CANONICAL_DIR / f"{pairs[0]}.parquet", columns=["timestamp"])
    if len(reference) < max_rows + 3:
        raise ContractError("reference canonical series is shorter than requested window")
    start = reference.iloc[-max_rows - 3]["timestamp"]
    end = reference.iloc[-1]["timestamp"]
    joined: pd.DataFrame | None = None
    for pair in pairs:
        frame = pd.read_parquet(CANONICAL_DIR / f"{pair}.parquet", columns=["timestamp", "close"],
                                filters=[("timestamp", ">=", start), ("timestamp", "<=", end)])
        frame = frame.rename(columns={"close": pair})
        joined = frame if joined is None else joined.merge(frame, on="timestamp", how="inner",
                                                            validate="one_to_one")
    assert joined is not None
    joined = joined.sort_values("timestamp", kind="stable").reset_index(drop=True)
    times = epoch_ns(joined["timestamp"])
    close = joined.loc[:, list(pairs)].to_numpy(dtype=np.float64)
    if len(times) < 32 or not np.all(np.diff(times) > 0) or not np.isfinite(close).all() or np.any(close <= 0.0):
        raise ContractError("latest ten-pair joined window violates input contract")
    return times, np.log(close)


def causal_z_samples(times: np.ndarray, log_close: np.ndarray, max_samples: int) -> tuple[list[dict[str, object]], dict[str, int]]:
    if max_samples < 1:
        raise ContractError("max_samples must be positive")
    sigma2 = np.full(N_QUBITS, 1e-8, dtype=np.float64)
    valid: list[dict[str, object]] = []
    post_gap_skips = pre_gap_skips = 0
    for i in range(1, len(times) - 1):
        previous_ok = contiguous_60s(int(times[i - 1]), int(times[i]), DT_NS)
        next_ok = contiguous_60s(int(times[i]), int(times[i + 1]), DT_NS)
        if not previous_ok:
            sigma2.fill(1e-8)
            post_gap_skips += 1
            continue
        if not next_ok:
            pre_gap_skips += 1
            continue
        ret = log_close[i] - log_close[i - 1]
        sigma2 = (1.0 - EWMA_ALPHA) * sigma2 + EWMA_ALPHA * ret * ret
        z = ret / np.sqrt(np.maximum(sigma2, 1e-16))
        valid.append({"timestamp": int(times[i]), "z": z})
    if not valid:
        raise ContractError("no contiguous causal inputs available")
    selection = np.linspace(0, len(valid) - 1, min(max_samples, len(valid)), dtype=np.int64)
    samples = [valid[int(index)] for index in selection]
    return samples, {"valid_causal_rows": len(valid), "post_gap_reset_skips": post_gap_skips,
                     "pre_gap_skips": pre_gap_skips, "selected_samples": len(samples)}


def build_circuit(z: np.ndarray) -> QuantumCircuit:
    """Fixed three-reupload circuit with causal inputs substituted only as angles."""
    if z.shape != (N_QUBITS,) or not np.isfinite(z).all():
        raise ContractError("circuit encoding requires ten finite standardized returns")
    clipped = np.clip(z, -4.0, 4.0)
    circuit = QuantumCircuit(N_QUBITS)
    for qubit in range(N_QUBITS):
        circuit.h(qubit)
    for layer in range(3):
        for qubit in range(N_QUBITS):
            circuit.ry(0.55 * math.tanh(float(clipped[qubit])) + 0.13 * (layer + 1), qubit)
            circuit.rz(0.35 * float(clipped[qubit]) - 0.07 * (layer + 1), qubit)
        for qubit in range(N_QUBITS):
            neighbor = (qubit + 1) % N_QUBITS
            angle = 0.14 * math.tanh(float(clipped[qubit] * clipped[neighbor]))
            circuit.rzz(angle, qubit, neighbor)
            circuit.cx(qubit, neighbor)
    return circuit


def measured_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    result = QuantumCircuit(N_QUBITS, N_QUBITS)
    result.compose(circuit, qubits=range(N_QUBITS), inplace=True)
    result.measure(range(N_QUBITS), range(N_QUBITS))
    return result


def z_observables_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.shape != (1 << N_QUBITS,):
        raise ContractError("state probability vector has incorrect dimension")
    indices = np.arange(1 << N_QUBITS, dtype=np.uint16)
    z_values = np.vstack([1.0 - 2.0 * ((indices >> qubit) & 1) for qubit in range(N_QUBITS)])
    z_expectation = z_values @ probabilities
    zz_expectation = np.array([np.dot(z_values[qubit] * z_values[(qubit + 1) % N_QUBITS], probabilities)
                               for qubit in range(N_QUBITS)])
    parity = float(np.dot(np.prod(z_values, axis=0), probabilities))
    return np.concatenate([z_expectation, zz_expectation, [parity]])


def observables_from_counts(counts: dict[str, int]) -> tuple[np.ndarray, int]:
    total = int(sum(counts.values()))
    if total <= 0:
        raise ContractError("Aer returned no measurement counts")
    z_sum = np.zeros(N_QUBITS, dtype=np.float64)
    zz_sum = np.zeros(N_QUBITS, dtype=np.float64)
    parity_sum = 0.0
    for key, count in counts.items():
        bits = key.replace(" ", "")
        if len(bits) != N_QUBITS or any(bit not in "01" for bit in bits):
            raise ContractError(f"unexpected Aer count key {key!r}")
        # Qiskit count strings are highest classical bit first; c[q] maps to q.
        z = np.array([1.0 - 2.0 * int(bits[-1 - qubit]) for qubit in range(N_QUBITS)])
        z_sum += count * z
        zz_sum += count * z * np.roll(z, -1)
        parity_sum += count * float(np.prod(z))
    return np.concatenate([z_sum / total, zz_sum / total, [parity_sum / total]]), total


def noise_model() -> NoiseModel:
    """Declared synthetic gate/readout noise; it is not copied from a QPU calibration."""
    model = NoiseModel()
    one_qubit = depolarizing_error(0.001, 1)
    two_qubit = depolarizing_error(0.012, 2)
    model.add_all_qubit_quantum_error(one_qubit, ["h", "ry", "rz"])
    model.add_all_qubit_quantum_error(two_qubit, ["rzz", "cx"])
    model.add_all_qubit_readout_error(ReadoutError([[0.992, 0.008], [0.012, 0.988]]))
    return model


def self_check() -> dict[str, object]:
    circuit = build_circuit(np.linspace(-1.0, 1.0, N_QUBITS))
    ideal = Statevector.from_instruction(circuit)
    norm_error = abs(float(np.vdot(ideal.data, ideal.data).real) - 1.0)
    observations = z_observables_from_probabilities(np.abs(ideal.data) ** 2)
    if not np.all(np.abs(observations) <= 1.0 + 1e-12):
        raise ContractError("ideal Pauli observables fell outside [-1, 1]")
    measured = measured_circuit(circuit)
    simulator = AerSimulator(method="density_matrix", noise_model=noise_model())
    transpiled = transpile(measured, simulator, optimization_level=0, seed_transpiler=SEED)
    result = simulator.run(transpiled, shots=512, seed_simulator=SEED).result()
    noisy, shots = observables_from_counts(result.get_counts())
    return {
        "passed": bool(norm_error < 1e-12 and shots == 512
                       and np.isfinite(noisy).all() and np.all(np.abs(noisy) <= 1.0 + 1e-12)),
        "qiskit_version": qiskit.__version__,
        "qiskit_aer_version": qiskit_aer.__version__,
        "ideal_state_norm_error": norm_error,
        "noisy_shots": shots,
        "noisy_observable_range_error": float(max(0.0, np.max(np.abs(noisy)) - 1.0)),
    }


def run(max_rows: int, max_samples: int, shots: int, out_dir: Path) -> dict[str, object]:
    if shots < 128:
        raise ContractError("shots must be at least 128 for a noise-calibration result")
    pairs = pair_order()
    times, log_close = latest_common_prices(pairs, max_rows)
    samples, causal_counts = causal_z_samples(times, log_close, max_samples)
    simulator = AerSimulator(method="density_matrix", noise_model=noise_model())
    rows: list[dict[str, object]] = []
    for sample_index, sample in enumerate(samples):
        circuit = build_circuit(sample["z"])
        ideal = Statevector.from_instruction(circuit)
        ideal_observables = z_observables_from_probabilities(np.abs(ideal.data) ** 2)
        transpiled = transpile(measured_circuit(circuit), simulator, optimization_level=0,
                                seed_transpiler=SEED)
        result = simulator.run(transpiled, shots=shots, seed_simulator=SEED + sample_index).result()
        noisy_observables, actual_shots = observables_from_counts(result.get_counts())
        difference = noisy_observables - ideal_observables
        row: dict[str, object] = {
            "timestamp": pd.Timestamp(sample["timestamp"], unit="ns", tz="UTC"),
            "sample_index": sample_index,
            "shots": actual_shots,
            "circuit_depth": circuit.depth(),
            "circuit_size": circuit.size(),
            "ideal_state_norm_error": abs(float(np.vdot(ideal.data, ideal.data).real) - 1.0),
            "observable_mean_absolute_error": float(np.mean(np.abs(difference))),
            "observable_max_absolute_error": float(np.max(np.abs(difference))),
        }
        for pair, value in zip(pairs, sample["z"], strict=True):
            row[f"z_{pair.lower()}"] = float(value)
        for qubit in range(N_QUBITS):
            row[f"ideal_z_q{qubit}"] = float(ideal_observables[qubit])
            row[f"noisy_z_q{qubit}"] = float(noisy_observables[qubit])
            row[f"ideal_zz_q{qubit}_{(qubit + 1) % N_QUBITS}"] = float(ideal_observables[N_QUBITS + qubit])
            row[f"noisy_zz_q{qubit}_{(qubit + 1) % N_QUBITS}"] = float(noisy_observables[N_QUBITS + qubit])
        row["ideal_global_z_parity"] = float(ideal_observables[-1])
        row["noisy_global_z_parity"] = float(noisy_observables[-1])
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ContractError("noise-calibration artifact contains non-finite values")
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "quantum_aer_noise_calibration.parquet"
    summary_path = out_dir / "quantum_aer_noise_summary.json"
    frame.to_parquet(artifact, index=False, compression="zstd")
    summary = {
        "version": VERSION,
        "interpretation": (
            "local Aer density-matrix simulation with declared synthetic gate and readout noise; "
            "not hardware execution, backend calibration, physical-market evidence, or a trading result"),
        "sdk": {"qiskit": qiskit.__version__, "qiskit_aer": qiskit_aer.__version__,
                "python": sys.version.split()[0]},
        "pair_scope_and_qubit_order": list(pairs),
        "causality": {"input": "valid t-1 to t returns with resettable causal EWMA",
                       "target": None, "cross_gap_state_updates": 0,
                       "selection": "evenly spaced causal inputs; label-free"},
        "noise_model": {
            "one_qubit_depolarizing_probability": 0.001,
            "two_qubit_depolarizing_probability": 0.012,
            "readout_assignment_matrix": [[0.992, 0.008], [0.012, 0.988]],
            "origin": "synthetic declared calibration; not derived from a real QPU",
        },
        "circuit": {"qubits": N_QUBITS, "layers": 3,
                    "per_layer": "10 RY + 10 RZ + 10 RZZ + 10 CX ring gates",
                    "observables": "10 Z, 10 ring ZZ, global Z parity"},
        "replay": {**causal_counts, "max_rows": max_rows, "shots_per_sample": shots,
                   "max_ideal_state_norm_error": float(frame["ideal_state_norm_error"].max()),
                   "mean_observable_absolute_error": float(frame["observable_mean_absolute_error"].mean()),
                   "p95_observable_absolute_error": float(frame["observable_mean_absolute_error"].quantile(0.95)),
                   "max_observable_absolute_error": float(frame["observable_max_absolute_error"].max()),
                   "max_circuit_depth": int(frame["circuit_depth"].max()),
                   "max_circuit_size": int(frame["circuit_size"].max())},
        "provenance": {"manifest_sha256": sha256_file(MANIFEST_PATH),
                       "script_sha256": sha256_file(Path(__file__).resolve()), "seed": SEED},
        "promotion_status": "noise-calibration only; cannot promote any quantum predictive branch",
        "output": str(artifact.relative_to(ROOT)).replace("\\", "/"),
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--out-dir", type=Path, default=DERIVED_DIR)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.self_check:
            result = self_check()
            print(json.dumps(result, indent=2))
            return 0 if result["passed"] else 1
        run(args.max_rows, args.max_samples, args.shots, args.out_dir.resolve())
        return 0
    except (ContractError, ValueError, OSError) as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

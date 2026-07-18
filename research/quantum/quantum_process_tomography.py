"""Channel tomography for the nonselective qutrit quantum-research map.

This verifies a genuine quantum channel made from the unitary, the complete
Kraus instrument, and dephasing.  It does not turn the conditional FX filter
into a physical market measurement model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.quantum import quantum_lindblad as ql


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "derived"
DIM = 3
DEFAULT_MAX_STEPS = 250_000
DEFAULT_MAX_SAMPLES = 512
VERSION = "quantum-process-tomography-1.0.0"


class ContractError(RuntimeError):
    pass


def channel_kraus(z: np.ndarray, coupling: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, float]:
    """Kraus operators for the *nonselective* CPTP channel at one causal bar."""
    signal = np.clip(z, -ql.MAX_STANDARDIZED_RETURN, ql.MAX_STANDARDIZED_RETURN)
    symmetric = 0.5 * (coupling + coupling.T) * ql.DT_S * ql.DT_S
    hamiltonian = ql.PHASE_GAIN * (np.diag(signal) + ql.COUPLING_GAIN * symmetric)
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.T)
    hamiltonian -= np.eye(DIM) * np.trace(hamiltonian) / DIM
    unitary = ql.unitary_from_hamiltonian(hamiltonian)
    k0, k1, instrument_error = ql.measurement_instrument(signal)

    # rho -> (1-p) rho + p diag(rho) is the qutrit finite-step dephasing map.
    dephasing = [math.sqrt(1.0 - ql.DEPHASING_PER_STEP) * np.eye(DIM, dtype=np.complex128)]
    dephasing.extend(math.sqrt(ql.DEPHASING_PER_STEP) * np.diag(np.eye(DIM)[j])
                      for j in range(DIM))
    operators = [d @ k @ unitary for d in dephasing for k in (k0, k1)]
    return operators, hamiltonian, instrument_error


def apply_channel(rho: np.ndarray, operators: list[np.ndarray]) -> np.ndarray:
    return sum((operator @ rho @ operator.conj().T for operator in operators),
               start=np.zeros((DIM, DIM), dtype=np.complex128))


def choi_matrix(operators: list[np.ndarray]) -> np.ndarray:
    """J(E)=sum_ij |i><j| tensor E(|i><j|), ordered input then output."""
    choi = np.zeros((DIM * DIM, DIM * DIM), dtype=np.complex128)
    for i in range(DIM):
        for j in range(DIM):
            basis = np.zeros((DIM, DIM), dtype=np.complex128)
            basis[i, j] = 1.0
            choi += np.kron(basis, apply_channel(basis, operators))
    return choi


def choi_checks(choi: np.ndarray) -> tuple[float, float, float]:
    hermitian_error = float(np.max(np.abs(choi - choi.conj().T)))
    min_eigenvalue = float(np.linalg.eigvalsh(0.5 * (choi + choi.conj().T)).min())
    # np.kron(input, output) is indexed (input_row, output_row,
    # input_col, output_col), so trace the second and fourth tensor axes.
    tensor = choi.reshape(DIM, DIM, DIM, DIM)
    traced_output = np.einsum("iaja->ij", tensor)
    trace_preservation_error = float(np.max(np.abs(traced_output - np.eye(DIM))))
    return hermitian_error, min_eigenvalue, trace_preservation_error


def operator_basis() -> list[np.ndarray]:
    """Orthonormal identity plus eight Gell-Mann operators for qutrit transfer maps."""
    e = np.eye(DIM, dtype=np.complex128)
    basis = [e / math.sqrt(DIM)]
    basis.extend([
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.complex128),
        np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=np.complex128),
        np.diag([1, -1, 0]).astype(np.complex128),
        np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=np.complex128),
        np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=np.complex128),
        np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.complex128),
        np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=np.complex128),
        np.diag([1, 1, -2]).astype(np.complex128) / math.sqrt(3.0),
    ])
    return [item / math.sqrt(float(np.trace(item.conj().T @ item).real)) for item in basis]


OPERATOR_BASIS = operator_basis()


def transfer_matrix(operators: list[np.ndarray]) -> np.ndarray:
    transfer = np.empty((DIM * DIM, DIM * DIM), dtype=np.float64)
    for row, left in enumerate(OPERATOR_BASIS):
        for col, right in enumerate(OPERATOR_BASIS):
            value = np.trace(left.conj().T @ apply_channel(right, operators))
            if abs(value.imag) > 1e-10:
                raise ContractError("Hermitian operator-basis transfer coefficient became complex")
            transfer[row, col] = float(value.real)
    return transfer


def self_check() -> dict[str, object]:
    z = np.array([1.0, -0.5, 0.2])
    coupling = np.array([[0.0, 1e-5, -2e-5], [1e-5, 0.0, 3e-5], [-2e-5, 3e-5, 0.0]])
    operators, _, instrument_error = channel_kraus(z, coupling)
    completeness = sum((operator.conj().T @ operator for operator in operators),
                       start=np.zeros((DIM, DIM), dtype=np.complex128))
    choi = choi_matrix(operators)
    hermitian_error, min_eigenvalue, trace_preservation_error = choi_checks(choi)
    rho = np.eye(DIM, dtype=np.complex128) / DIM
    output = apply_channel(rho, operators)
    output_trace_error = abs(float(np.trace(output).real) - 1.0)
    output_min_eigenvalue = float(np.linalg.eigvalsh(output).min())
    transfer = transfer_matrix(operators)
    return {
        "passed": bool(instrument_error < 1e-12
                       and np.max(np.abs(completeness - np.eye(DIM))) < 1e-12
                       and hermitian_error < 1e-12 and min_eigenvalue >= -1e-12
                       and trace_preservation_error < 1e-12
                       and output_trace_error < 1e-12 and output_min_eigenvalue >= -1e-12
                       and np.isfinite(transfer).all()),
        "instrument_completeness_error": instrument_error,
        "channel_kraus_completeness_error": float(np.max(np.abs(completeness - np.eye(DIM)))),
        "choi_hermitian_error": hermitian_error,
        "choi_min_eigenvalue": min_eigenvalue,
        "trace_preservation_error": trace_preservation_error,
        "output_trace_error": output_trace_error,
        "output_min_eigenvalue": output_min_eigenvalue,
    }


def causal_samples(max_steps: int, max_samples: int) -> tuple[list[dict[str, object]], dict[str, int]]:
    times, prices = ql.load_prices()
    coupling = ql.load_coupling()
    first = int(np.searchsorted(times, coupling["times"][0], side="left"))
    while first < len(times) and (first == 0 or times[first] - times[first - 1] != ql.DT_NS):
        first += 1
    end = min(len(times) - 1, first + max_steps)
    if first >= end:
        raise ContractError("no causal sample span")
    cursor = int(np.searchsorted(coupling["times"], times[first], side="right") - 1)
    if cursor < 0:
        raise ContractError("no causal coupling at start")
    sigma2 = np.full(DIM, 1e-8, dtype=np.float64)
    stride = max(1, (end - first) // max_samples)
    samples: list[dict[str, object]] = []
    valid = gap = leading_gap = stale = 0
    for i in range(first, end):
        now_ns = int(times[i])
        next_ns = int(times[i + 1])
        while cursor + 1 < len(coupling["times"]) and coupling["times"][cursor + 1] <= now_ns:
            cursor += 1
        if i > first and now_ns - int(times[i - 1]) != ql.DT_NS:
            sigma2.fill(1e-8)
            gap += 1
            continue
        if next_ns - now_ns != ql.DT_NS:
            leading_gap += 1
            continue
        age_s = (now_ns - int(coupling["times"][cursor])) / 1_000_000_000.0
        if age_s > ql.MAX_COUPLING_AGE_S:
            sigma2.fill(1e-8)
            stale += 1
            continue
        ret = prices[i] - prices[i - 1]
        sigma2 = (1.0 - ql.EWMA_ALPHA) * sigma2 + ql.EWMA_ALPHA * ret * ret
        z = ret / np.sqrt(np.maximum(sigma2, 1e-16))
        valid += 1
        if valid % stride:
            continue
        samples.append({
            "timestamp": ql.utc_timestamp(now_ns),
            "coupling_update_time": ql.utc_timestamp(int(coupling["times"][cursor])),
            "coupling_age_s": age_s,
            "z": z.copy(),
            "coupling": coupling["matrices"][cursor].copy(),
        })
        if len(samples) >= max_samples:
            break
    return samples, {"valid_causal_updates_seen": valid, "gap_skips": gap,
                     "leading_gap_skips": leading_gap, "stale_coupling_skips": stale,
                     "sampling_stride_valid_updates": stride}


def run(max_steps: int, max_samples: int, out_dir: Path) -> dict[str, object]:
    if max_steps < 2 or max_samples < 1:
        raise ContractError("max_steps must be >=2 and max_samples must be >=1")
    samples, counts = causal_samples(max_steps, max_samples)
    if not samples:
        raise ContractError("no sampled causal channels")
    rows: list[dict[str, object]] = []
    max_instrument_error = max_choi_hermitian = max_tp_error = 0.0
    minimum_choi_eigenvalue = float("inf")
    maximum_traceless_radius = 0.0
    for sample in samples:
        operators, hamiltonian, instrument_error = channel_kraus(sample["z"], sample["coupling"])
        choi = choi_matrix(operators)
        choi_hermitian, choi_minimum, tp_error = choi_checks(choi)
        transfer = transfer_matrix(operators)
        radius = float(np.max(np.abs(np.linalg.eigvals(transfer[1:, 1:]))))
        max_instrument_error = max(max_instrument_error, instrument_error)
        max_choi_hermitian = max(max_choi_hermitian, choi_hermitian)
        max_tp_error = max(max_tp_error, tp_error)
        minimum_choi_eigenvalue = min(minimum_choi_eigenvalue, choi_minimum)
        maximum_traceless_radius = max(maximum_traceless_radius, radius)
        row: dict[str, object] = {
            "timestamp": sample["timestamp"],
            "coupling_update_time": sample["coupling_update_time"],
            "coupling_age_s": sample["coupling_age_s"],
            "instrument_completeness_error": instrument_error,
            "choi_hermitian_error": choi_hermitian,
            "choi_min_eigenvalue": choi_minimum,
            "trace_preservation_error": tp_error,
            "traceless_transfer_spectral_radius": radius,
            "hamiltonian_frobenius_norm": float(np.linalg.norm(hamiltonian)),
        }
        for name, values in (("z", sample["z"]), ("h_diag", np.diag(hamiltonian).real)):
            for pair, value in zip(ql.PAIRS, values, strict=True):
                row[f"{name}_{pair.lower()}"] = float(value)
        rows.append(row)
    if (max_instrument_error >= 1e-12 or max_choi_hermitian >= 1e-12
            or max_tp_error >= 1e-12 or minimum_choi_eigenvalue < -1e-12):
        raise ContractError("sampled channel failed complete-positivity/trace-preservation contract")
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / "quantum_process_tomography.parquet"
    summary_path = out_dir / "quantum_process_tomography_summary.json"
    pd.DataFrame(rows).to_parquet(artifact, index=False, compression="zstd")
    summary = {
        "version": VERSION,
        "interpretation": (
            "tomography of a data-conditioned mathematical qutrit channel; not evidence "
            "of physical quantum FX dynamics or a forecast/trading signal"),
        "channel": (
            "nonselective composition of a unitary, complete two-outcome Kraus instrument, "
            "and qutrit pure dephasing; the conditional normalized filter itself is nonlinear "
            "and is intentionally not represented as a standalone CPTP channel"),
        "pair_scope": list(ql.PAIRS),
        "samples": len(rows),
        "causal_replay_counts": counts,
        "channel_checks": {
            "max_instrument_completeness_error": max_instrument_error,
            "max_choi_hermitian_error": max_choi_hermitian,
            "minimum_choi_eigenvalue": minimum_choi_eigenvalue,
            "max_trace_preservation_error": max_tp_error,
            "max_traceless_transfer_spectral_radius": maximum_traceless_radius,
        },
        "promotion_status": "numerical channel validation only; does not promote OQ-13",
        "output": str(artifact.relative_to(ROOT)).replace("\\", "/"),
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        result = self_check()
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    try:
        run(args.max_steps, args.max_samples, args.out_dir.resolve())
        return 0
    except (ContractError, ValueError, np.linalg.LinAlgError, OSError) as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

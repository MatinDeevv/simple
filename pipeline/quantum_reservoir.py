"""Causal 10-qubit quantum-reservoir software experiment for the FX data lake.

This uses actual state-vector circuit mathematics (complex amplitudes, unitary
data re-uploading rotations, and a fixed entangling ring) as a *statistical
representation*.  It is not a claim that markets are physical quantum systems,
and it is not part of any trading or canonical state path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "data_canonical"
DERIVED_DIR = ROOT / "data_derived"

PAIRS = (
    "AUDUSD", "EURGBP", "EURJPY", "EURUSD", "GBPJPY",
    "GBPUSD", "USDCAD", "USDCHF", "USDCNH", "USDJPY",
)
N_QUBITS = len(PAIRS)
DIMENSION = 1 << N_QUBITS
LAYERS = 3
N_RING = N_QUBITS
N_OBSERVABLES = N_QUBITS + N_RING + 1  # <Z_q>, <Z_q Z_(q+1)>, global parity

DT_NS = 60_000_000_000
HALFLIFE_STEPS = 1_440.0
EWMA_ALPHA = 1.0 - math.exp(-math.log(2.0) / HALFLIFE_STEPS)
MAX_STANDARDIZED_RETURN = 5.0

DEFAULT_MAX_STEPS = 50_000
DEFAULT_SEED = 20260718
DEFAULT_OOS_FRACTION = 0.70
DEFAULT_RIDGE = 5.0
VERSION = "quantum-reservoir-1.0.0"


class ContractError(RuntimeError):
    """The experimental causal or numerical contract was violated."""


@dataclass(frozen=True)
class GateLayout:
    """Precomputed basis index and Pauli-Z signs for a 10-qubit state vector."""

    z_signs: np.ndarray
    single_zero: tuple[np.ndarray, ...]
    single_one: tuple[np.ndarray, ...]
    cnot_zero: tuple[np.ndarray, ...]
    cnot_one: tuple[np.ndarray, ...]
    zz_signs: np.ndarray
    observable_signs: np.ndarray
    ring_cnot_source: np.ndarray


@dataclass(frozen=True)
class ReservoirParameters:
    """Fixed random circuit parameters, sampled once before reading targets."""

    theta_weights: np.ndarray
    theta_bias: np.ndarray
    phi_weights: np.ndarray
    phi_bias: np.ndarray
    zz_angles: np.ndarray


def epoch_ns(values: object) -> np.ndarray:
    return pd.DatetimeIndex(values).as_unit("ns").asi8


def utc_timestamp(ns: int) -> pd.Timestamp:
    return pd.Timestamp(ns, unit="ns", tz="UTC")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def make_layout() -> GateLayout:
    """Build index maps without relying on a quantum SDK or bit-order ambiguity."""
    basis = np.arange(DIMENSION, dtype=np.int64)
    z_rows: list[np.ndarray] = []
    zero_rows: list[np.ndarray] = []
    one_rows: list[np.ndarray] = []
    for qubit in range(N_QUBITS):
        mask = 1 << qubit
        signs = np.where((basis & mask) == 0, 1.0, -1.0).astype(np.float64)
        zero = basis[(basis & mask) == 0]
        z_rows.append(signs)
        zero_rows.append(zero)
        one_rows.append(zero | mask)

    cnot_zero: list[np.ndarray] = []
    cnot_one: list[np.ndarray] = []
    zz_rows: list[np.ndarray] = []
    for control in range(N_QUBITS):
        target = (control + 1) % N_QUBITS
        control_mask = 1 << control
        target_mask = 1 << target
        source = basis[
            ((basis & control_mask) != 0) & ((basis & target_mask) == 0)
        ]
        cnot_zero.append(source)
        cnot_one.append(source | target_mask)
        zz_rows.append(z_rows[control] * z_rows[target])

    z_signs = np.stack(z_rows, axis=0)
    zz_signs = np.stack(zz_rows, axis=0)
    # Compose the complete directed CNOT ring once.  For this permutation p,
    # state_after[j] = state_before[p[j]].  This is exactly equivalent to the
    # ten individual CNOT gates and removes ten Python/NumPy dispatches per layer.
    output_by_input = basis.copy()
    for control in range(N_QUBITS):
        target = (control + 1) % N_QUBITS
        control_mask = 1 << control
        target_mask = 1 << target
        flip = (output_by_input & control_mask) != 0
        output_by_input[flip] ^= target_mask
    ring_cnot_source = np.empty(DIMENSION, dtype=np.int64)
    ring_cnot_source[output_by_input] = basis
    global_parity = np.prod(z_signs, axis=0, dtype=np.float64)[None, :]
    observable_signs = np.concatenate((z_signs, zz_signs, global_parity), axis=0)
    if observable_signs.shape != (N_OBSERVABLES, DIMENSION):
        raise ContractError("observable layout shape mismatch")
    return GateLayout(
        z_signs=z_signs,
        single_zero=tuple(zero_rows),
        single_one=tuple(one_rows),
        cnot_zero=tuple(cnot_zero),
        cnot_one=tuple(cnot_one),
        zz_signs=zz_signs,
        observable_signs=observable_signs,
        ring_cnot_source=ring_cnot_source,
    )


def make_parameters(seed: int) -> ReservoirParameters:
    """Sample fixed circuit weights before source returns/targets are examined."""
    rng = np.random.default_rng(seed)
    return ReservoirParameters(
        theta_weights=rng.normal(0.0, 0.55, size=(LAYERS, N_QUBITS, N_QUBITS)),
        theta_bias=rng.uniform(-math.pi, math.pi, size=(LAYERS, N_QUBITS)),
        phi_weights=rng.normal(0.0, 0.40, size=(LAYERS, N_QUBITS, N_QUBITS)),
        phi_bias=rng.uniform(-math.pi, math.pi, size=(LAYERS, N_QUBITS)),
        zz_angles=rng.uniform(-0.36, 0.36, size=(LAYERS, N_RING)),
    )


def parameter_hash(parameters: ReservoirParameters) -> str:
    digest = hashlib.sha256()
    for array in (
        parameters.theta_weights, parameters.theta_bias, parameters.phi_weights,
        parameters.phi_bias, parameters.zz_angles,
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def entangler_phases(parameters: ReservoirParameters, layout: GateLayout) -> np.ndarray:
    """Precompile each layer's commuting local RZZ ring into diagonal phases."""
    phase_angles = parameters.zz_angles @ layout.zz_signs
    return np.exp(-0.5j * phase_angles)


def zero_state() -> np.ndarray:
    state = np.zeros(DIMENSION, dtype=np.complex128)
    state[0] = 1.0
    return state


def apply_ry_inplace(state: np.ndarray, qubit: int, angle: float,
                     layout: GateLayout) -> None:
    """Apply exp(-i angle Y_q / 2) using paired basis amplitudes."""
    zero = layout.single_zero[qubit]
    one = layout.single_one[qubit]
    old_zero = state[zero].copy()
    old_one = state[one].copy()
    c = math.cos(0.5 * angle)
    s = math.sin(0.5 * angle)
    state[zero] = c * old_zero - s * old_one
    state[one] = s * old_zero + c * old_one


def apply_rz_inplace(state: np.ndarray, qubit: int, angle: float,
                     layout: GateLayout) -> None:
    """Apply exp(-i angle Z_q / 2), diagonal in the computational basis."""
    state *= np.exp(-0.5j * angle * layout.z_signs[qubit])


def apply_ry_rz_inplace(state: np.ndarray, qubit: int, theta: float,
                        phi: float, layout: GateLayout) -> None:
    """Fuse the sequential RY then RZ into one local 2x2 complex unitary."""
    zero = layout.single_zero[qubit]
    one = layout.single_one[qubit]
    old_zero = state[zero].copy()
    old_one = state[one].copy()
    c = math.cos(0.5 * theta)
    s = math.sin(0.5 * theta)
    phase_zero = np.exp(-0.5j * phi)
    phase_one = np.exp(0.5j * phi)
    state[zero] = phase_zero * (c * old_zero - s * old_one)
    state[one] = phase_one * (s * old_zero + c * old_one)


def apply_cnot_inplace(state: np.ndarray, edge: int, layout: GateLayout) -> None:
    """Apply the fixed CNOT q -> (q+1 mod 10) ring edge."""
    zero = layout.cnot_zero[edge]
    one = layout.cnot_one[edge]
    old_zero = state[zero].copy()
    state[zero] = state[one]
    state[one] = old_zero


def apply_rzz_inplace(state: np.ndarray, edge: int, angle: float,
                      layout: GateLayout) -> None:
    """Apply exp(-i angle Z_q Z_(q+1) / 2) on one local ring edge."""
    state *= np.exp(-0.5j * angle * layout.zz_signs[edge])


def advance_reservoir(state: np.ndarray, standardized_return: np.ndarray,
                      parameters: ReservoirParameters, layout: GateLayout,
                      compiled_entangler_phases: np.ndarray) -> None:
    """Causally re-upload one return vector through three unitary circuit layers."""
    if state.shape != (DIMENSION,) or not np.isfinite(state).all():
        raise ContractError("state vector is malformed before circuit advancement")
    z = np.clip(np.asarray(standardized_return, dtype=np.float64),
                -MAX_STANDARDIZED_RETURN, MAX_STANDARDIZED_RETURN)
    if z.shape != (N_QUBITS,) or not np.isfinite(z).all():
        raise ContractError("causal encoded return vector is malformed")

    for layer in range(LAYERS):
        # Each layer re-encodes the same current causal observation.  Parameters
        # are fixed from seed and never fitted to the next-bar target.
        theta = 1.20 * np.tanh(parameters.theta_weights[layer] @ z
                               + parameters.theta_bias[layer])
        phi = 1.20 * np.tanh(parameters.phi_weights[layer] @ z
                             + parameters.phi_bias[layer])
        for qubit in range(N_QUBITS):
            apply_ry_rz_inplace(state, qubit, float(theta[qubit]),
                                float(phi[qubit]), layout)
        state *= compiled_entangler_phases[layer]
        state[:] = state[layout.ring_cnot_source]

    norm = float(np.vdot(state, state).real)
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-10:
        raise ContractError(f"unitary circuit did not preserve norm: {norm!r}")


def observables(state: np.ndarray, layout: GateLayout) -> np.ndarray:
    """Return real Pauli-Z, nearest-ring ZZ, and global parity expectations."""
    probability = np.abs(state) ** 2
    result = layout.observable_signs @ probability
    if result.shape != (N_OBSERVABLES,) or not np.isfinite(result).all():
        raise ContractError("measurement produced non-finite observables")
    if np.any(np.abs(result) > 1.0 + 1e-12):
        raise ContractError("Pauli observable escaped the physical [-1, 1] range")
    return result.astype(np.float64, copy=False)


def load_prices() -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Inner-join the ten canonical price histories only; no targets are loaded."""
    joined: pd.DataFrame | None = None
    source_hashes: dict[str, str] = {}
    for pair in PAIRS:
        path = CANONICAL_DIR / f"{pair}.parquet"
        source_hashes[f"data_canonical/{pair}.parquet"] = sha256_file(path)
        frame = pd.read_parquet(path, columns=["timestamp", "close"])
        frame = frame.rename(columns={"close": pair})
        joined = frame if joined is None else joined.merge(
            frame, on="timestamp", how="inner", validate="one_to_one"
        )
    assert joined is not None
    joined = joined.sort_values("timestamp", kind="stable").reset_index(drop=True)
    times = epoch_ns(joined["timestamp"])
    log_close = np.log(joined.loc[:, list(PAIRS)].to_numpy(dtype=np.float64))
    if (len(times) < 10 or not np.all(np.diff(times) > 0)
            or not np.isfinite(log_close).all()):
        raise ContractError("canonical ten-pair input is invalid")
    return times, log_close, source_hashes


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponents = np.exp(np.clip(shifted, -700.0, 0.0))
    return exponents / exponents.sum(axis=1, keepdims=True)


def brier(probabilities: np.ndarray, targets: np.ndarray) -> float:
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(targets)), targets] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def log_loss(probabilities: np.ndarray, targets: np.ndarray) -> float:
    return float(-np.mean(np.log(np.maximum(probabilities[np.arange(len(targets)), targets],
                                          1e-15))))


def fit_ridge_readout(features: np.ndarray, targets: np.ndarray,
                      ridge: float) -> dict[str, np.ndarray]:
    """Fit only pre-OOS features/targets; OOS data never enters this routine."""
    if len(features) != len(targets) or len(features) < N_OBSERVABLES + 2:
        raise ContractError("insufficient matched pre-OOS samples for ridge readout")
    if ridge <= 0.0:
        raise ContractError("ridge penalty must be positive")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (features - mean) / scale
    design = np.column_stack((np.ones(len(standardized)), standardized))
    labels = np.zeros((len(targets), N_QUBITS), dtype=np.float64)
    labels[np.arange(len(targets)), targets] = 1.0
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    try:
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ labels)
    except np.linalg.LinAlgError as exc:
        raise ContractError(f"ridge readout solve failed: {exc}") from exc
    if not np.isfinite(weights).all():
        raise ContractError("ridge readout weights are non-finite")
    return {"mean": mean, "scale": scale, "weights": weights}


def predict_ridge_readout(features: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    standardized = (features - model["mean"]) / model["scale"]
    design = np.column_stack((np.ones(len(standardized)), standardized))
    return softmax(design @ model["weights"])


def empty_minute_record(timestamp_ns: int, row_index: int, previous_ns: int,
                        next_ns: int, phase: str, previous_contiguous: bool,
                        next_contiguous: bool, reason: str, reset: bool) -> dict[str, object]:
    record: dict[str, object] = {
        "timestamp": utc_timestamp(timestamp_ns),
        "row_index": int(row_index),
        "previous_timestamp": utc_timestamp(previous_ns),
        "next_timestamp": utc_timestamp(next_ns),
        "chronological_phase": phase,
        "previous_contiguous_60s": bool(previous_contiguous),
        "next_contiguous_60s": bool(next_contiguous),
        "reason": reason,
        "state_reset": bool(reset),
        "target_next_largest_abs_return_index": -1,
        "is_oos_evaluation": False,
        "state_norm_error": float("nan"),
        "model_predicted_index": -1,
    }
    for pair in PAIRS:
        record[f"log_return_{pair.lower()}"] = float("nan")
        record[f"z_{pair.lower()}"] = float("nan")
        record[f"model_p_{pair.lower()}"] = float("nan")
    for qubit in range(N_QUBITS):
        record[f"observable_z_q{qubit}"] = float("nan")
        record[f"observable_zz_ring_q{qubit}_q{(qubit + 1) % N_QUBITS}"] = float("nan")
    record["observable_global_z_parity"] = float("nan")
    return record


def daily_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Summarize persisted minute rows without treating train predictions as OOS scores."""
    scored = frame[frame["reason"] == "updated_and_target_available"].copy()
    if scored.empty:
        raise ContractError("cannot build daily artifact without updated rows")
    scored["day"] = scored["timestamp"].dt.floor("D")
    result: list[dict[str, object]] = []
    for day, subset in scored.groupby("day", sort=True):
        oos = subset[subset["is_oos_evaluation"]]
        row: dict[str, object] = {
            "timestamp": day,
            "reservoir_updates": int(len(subset)),
            "oos_evaluated_updates": int(len(oos)),
            "mean_state_norm_error": float(subset["state_norm_error"].mean()),
            "mean_global_z_parity": float(subset["observable_global_z_parity"].mean()),
        }
        for qubit in range(N_QUBITS):
            row[f"mean_z_q{qubit}"] = float(subset[f"observable_z_q{qubit}"].mean())
            row[f"mean_zz_ring_q{qubit}_q{(qubit + 1) % N_QUBITS}"] = float(
                subset[f"observable_zz_ring_q{qubit}_q{(qubit + 1) % N_QUBITS}"].mean()
            )
        if not oos.empty:
            probabilities = oos[[f"model_p_{pair.lower()}" for pair in PAIRS]].to_numpy()
            targets = oos["target_next_largest_abs_return_index"].to_numpy(dtype=np.int64)
            row["oos_top1_accuracy"] = float(
                np.mean(np.argmax(probabilities, axis=1) == targets)
            )
            row["oos_brier_score"] = brier(probabilities, targets)
        else:
            row["oos_top1_accuracy"] = float("nan")
            row["oos_brier_score"] = float("nan")
        result.append(row)
    return result


def self_check() -> dict[str, object]:
    """Numerical and causal-free invariant checks for the state-vector machinery."""
    layout = make_layout()
    parameters = make_parameters(DEFAULT_SEED)
    compiled_phases = entangler_phases(parameters, layout)
    state_a = zero_state()
    state_b = zero_state()
    z = np.linspace(-1.0, 1.0, N_QUBITS)
    advance_reservoir(state_a, z, parameters, layout, compiled_phases)
    advance_reservoir(state_b, z, parameters, layout, compiled_phases)
    measured = observables(state_a, layout)
    initial_observables = observables(zero_state(), layout)

    # A separate known RY(pi) gate checks basis-pair orientation.
    basis_test = zero_state()
    apply_ry_inplace(basis_test, 0, math.pi, layout)
    ry_probability_error = float(abs(abs(basis_test[1]) ** 2 - 1.0))
    cnot_test = np.zeros(DIMENSION, dtype=np.complex128)
    cnot_test[1] = 1.0
    apply_cnot_inplace(cnot_test, 0, layout)
    cnot_probability_error = float(abs(abs(cnot_test[3]) ** 2 - 1.0))

    # Verify that the compiled diagonal RZZ ring and compiled CNOT permutation
    # equal the declared gate-by-gate circuit for one random normalized state.
    rng = np.random.default_rng(17)
    raw = rng.normal(size=DIMENSION) + 1j * rng.normal(size=DIMENSION)
    manual_ring = raw / np.linalg.norm(raw)
    compiled_ring = manual_ring.copy()
    for edge in range(N_RING):
        apply_rzz_inplace(
            manual_ring, edge, float(parameters.zz_angles[0, edge]), layout
        )
    for edge in range(N_RING):
        apply_cnot_inplace(manual_ring, edge, layout)
    compiled_ring *= compiled_phases[0]
    compiled_ring[:] = compiled_ring[layout.ring_cnot_source]
    compiled_ring_equivalence_error = float(
        np.max(np.abs(manual_ring - compiled_ring))
    )
    deterministic = bool(np.array_equal(state_a, state_b))
    state_norm_error = abs(float(np.vdot(state_a, state_a).real) - 1.0)
    observable_range_error = float(max(0.0, np.max(np.abs(measured)) - 1.0))
    initial_error = float(np.max(np.abs(initial_observables - 1.0)))

    passed = bool(
        deterministic
        and state_norm_error < 1e-12
        and observable_range_error < 1e-12
        and initial_error < 1e-12
        and ry_probability_error < 1e-12
        and cnot_probability_error < 1e-12
        and compiled_ring_equivalence_error < 1e-12
        and parameter_hash(parameters) == parameter_hash(make_parameters(DEFAULT_SEED))
    )
    return {
        "passed": passed,
        "dimension": DIMENSION,
        "qubits": N_QUBITS,
        "layers": LAYERS,
        "state_norm_error": state_norm_error,
        "observable_range_error": observable_range_error,
        "initial_z_and_zz_and_parity_error": initial_error,
        "ry_pi_probability_error": ry_probability_error,
        "cnot_probability_error": cnot_probability_error,
        "compiled_ring_equivalence_error": compiled_ring_equivalence_error,
        "seed_reproducible": deterministic,
        "parameter_hash": parameter_hash(parameters),
    }


def run(max_steps: int, seed: int, oos_fraction: float, ridge: float,
        out_dir: Path) -> dict[str, object]:
    if max_steps < 100:
        raise ContractError("--max-steps must be at least 100")
    if not 0.0 < oos_fraction < 1.0:
        raise ContractError("--oos-fraction must lie strictly between zero and one")
    layout = make_layout()
    parameters = make_parameters(seed)
    compiled_phases = entangler_phases(parameters, layout)
    times, log_close, input_hashes = load_prices()

    # The bounded run is the latest requested history.  Its last 30% is an
    # immutable chronological OOS segment; all train statistics are frozen first.
    end = len(times) - 1
    start = max(1, end - max_steps)
    if end - start < 100:
        raise ContractError("canonical input is too short for requested reservoir run")
    oos_start = start + int((end - start) * oos_fraction)
    if not start < oos_start < end:
        raise ContractError("chronological split is malformed")

    state = zero_state()
    sigma2 = np.full(N_QUBITS, 1e-8, dtype=np.float64)
    minute: list[dict[str, object]] = []
    feature_rows: list[np.ndarray] = []
    target_rows: list[int] = []
    record_rows: list[int] = []
    raw_row_indices: list[int] = []
    state_resets = 0
    skipped_cross_gap_updates = 0
    max_norm_error = 0.0
    updates = 0

    for i in range(start, end):
        now_ns = int(times[i])
        previous_ns = int(times[i - 1])
        next_ns = int(times[i + 1])
        previous_contiguous = now_ns - previous_ns == DT_NS
        next_contiguous = next_ns - now_ns == DT_NS
        phase = "oos" if i >= oos_start else "train"
        if not previous_contiguous:
            # Never form x_t - x_(t-1), encode it, or let state memory bridge a gap.
            state = zero_state()
            sigma2.fill(1e-8)
            state_resets += 1
            skipped_cross_gap_updates += 1
            minute.append(empty_minute_record(
                now_ns, i, previous_ns, next_ns, phase, previous_contiguous,
                next_contiguous, "post_gap_reset_skip", True,
            ))
            continue
        if not next_contiguous:
            # This row cannot receive a valid t+1 target and is not encoded.
            minute.append(empty_minute_record(
                now_ns, i, previous_ns, next_ns, phase, previous_contiguous,
                next_contiguous, "pre_gap_target_skip", False,
            ))
            continue

        returns = log_close[i] - log_close[i - 1]
        sigma2 = (1.0 - EWMA_ALPHA) * sigma2 + EWMA_ALPHA * returns * returns
        standardized = returns / np.sqrt(np.maximum(sigma2, 1e-16))
        advance_reservoir(state, standardized, parameters, layout, compiled_phases)
        measured = observables(state, layout)
        norm_error = abs(float(np.vdot(state, state).real) - 1.0)
        max_norm_error = max(max_norm_error, norm_error)
        next_returns = log_close[i + 1] - log_close[i]
        target = int(np.argmax(np.abs(next_returns)))

        record = empty_minute_record(
            now_ns, i, previous_ns, next_ns, phase, previous_contiguous,
            next_contiguous, "updated_and_target_available", False,
        )
        record["target_next_largest_abs_return_index"] = target
        record["state_norm_error"] = norm_error
        for pair_index, pair in enumerate(PAIRS):
            record[f"log_return_{pair.lower()}"] = float(returns[pair_index])
            record[f"z_{pair.lower()}"] = float(standardized[pair_index])
        for qubit in range(N_QUBITS):
            record[f"observable_z_q{qubit}"] = float(measured[qubit])
            record[f"observable_zz_ring_q{qubit}_q{(qubit + 1) % N_QUBITS}"] = float(
                measured[N_QUBITS + qubit]
            )
        record["observable_global_z_parity"] = float(measured[-1])
        minute.append(record)
        feature_rows.append(measured)
        target_rows.append(target)
        record_rows.append(len(minute) - 1)
        raw_row_indices.append(i)
        updates += 1

    if updates < 100:
        raise ContractError("too few contiguous reservoir updates")
    features = np.vstack(feature_rows)
    targets = np.asarray(target_rows, dtype=np.int64)
    raw_indices = np.asarray(raw_row_indices, dtype=np.int64)
    train_mask = raw_indices < oos_start
    oos_mask = ~train_mask
    if int(train_mask.sum()) < N_OBSERVABLES + 2 or int(oos_mask.sum()) < 10:
        raise ContractError("gap-safe split left insufficient train or OOS samples")
    if np.any(raw_indices[train_mask] >= oos_start) or np.any(raw_indices[oos_mask] < oos_start):
        raise ContractError("chronological split leakage check failed")

    # This is the sole learned component.  It only sees train-fold features and
    # labels, while the quantum circuit itself was fixed from seed before all data.
    model = fit_ridge_readout(features[train_mask], targets[train_mask], ridge)
    oos_probabilities = predict_ridge_readout(features[oos_mask], model)
    train_counts = np.bincount(targets[train_mask], minlength=N_QUBITS)
    if int(train_counts.sum()) != int(train_mask.sum()):
        raise ContractError("train target count mismatch")
    prior = train_counts / train_counts.sum()
    constant_index = int(np.argmax(prior))
    oos_targets = targets[oos_mask]
    uniform = np.full((len(oos_targets), N_QUBITS), 1.0 / N_QUBITS)
    prior_matrix = np.repeat(prior[None, :], len(oos_targets), axis=0)
    constant = np.zeros_like(uniform)
    constant[:, constant_index] = 1.0

    all_oos_positions = np.flatnonzero(oos_mask)
    for local_position, feature_position in enumerate(all_oos_positions):
        record = minute[record_rows[int(feature_position)]]
        probabilities = oos_probabilities[local_position]
        record["is_oos_evaluation"] = True
        record["model_predicted_index"] = int(np.argmax(probabilities))
        for pair_index, pair in enumerate(PAIRS):
            record[f"model_p_{pair.lower()}"] = float(probabilities[pair_index])

    # Persist the model's complete fixed run configuration without presenting a
    # train score as a holdout result.
    readout_hash = sha256_array(model["weights"])
    oos_top1 = float(np.mean(np.argmax(oos_probabilities, axis=1) == oos_targets))
    oos_brier = brier(oos_probabilities, oos_targets)
    oos_log_loss = log_loss(oos_probabilities, oos_targets)
    baseline_metrics = {
        "uniform": {
            # A tied uniform probability vector has no deterministic top-1;
            # use randomized-tie expected accuracy, not class-zero frequency.
            "top1_accuracy": 1.0 / N_QUBITS,
            "brier_score": brier(uniform, oos_targets),
            "log_loss": log_loss(uniform, oos_targets),
        },
        "frozen_train_class_prior": {
            "probabilities": prior.tolist(),
            "top1_accuracy": float(np.mean(np.argmax(prior_matrix, axis=1) == oos_targets)),
            "brier_score": brier(prior_matrix, oos_targets),
            "log_loss": log_loss(prior_matrix, oos_targets),
        },
        "frozen_train_modal_class": {
            "target_index": constant_index,
            "target_pair": PAIRS[constant_index],
            "top1_accuracy": float(np.mean(np.argmax(constant, axis=1) == oos_targets)),
            "brier_score": brier(constant, oos_targets),
            "log_loss": log_loss(constant, oos_targets),
        },
    }
    best_baseline_brier = min(item["brier_score"] for item in baseline_metrics.values())
    best_baseline_top1 = max(item["top1_accuracy"] for item in baseline_metrics.values())
    passes_fixed_baselines = bool(
        oos_brier < best_baseline_brier and oos_top1 > best_baseline_top1
    )
    if max_norm_error >= 1e-10:
        raise ContractError("state-vector norm contract failed in real run")

    out_dir.mkdir(parents=True, exist_ok=True)
    daily_path = out_dir / "quantum_reservoir_daily.parquet"
    minute_path = out_dir / "quantum_reservoir_minute.parquet"
    validation_path = out_dir / "quantum_reservoir_validation.json"
    summary_path = out_dir / "quantum_reservoir_summary.json"
    minute_frame = pd.DataFrame(minute)
    minute_frame.to_parquet(minute_path, index=False, compression="zstd")
    pd.DataFrame(daily_rows(minute_frame)).to_parquet(
        daily_path, index=False, compression="zstd"
    )
    validation = self_check()
    validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    if not validation["passed"]:
        raise ContractError("quantum reservoir self-check failed")

    source_hashes = {"pipeline/quantum_reservoir.py": sha256_file(Path(__file__))}
    source_hashes.update(input_hashes)
    summary = {
        "version": VERSION,
        "interpretation": (
            "fixed quantum-circuit reservoir software representation; not evidence "
            "of physical quantum FX dynamics, causality, tradability, or a "
            "production model"
        ),
        "pair_scope": list(PAIRS),
        "run_window": {
            "selection": "latest bounded canonical window",
            "start": utc_timestamp(int(times[start])).isoformat(),
            "end": utc_timestamp(int(times[end])).isoformat(),
            "raw_rows_processed": int(end - start),
            "fixed_oos_split": "final 30% chronological raw rows",
            "oos_prediction_start": utc_timestamp(int(times[oos_start])).isoformat(),
        },
        "circuit": {
            "backend": "NumPy complex128 state vector; no quantum SDK or hardware",
            "qubits": N_QUBITS,
            "hilbert_dimension": DIMENSION,
            "data_reupload_layers": LAYERS,
            "per_layer": (
                "10 fixed-seeded data-dependent RY + 10 RZ rotations, "
                "then 10 fixed RZZ and 10 CNOT nearest-neighbour ring gates"
            ),
            "observables": (
                "10 Pauli-Z expectations, 10 nearest-neighbour ZZ correlations, "
                "and one global Z parity"
            ),
            "seed": seed,
            "fixed_parameter_sha256": parameter_hash(parameters),
        },
        "causality": {
            "input_at_t": (
                "log close returns through t only, standardized with a resettable "
                "causal 1,440-step EWMA"
            ),
            "target": "which of ten pairs has largest absolute valid t+1 log return",
            "target_excluded_from_circuit": True,
            "ridge_readout_fit_rows": int(train_mask.sum()),
            "ridge_readout_oos_rows": int(oos_mask.sum()),
            "feature_standardization_fit_on_train_only": True,
            "fixed_before_oos": ["circuit seed/parameters", "ridge weights", "baselines"],
            "cross_gap_state_updates": 0,
            "gap_state_resets": state_resets,
            "cross_gap_skipped_rows": skipped_cross_gap_updates,
        },
        "readout": {
            "type": "multiclass one-hot ridge classifier followed by softmax",
            "ridge_penalty": ridge,
            "feature_count": N_OBSERVABLES,
            "readout_weight_sha256": readout_hash,
            "train_target_counts": train_counts.tolist(),
            "weights": model["weights"].tolist(),
            "feature_mean": model["mean"].tolist(),
            "feature_scale": model["scale"].tolist(),
        },
        "physics_numerics": {
            "max_state_norm_error": max_norm_error,
            "state_resets": state_resets,
            "self_check_passed": validation["passed"],
        },
        "next_bar_oos_experimental_metric": {
            "samples": int(len(oos_targets)),
            "top1_accuracy": oos_top1,
            "brier_score": oos_brier,
            "log_loss": oos_log_loss,
            "fixed_baselines": baseline_metrics,
            "passes_all_fixed_baselines": passes_fixed_baselines,
            "promotion": (
                "rejected unless it beats every frozen baseline on both Brier and "
                "top-1, then separately passes red-team and untouched-holdout gates"
            ),
        },
        "source_sha256": source_hashes,
        "outputs": {
            "daily": str(daily_path.relative_to(ROOT)).replace("\\", "/"),
            "minute": str(minute_path.relative_to(ROOT)).replace("\\", "/"),
            "validation": str(validation_path.relative_to(ROOT)).replace("\\", "/"),
        },
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--oos-fraction", type=float, default=DEFAULT_OOS_FRACTION)
    parser.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    parser.add_argument("--out-dir", type=Path, default=DERIVED_DIR)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        result = self_check()
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    try:
        run(args.max_steps, args.seed, args.oos_fraction, args.ridge,
            args.out_dir.resolve())
        return 0
    except (ContractError, ValueError, OSError, np.linalg.LinAlgError) as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

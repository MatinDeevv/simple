"""Causal 10-qubit quantum-fidelity-kernel experiment for the FX data lake.

This is an isolated *classical simulation of quantum circuit mathematics*.
It makes no claim that FX markets are quantum physical systems and does not
write to the canonical simulator or trading surface.
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

from contracts import ContractError as SharedContractError
from contracts import canonical_pair_order, contiguous_60s, validate_generated_manifest


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "data_canonical"
DERIVED_DIR = ROOT / "data_derived"
MANIFEST_PATH = CANONICAL_DIR / "manifest.json"
COUPLING_PATH = DERIVED_DIR / "coupling_estimates.parquet"

# This is the canonical manifest order and is intentionally also the qubit order.
PAIRS = canonical_pair_order(ROOT)
N_QUBITS = len(PAIRS)
DIMENSION = 1 << N_QUBITS
DT_NS = 60_000_000_000
DT_S = 60.0
HALFLIFE_STEPS = 1_440.0
EWMA_ALPHA = 1.0 - math.exp(-math.log(2.0) / HALFLIFE_STEPS)
MAX_COUPLING_AGE_S = 36.0 * 3_600.0

DEFAULT_MAX_STEPS = 250_000
DEFAULT_MAX_TRAIN_SAMPLES = 384
DEFAULT_MAX_OOS_SAMPLES = 192
DEFAULT_LANDMARKS = 64
DEFAULT_SEED = 20260718
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_RIDGE = 1.0e-2
VERSION = "quantum-kernel-1.0.0"


class ContractError(RuntimeError):
    """A causal, numerical, or artifact contract failed."""


@dataclass(frozen=True)
class CausalCandidates:
    """All valid causal observations before bounded sampling."""

    source_rows: np.ndarray
    timestamps_ns: np.ndarray
    target_timestamps_ns: np.ndarray
    raw_z: np.ndarray
    coupling_rows: np.ndarray
    targets: np.ndarray
    last_magnitude: np.ndarray


def utc_timestamp(ns: int) -> pd.Timestamp:
    return pd.Timestamp(ns, unit="ns", tz="UTC")


def epoch_ns(values: object) -> np.ndarray:
    return pd.DatetimeIndex(values).as_unit("ns").asi8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def manifest_provenance() -> dict[str, object]:
    manifest = validate_generated_manifest(ROOT)
    return {
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "canonical_data_sha256": {pair: manifest["pairs"][pair]["data_sha256"]
                                  for pair in PAIRS},
        "coupling_estimates_sha256": sha256_file(COUPLING_PATH),
    }


def load_window(max_steps: int, first_coupling_time: pd.Timestamp) -> tuple[np.ndarray, np.ndarray]:
    """Load a bounded common 10-pair window instead of the full 10-year lake."""
    reference = pd.read_parquet(CANONICAL_DIR / f"{PAIRS[0]}.parquet",
                                columns=["timestamp"])
    ref_ns = epoch_ns(reference["timestamp"])
    first_ref = int(np.searchsorted(ref_ns, first_coupling_time.value, side="left"))
    # The extra rows absorb sparse pair-specific timestamp differences before inner join.
    # USDCNH and cross pairs can remove materially more than a few thousand rows.
    upper_ref = min(len(reference) - 1, first_ref + max_steps + max(50_000, max_steps // 5))
    start = reference.iloc[max(0, first_ref - 4)]["timestamp"]
    stop = reference.iloc[upper_ref]["timestamp"]
    joined: pd.DataFrame | None = None
    for pair in PAIRS:
        frame = pd.read_parquet(
            CANONICAL_DIR / f"{pair}.parquet", columns=["timestamp", "close"],
            filters=[("timestamp", ">=", start), ("timestamp", "<=", stop)],
        ).rename(columns={"close": pair})
        joined = frame if joined is None else joined.merge(
            frame, on="timestamp", how="inner", validate="one_to_one"
        )
    assert joined is not None
    joined = joined.sort_values("timestamp", kind="stable").reset_index(drop=True)
    times = epoch_ns(joined["timestamp"])
    close = joined.loc[:, list(PAIRS)].to_numpy(dtype=np.float64)
    if (len(times) < max_steps + 2 or not np.all(np.diff(times) > 0)
            or not np.isfinite(close).all() or np.any(close <= 0.0)):
        raise ContractError("bounded common canonical window is invalid or too short")
    return times, np.log(close)


def load_coupling() -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(COUPLING_PATH)
    required = {"update_time", "affected_index", "source_index", "c_ij_s_minus_2"}
    if not required.issubset(frame.columns):
        raise ContractError("coupling-estimate schema is incomplete")
    if set(frame["affected_symbol"].unique()) != set(PAIRS):
        raise ContractError("coupling source does not cover the canonical ten-pair universe")
    columns = pd.MultiIndex.from_product([range(N_QUBITS), range(N_QUBITS)])
    wide = frame.pivot(index="update_time", columns=["affected_index", "source_index"],
                       values="c_ij_s_minus_2").reindex(columns=columns)
    if wide.empty or wide.isna().any().any():
        raise ContractError("coupling matrices are incomplete")
    matrices = wide.to_numpy(dtype=np.float64).reshape(-1, N_QUBITS, N_QUBITS)
    if not np.isfinite(matrices).all() or not np.allclose(
            np.diagonal(matrices, axis1=1, axis2=2), 0.0, atol=0.0, rtol=0.0):
        raise ContractError("coupling matrices violate finite/zero-diagonal contract")
    return epoch_ns(wide.index), matrices


def choose_evenly(indices: np.ndarray, cap: int) -> np.ndarray:
    if cap <= 0:
        raise ContractError("sample cap must be positive")
    if len(indices) == 0:
        return np.empty(0, dtype=np.int64)
    if len(indices) <= cap:
        return indices.copy()
    positions = np.linspace(0, len(indices) - 1, num=cap, dtype=np.int64)
    return indices[positions]


def safe_softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(np.clip(shifted, -700.0, 0.0))
    result = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1.0):
        raise ContractError("classifier softmax produced invalid probabilities")
    return result


def apply_hadamard(state: np.ndarray, qubit: int) -> None:
    stride = 1 << qubit
    view = state.reshape(-1, 2 * stride)
    lo = view[:, :stride].copy()
    hi = view[:, stride:].copy()
    scale = 1.0 / math.sqrt(2.0)
    view[:, :stride] = (lo + hi) * scale
    view[:, stride:] = (lo - hi) * scale


def apply_ry(state: np.ndarray, qubit: int, theta: float) -> None:
    stride = 1 << qubit
    view = state.reshape(-1, 2 * stride)
    lo = view[:, :stride].copy()
    hi = view[:, stride:].copy()
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    view[:, :stride] = c * lo - s * hi
    view[:, stride:] = s * lo + c * hi


def apply_rz(state: np.ndarray, qubit: int, theta: float) -> None:
    stride = 1 << qubit
    view = state.reshape(-1, 2 * stride)
    view[:, :stride] *= np.exp(-0.5j * theta)
    view[:, stride:] *= np.exp(0.5j * theta)


def z_eigenvalues(qubit: int) -> np.ndarray:
    index = np.arange(DIMENSION, dtype=np.uint16)
    return np.where(((index >> qubit) & 1) == 0, 1.0, -1.0)


_Z_EIGENVALUES = tuple(z_eigenvalues(q) for q in range(N_QUBITS))
_BASIS_INDEX = np.arange(DIMENSION, dtype=np.uint16)


def apply_rzz(state: np.ndarray, left: int, right: int, theta: float) -> None:
    eigen = _Z_EIGENVALUES[left] * _Z_EIGENVALUES[right]
    state *= np.exp(-0.5j * theta * eigen)


def apply_rzx(state: np.ndarray, left: int, right: int, theta: float) -> None:
    """exp(-i theta Z_left X_right / 2), a noncommuting nearest-neighbor gate."""
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    flipped = state[_BASIS_INDEX ^ np.uint16(1 << right)]
    state[:] = c * state - 1j * s * _Z_EIGENVALUES[left] * flipped


def encode_state(normalized_z: np.ndarray, coupling: np.ndarray) -> np.ndarray:
    """Fixed 10-qubit feature map with noncommuting local and nearest-neighbor gates.

    The fixed circuit is H^10, then RY/RZ on each qubit, then RZZ/RZX on
    edges (0,1)...(8,9).  Only the angles are data dependent.  Causal coupling
    enters only through the contemporaneously available C matrix.
    """
    if normalized_z.shape != (N_QUBITS,) or coupling.shape != (N_QUBITS, N_QUBITS):
        raise ContractError("feature-map inputs have invalid dimensions")
    if not np.isfinite(normalized_z).all() or not np.isfinite(coupling).all():
        raise ContractError("feature-map inputs must be finite")
    z = np.clip(normalized_z, -3.0, 3.0)
    state = np.zeros(DIMENSION, dtype=np.complex128)
    state[0] = 1.0
    for q in range(N_QUBITS):
        apply_hadamard(state, q)
    for q in range(N_QUBITS):
        apply_ry(state, q, 0.75 * math.tanh(float(z[q])))
        apply_rz(state, q, 0.55 * float(z[q]))
    coupling_scaled = np.clip(coupling * DT_S * DT_S, -3.0, 3.0)
    for q in range(N_QUBITS - 1):
        edge_product = math.tanh(float(z[q] * z[q + 1]))
        edge_difference = math.tanh(float(z[q] - z[q + 1]))
        symmetric_c = 0.5 * (coupling_scaled[q, q + 1] + coupling_scaled[q + 1, q])
        antisymmetric_c = 0.5 * (coupling_scaled[q, q + 1] - coupling_scaled[q + 1, q])
        apply_rzz(state, q, q + 1, 0.24 * edge_product + 0.08 * math.tanh(float(symmetric_c)))
        apply_rzx(state, q, q + 1, 0.17 * edge_difference + 0.05 * math.tanh(float(antisymmetric_c)))
    norm = float(np.vdot(state, state).real)
    if not math.isfinite(norm) or abs(norm - 1.0) > 2e-12:
        raise ContractError(f"feature-map state lost normalization: {norm}")
    return state


def fidelity_kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """K_ij=|<psi(x_i)|psi(y_j)>|^2, a PSD quantum fidelity kernel."""
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != DIMENSION or right.shape[1] != DIMENSION:
        raise ContractError("state batches have invalid shape")
    kernel = np.abs(left.conj() @ right.T) ** 2
    if not np.isfinite(kernel).all() or np.any(kernel < -1e-13) or np.any(kernel > 1.0 + 1e-10):
        raise ContractError("fidelity kernel escaped [0, 1]")
    return kernel.astype(np.float64, copy=False)


def nyström_features(kernel_to_landmarks: np.ndarray, landmark_kernel: np.ndarray,
                     eig_floor: float = 1e-10) -> tuple[np.ndarray, dict[str, float]]:
    symmetric = 0.5 * (landmark_kernel + landmark_kernel.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if not np.isfinite(eigenvalues).all():
        raise ContractError("landmark kernel eigendecomposition is non-finite")
    floored = np.maximum(eigenvalues, eig_floor)
    inverse_sqrt = (eigenvectors * (1.0 / np.sqrt(floored))) @ eigenvectors.T
    features = kernel_to_landmarks @ inverse_sqrt
    if not np.isfinite(features).all():
        raise ContractError("Nyström features are non-finite")
    return features, {
        "landmark_eigenvalue_min": float(eigenvalues.min()),
        "landmark_eigenvalue_max": float(eigenvalues.max()),
        "landmark_eigenvalues_floored": int(np.count_nonzero(eigenvalues < eig_floor)),
        "nystrom_eigen_floor": eig_floor,
    }


def fit_kernel_ridge(phi_train: np.ndarray, labels: np.ndarray, ridge: float) -> np.ndarray:
    if ridge <= 0.0 or not math.isfinite(ridge):
        raise ContractError("ridge regularization must be finite and positive")
    if len(labels) != len(phi_train) or np.any(labels < 0) or np.any(labels >= N_QUBITS):
        raise ContractError("training labels are invalid")
    one_hot = np.eye(N_QUBITS, dtype=np.float64)[labels]
    gram = phi_train.T @ phi_train + ridge * np.eye(phi_train.shape[1])
    try:
        weights = np.linalg.solve(gram, phi_train.T @ one_hot)
    except np.linalg.LinAlgError as exc:
        raise ContractError(f"regularized kernel ridge solve failed: {exc}") from exc
    if not np.isfinite(weights).all():
        raise ContractError("kernel-ridge weights are non-finite")
    return weights


def build_candidates(times: np.ndarray, log_close: np.ndarray, coupling_times: np.ndarray,
                     coupling_matrices: np.ndarray, first: int, end: int,
                     split_source_row: int) -> tuple[CausalCandidates, dict[str, np.ndarray]]:
    """Produce causal candidates and a row-level audit surface for every source minute."""
    length = end - first
    reasons = np.full(length, "not_initialized", dtype=object)
    phases = np.where(np.arange(first, end) < split_source_row, "train", "oos").astype(object)
    coupling_time_ns = np.full(length, -1, dtype=np.int64)
    coupling_age_s = np.full(length, np.nan, dtype=np.float64)
    raw_z_audit = np.full((length, N_QUBITS), np.nan, dtype=np.float64)
    target_audit = np.full(length, -1, dtype=np.int16)
    last_audit = np.full(length, -1, dtype=np.int16)

    source_rows: list[int] = []
    timestamps: list[int] = []
    target_timestamps: list[int] = []
    raw_z_rows: list[np.ndarray] = []
    coupling_rows: list[int] = []
    targets: list[int] = []
    last_magnitude: list[int] = []
    sigma2 = np.full(N_QUBITS, 1e-8, dtype=np.float64)
    cursor = int(np.searchsorted(coupling_times, times[first], side="right") - 1)
    if cursor < 0:
        raise ContractError("no causal coupling matrix exists at experiment start")

    for i in range(first, end):
        output_i = i - first
        now = int(times[i])
        previous_contiguous = i > 0 and contiguous_60s(times[i - 1], times[i], DT_NS)
        next_contiguous = contiguous_60s(times[i], times[i + 1], DT_NS)
        while cursor + 1 < len(coupling_times) and coupling_times[cursor + 1] <= now:
            cursor += 1
        age_s = (now - int(coupling_times[cursor])) / 1e9
        coupling_time_ns[output_i] = int(coupling_times[cursor])
        coupling_age_s[output_i] = age_s
        if not previous_contiguous:
            sigma2.fill(1e-8)
            reasons[output_i] = "post_gap_reset_skip"
            continue
        if not next_contiguous:
            reasons[output_i] = "pre_gap_skip"
            continue
        if age_s < 0.0 or age_s > MAX_COUPLING_AGE_S:
            reasons[output_i] = "stale_coupling_skip"
            continue
        ret = log_close[i] - log_close[i - 1]
        sigma2 = (1.0 - EWMA_ALPHA) * sigma2 + EWMA_ALPHA * ret * ret
        z = ret / np.sqrt(np.maximum(sigma2, 1e-16))
        target = int(np.argmax(np.abs(log_close[i + 1] - log_close[i])))
        last = int(np.argmax(np.abs(ret)))
        raw_z_audit[output_i] = z
        target_audit[output_i] = target
        last_audit[output_i] = last
        reasons[output_i] = "eligible_unselected"
        source_rows.append(i)
        timestamps.append(now)
        target_timestamps.append(int(times[i + 1]))
        raw_z_rows.append(z)
        coupling_rows.append(cursor)
        targets.append(target)
        last_magnitude.append(last)

    candidates = CausalCandidates(
        source_rows=np.asarray(source_rows, dtype=np.int64),
        timestamps_ns=np.asarray(timestamps, dtype=np.int64),
        target_timestamps_ns=np.asarray(target_timestamps, dtype=np.int64),
        raw_z=np.asarray(raw_z_rows, dtype=np.float64),
        coupling_rows=np.asarray(coupling_rows, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.int16),
        last_magnitude=np.asarray(last_magnitude, dtype=np.int16),
    )
    if len(candidates.source_rows) < 32:
        raise ContractError("too few contiguous causal candidates")
    audit = {
        "reasons": reasons,
        "phases": phases,
        "coupling_time_ns": coupling_time_ns,
        "coupling_age_s": coupling_age_s,
        "raw_z": raw_z_audit,
        "target": target_audit,
        "last_magnitude": last_audit,
    }
    return candidates, audit


def make_minute_frame(times: np.ndarray, first: int, end: int, audit: dict[str, np.ndarray],
                      selected_train_rows: np.ndarray, selected_oos_rows: np.ndarray,
                      selected_prediction: dict[int, np.ndarray],
                      frozen_prior: np.ndarray, frozen_constant: int) -> pd.DataFrame:
    """Every source minute gets a reason; selected OOS rows additionally get scores."""
    length = end - first
    selected_train_set = set(int(v) for v in selected_train_rows)
    selected_oos_set = set(int(v) for v in selected_oos_rows)
    reasons = audit["reasons"].copy()
    for source_row in selected_train_set:
        reasons[source_row - first] = "selected_train_feature_map"
    for source_row in selected_oos_set:
        reasons[source_row - first] = "selected_oos_scored"
    frame: dict[str, object] = {
        "timestamp": pd.to_datetime(times[first:end], unit="ns", utc=True),
        "target_timestamp": pd.to_datetime(times[first + 1:end + 1], unit="ns", utc=True),
        "phase": audit["phases"],
        "reason": reasons,
        "coupling_update_time": pd.to_datetime(
            np.where(audit["coupling_time_ns"] >= 0, audit["coupling_time_ns"], 0), unit="ns", utc=True
        ),
        "coupling_age_s": audit["coupling_age_s"],
        "target_index": audit["target"],
        "last_magnitude_baseline_index": audit["last_magnitude"],
        "model_top1_index": np.full(length, -1, dtype=np.int16),
        "model_brier": np.full(length, np.nan),
        "prior_brier": np.full(length, np.nan),
        "uniform_brier": np.full(length, np.nan),
        "constant_brier": np.full(length, np.nan),
        "last_magnitude_brier": np.full(length, np.nan),
    }
    for pair_i, pair in enumerate(PAIRS):
        frame[f"z_{pair.lower()}"] = audit["raw_z"][:, pair_i]
        frame[f"p_model_{pair.lower()}"] = np.full(length, np.nan)
        frame[f"p_prior_{pair.lower()}"] = np.full(length, np.nan)
    for source_row, probabilities in selected_prediction.items():
        j = source_row - first
        target = int(audit["target"][j])
        one_hot = np.eye(N_QUBITS)[target]
        frame["model_top1_index"][j] = int(np.argmax(probabilities))
        frame["model_brier"][j] = float(np.sum((probabilities - one_hot) ** 2))
        frame["prior_brier"][j] = float(np.sum((frozen_prior - one_hot) ** 2))
        frame["uniform_brier"][j] = float(np.sum((1.0 / N_QUBITS - one_hot) ** 2))
        constant = np.zeros(N_QUBITS)
        constant[frozen_constant] = 1.0
        frame["constant_brier"][j] = float(np.sum((constant - one_hot) ** 2))
        recent = np.zeros(N_QUBITS)
        recent[int(audit["last_magnitude"][j])] = 1.0
        frame["last_magnitude_brier"][j] = float(np.sum((recent - one_hot) ** 2))
        for pair_i, pair in enumerate(PAIRS):
            frame[f"p_model_{pair.lower()}"][j] = float(probabilities[pair_i])
            frame[f"p_prior_{pair.lower()}"][j] = float(frozen_prior[pair_i])
    # It is clearer and type-stable to represent unavailable causal C timestamps as NaT.
    frame["coupling_update_time"] = pd.to_datetime(audit["coupling_time_ns"], unit="ns", utc=True,
                                                     errors="coerce")
    return pd.DataFrame(frame)


def self_check() -> dict[str, object]:
    z = np.linspace(-1.0, 1.0, N_QUBITS)
    c = np.zeros((N_QUBITS, N_QUBITS))
    c[0, 1], c[1, 0] = 2e-5, -1e-5
    first = encode_state(z, c)
    second = encode_state(z, c)
    altered = encode_state(z + 0.25, c)
    states = np.vstack([first, second, altered])
    k = fidelity_kernel(states, states)
    phi, diagnostics = nyström_features(k, k)
    w = fit_kernel_ridge(phi, np.array([0, 0, 1], dtype=np.int16), ridge=1e-2)
    probabilities = safe_softmax(phi @ w)
    passed = bool(
        abs(float(np.vdot(first, first).real) - 1.0) < 2e-12
        and np.array_equal(first, second)
        and np.allclose(np.diag(k), 1.0, atol=2e-12)
        and np.linalg.eigvalsh(0.5 * (k + k.T)).min() >= -1e-10
        and np.isfinite(probabilities).all()
        and np.allclose(probabilities.sum(axis=1), 1.0)
        and diagnostics["landmark_eigenvalue_min"] >= -1e-10
    )
    return {
        "passed": passed,
        "dimension": DIMENSION,
        "state_norm_error": abs(float(np.vdot(first, first).real) - 1.0),
        "same_input_state_bitwise_equal": bool(np.array_equal(first, second)),
        "kernel_diagonal_max_error": float(np.max(np.abs(np.diag(k) - 1.0))),
        "kernel_min_eigenvalue": float(np.linalg.eigvalsh(0.5 * (k + k.T)).min()),
        "classifier_probability_sum_max_error": float(np.max(np.abs(probabilities.sum(axis=1) - 1.0))),
        "nystrom": diagnostics,
    }


def run(max_steps: int, max_train_samples: int, max_oos_samples: int, landmarks: int,
        seed: int, train_fraction: float, ridge: float, out_dir: Path) -> dict[str, object]:
    if max_steps < 100 or max_train_samples < 16 or max_oos_samples < 8:
        raise ContractError("run caps are too small for a held-out kernel experiment")
    if not (0.5 <= train_fraction < 1.0) or landmarks < 4 or landmarks > max_train_samples:
        raise ContractError("split or landmark configuration is invalid")
    coupling_times, coupling_matrices = load_coupling()
    times, log_close = load_window(max_steps, utc_timestamp(int(coupling_times[0])))
    first = int(np.searchsorted(times, coupling_times[0], side="left"))
    while first < len(times) and (first == 0 or times[first] - times[first - 1] != DT_NS):
        first += 1
    end = first + max_steps
    if end >= len(times):
        raise ContractError("bounded canonical window did not retain requested source steps")
    split_source_row = first + int((end - first) * train_fraction)
    candidates, audit = build_candidates(times, log_close, coupling_times, coupling_matrices,
                                         first, end, split_source_row)

    train_candidates = np.flatnonzero(candidates.source_rows < split_source_row)
    oos_candidates = np.flatnonzero(candidates.source_rows >= split_source_row)
    selected_train_idx = choose_evenly(train_candidates, max_train_samples)
    selected_oos_idx = choose_evenly(oos_candidates, max_oos_samples)
    if len(selected_train_idx) < landmarks or len(selected_oos_idx) < 8:
        raise ContractError("not enough causal samples in one side of fixed chronological split")

    # Fit the normalizer from selected *training-period causal features only*.
    train_raw = candidates.raw_z[selected_train_idx]
    normalizer_mean = train_raw.mean(axis=0)
    normalizer_scale = train_raw.std(axis=0, ddof=0)
    normalizer_scale = np.maximum(normalizer_scale, 1e-6)
    all_selected_idx = np.concatenate([selected_train_idx, selected_oos_idx])
    normalized = np.clip((candidates.raw_z[all_selected_idx] - normalizer_mean) / normalizer_scale,
                         -3.0, 3.0)
    selected_coupling = coupling_matrices[candidates.coupling_rows[all_selected_idx]]
    states = np.vstack([encode_state(normalized[i], selected_coupling[i])
                        for i in range(len(all_selected_idx))])
    state_norm_errors = np.abs(np.sum(np.abs(states) ** 2, axis=1) - 1.0)
    if float(state_norm_errors.max()) >= 2e-12:
        raise ContractError("encoded state normalization check failed")
    train_count = len(selected_train_idx)
    train_states, oos_states = states[:train_count], states[train_count:]

    # Landmarks are a deterministic seeded subset of training states only.
    rng = np.random.default_rng(seed)
    landmark_train_positions = np.sort(rng.choice(train_count, size=landmarks, replace=False))
    landmark_states = train_states[landmark_train_positions]
    k_train_landmark = fidelity_kernel(train_states, landmark_states)
    k_oos_landmark = fidelity_kernel(oos_states, landmark_states)
    k_landmark = fidelity_kernel(landmark_states, landmark_states)
    phi_train, nystrom_diagnostics = nyström_features(k_train_landmark, k_landmark)
    phi_oos, _ = nyström_features(k_oos_landmark, k_landmark)
    train_labels = candidates.targets[selected_train_idx].astype(np.int64)
    oos_labels = candidates.targets[selected_oos_idx].astype(np.int64)
    weights = fit_kernel_ridge(phi_train, train_labels, ridge)
    oos_probabilities = safe_softmax(phi_oos @ weights)

    train_counts = np.bincount(train_labels, minlength=N_QUBITS)
    frozen_prior = train_counts / train_counts.sum()
    frozen_constant = int(np.argmax(frozen_prior))
    oos_one_hot = np.eye(N_QUBITS)[oos_labels]
    uniform = np.full(N_QUBITS, 1.0 / N_QUBITS)
    constant = np.eye(N_QUBITS)[frozen_constant]
    last_predictions = np.eye(N_QUBITS)[candidates.last_magnitude[selected_oos_idx]]
    metrics = {
        "top1_accuracy": float(np.mean(np.argmax(oos_probabilities, axis=1) == oos_labels)),
        "brier_score": float(np.mean(np.sum((oos_probabilities - oos_one_hot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(np.maximum(oos_probabilities[np.arange(len(oos_labels)), oos_labels], 1e-15)))),
    }
    baselines = {
        "uniform": {
            # A tied uniform vector has no unique deterministic top-1 class;
            # report randomized-tie expected accuracy rather than np.argmax's
            # arbitrary class-zero frequency.
            "top1_accuracy": 1.0 / N_QUBITS,
            "brier_score": float(np.mean(np.sum((uniform - oos_one_hot) ** 2, axis=1))),
        },
        "frozen_train_prior": {
            "probabilities": frozen_prior.tolist(),
            "top1_accuracy": float(np.mean(np.argmax(frozen_prior) == oos_labels)),
            "brier_score": float(np.mean(np.sum((frozen_prior - oos_one_hot) ** 2, axis=1))),
        },
        "frozen_train_constant": {
            "target_index": frozen_constant,
            "target_pair": PAIRS[frozen_constant],
            "top1_accuracy": float(np.mean(frozen_constant == oos_labels)),
            "brier_score": float(np.mean(np.sum((constant - oos_one_hot) ** 2, axis=1))),
        },
        "causal_last_magnitude": {
            "top1_accuracy": float(np.mean(candidates.last_magnitude[selected_oos_idx] == oos_labels)),
            "brier_score": float(np.mean(np.sum((last_predictions - oos_one_hot) ** 2, axis=1))),
        },
    }
    # Promotion requires beating both a proper-score baseline and the simple causal feature baseline.
    promoted = bool(metrics["brier_score"] < baselines["frozen_train_prior"]["brier_score"]
                    and metrics["top1_accuracy"] > baselines["causal_last_magnitude"]["top1_accuracy"])

    selected_prediction = {
        int(candidates.source_rows[idx]): oos_probabilities[position]
        for position, idx in enumerate(selected_oos_idx)
    }
    minute = make_minute_frame(times, first, end, audit,
                               candidates.source_rows[selected_train_idx],
                               candidates.source_rows[selected_oos_idx],
                               selected_prediction, frozen_prior, frozen_constant)
    minute_reason_counts = {str(reason): int(count) for reason, count in
                            minute["reason"].value_counts().sort_index().items()}
    daily = minute.loc[minute["reason"] == "selected_oos_scored"].copy()
    daily["date"] = daily["timestamp"].dt.floor("D")
    daily = daily.groupby("date", as_index=False).agg(
        oos_scored_minutes=("reason", "size"),
        top1_accuracy=("model_top1_index", lambda x: float(np.mean(x.to_numpy() == minute.loc[x.index, "target_index"].to_numpy()))),
        model_brier=("model_brier", "mean"),
        prior_brier=("prior_brier", "mean"),
        uniform_brier=("uniform_brier", "mean"),
        constant_brier=("constant_brier", "mean"),
        last_magnitude_brier=("last_magnitude_brier", "mean"),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    minute_path = out_dir / "quantum_kernel_minute.parquet"
    daily_path = out_dir / "quantum_kernel_daily.parquet"
    summary_path = out_dir / "quantum_kernel_summary.json"
    validation_path = out_dir / "quantum_kernel_validation.json"
    minute.to_parquet(minute_path, index=False, compression="zstd")
    daily.to_parquet(daily_path, index=False, compression="zstd")
    validation = self_check()
    validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    if not validation["passed"]:
        raise ContractError("quantum-kernel self-check failed")

    config = {
        "version": VERSION, "max_steps": max_steps, "max_train_samples": max_train_samples,
        "max_oos_samples": max_oos_samples, "landmarks": landmarks, "seed": seed,
        "train_fraction": train_fraction, "ridge": ridge,
        "circuit": "H^10; per-qubit RY/RZ; nearest-neighbor RZZ/RZX on edges 0-1..8-9",
    }
    summary = {
        "version": VERSION,
        "interpretation": "classical exact statevector simulation of a quantum fidelity feature map; not evidence of a physical quantum FX market and not a trading model",
        "pair_scope_and_qubit_order": list(PAIRS),
        "statevector_dimension": DIMENSION,
        "fixed_circuit": config["circuit"],
        "causality": {
            "features": "return t-1 to t, causal EWMA volatility, and latest coupling matrix whose update time is <= t",
            "target": "pair with largest absolute next valid 60-second log return",
            "split": "predeclared final 30% chronological source-row holdout",
            "normalizer": "mean/std fit on selected pre-holdout causal features only",
            "landmarks": "deterministic seeded subset of selected pre-holdout states only",
            "gap_policy": "no return/target/model emission across a non-60-second interval; EWMA resets on first post-gap bar",
        },
        "bounded_execution": {
            "source_rows_processed": max_steps,
            "source_span": [utc_timestamp(int(times[first])).isoformat(),
                            utc_timestamp(int(times[end])).isoformat()],
            "eligible_causal_rows": int(len(candidates.source_rows)),
            "selected_train_samples": int(len(selected_train_idx)),
            "selected_oos_samples": int(len(selected_oos_idx)),
            "landmarks": landmarks,
            "selection": "evenly spaced, label-agnostic selection within each chronological partition",
            "minute_reason_counts": minute_reason_counts,
        },
        "kernel_ridge": {
            "kernel": "K(x,y)=|<psi(x)|psi(y)>|^2",
            "nystrom": nystrom_diagnostics,
            "ridge": ridge,
            "state_norm_max_error": float(state_norm_errors.max()),
        },
        "oos_experimental_metric": {
            "samples": int(len(oos_labels)),
            "model": metrics,
            "frozen_train_baselines": baselines,
            "promotion_gate": "beat frozen train-prior Brier and causal last-magnitude top-1; only then eligible for a separate validation discussion",
            "promoted": promoted,
            "claim": "classification diagnostic only; no trading, causation, market-ontology, or hardware-performance claim",
        },
        "determinism": {
            "seed": seed,
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest(),
            "selected_candidate_indices_sha256": sha256_array(all_selected_idx.astype("<i8")),
            "selected_source_rows_sha256": sha256_array(candidates.source_rows[all_selected_idx].astype("<i8")),
            "encoded_states_sha256": sha256_array(states.view(np.float64)),
        },
        "input_provenance": manifest_provenance(),
        "outputs": {
            "minute": str(minute_path.relative_to(ROOT)).replace("\\", "/"),
            "daily": str(daily_path.relative_to(ROOT)).replace("\\", "/"),
            "validation": str(validation_path.relative_to(ROOT)).replace("\\", "/"),
            "minute_sha256": sha256_file(minute_path),
            "daily_sha256": sha256_file(daily_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-train-samples", type=int, default=DEFAULT_MAX_TRAIN_SAMPLES)
    parser.add_argument("--max-oos-samples", type=int, default=DEFAULT_MAX_OOS_SAMPLES)
    parser.add_argument("--landmarks", type=int, default=DEFAULT_LANDMARKS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    parser.add_argument("--out-dir", type=Path, default=DERIVED_DIR)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.self_check:
            result = self_check()
            print(json.dumps(result, indent=2))
            return 0 if result["passed"] else 1
        run(args.max_steps, args.max_train_samples, args.max_oos_samples, args.landmarks,
            args.seed, args.train_fraction, args.ridge, args.out_dir.resolve())
        return 0
    except (ContractError, SharedContractError, ValueError, OSError, np.linalg.LinAlgError) as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

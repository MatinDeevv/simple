"""Causal 10-qutrit MPS/TEBD quantum-trajectory research experiment.

This is a computational, open-quantum-system representation of canonical FX
observations.  It is deliberately isolated from the canonical simulator and
does not claim that markets are physical quantum systems or emit a signal.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fxresearch.core.contracts import canonical_pair_order, contiguous_60s


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DIR = ROOT / "data" / "canonical"
DERIVED_DIR = ROOT / "data" / "derived"

PAIRS = canonical_pair_order(ROOT)
N_SITES = len(PAIRS)
LOCAL_DIM = 3
DT_S = 60.0
DT_NS = int(DT_S * 1_000_000_000)
HALFLIFE_STEPS = 1_440.0
EWMA_ALPHA = 1.0 - math.exp(-math.log(2.0) / HALFLIFE_STEPS)
MAX_STANDARDIZED_RETURN = 5.0

# These are dimensionless algorithm parameters, not calibrated physical constants.
LOCAL_PHASE_GAIN = 0.085
LOCAL_DRIVE_GAIN = 0.115
COUPLING_GAIN = 0.85
MAX_EDGE_ANGLE = 0.35
DEPHASING_PER_SITE_STEP = 0.004
NO_JUMP_PROBABILITY = 1.0 - DEPHASING_PER_SITE_STEP

DEFAULT_MAX_STEPS = 5_000
DEFAULT_CHI = 16
DEFAULT_SVD_CUTOFF = 1.0e-10
DEFAULT_SEED = 20260718
DEFAULT_OOS_FRACTION = 0.70
VERSION = "quantum-mps-1.0.0"


class ContractError(RuntimeError):
    """Raised when an experimental causal/numerical contract is violated."""


@dataclass
class StepDiagnostics:
    jump_events: int = 0
    truncation_events: int = 0
    discarded_weight: float = 0.0
    max_pre_normalization_error: float = 0.0
    max_gate_unitarity_error: float = 0.0


@dataclass
class DailyAccumulator:
    day: int
    updates: int = 0
    activity_sum: np.ndarray | None = None
    entropy_sum: float = 0.0
    max_bond_dimension: int = 1
    jump_events: int = 0
    truncation_events: int = 0
    discarded_weight: float = 0.0
    max_norm_error: float = 0.0
    max_schmidt_entropy: float = 0.0

    def add(self, activity: np.ndarray, *, mean_entropy: float,
            max_entropy: float, max_bond_dim: int, step: StepDiagnostics,
            norm_error: float) -> None:
        if self.activity_sum is None:
            self.activity_sum = np.zeros(N_SITES, dtype=np.float64)
        self.updates += 1
        self.activity_sum += activity
        self.entropy_sum += mean_entropy
        self.max_schmidt_entropy = max(self.max_schmidt_entropy, max_entropy)
        self.max_bond_dimension = max(self.max_bond_dimension, max_bond_dim)
        self.jump_events += step.jump_events
        self.truncation_events += step.truncation_events
        self.discarded_weight += step.discarded_weight
        self.max_norm_error = max(self.max_norm_error, norm_error)

    def as_row(self) -> dict[str, object]:
        if self.updates == 0 or self.activity_sum is None:
            raise ContractError("cannot materialize empty daily accumulator")
        result: dict[str, object] = {
            "timestamp": utc_timestamp(self.day * 86_400_000_000_000),
            "updates": self.updates,
            "mean_schmidt_entropy": self.entropy_sum / self.updates,
            "max_schmidt_entropy": self.max_schmidt_entropy,
            "max_bond_dimension": self.max_bond_dimension,
            "jump_events": self.jump_events,
            "truncation_events": self.truncation_events,
            "discarded_weight": self.discarded_weight,
            "max_norm_error": self.max_norm_error,
        }
        for index, pair in enumerate(PAIRS):
            result[f"activity_{pair.lower()}"] = float(
                self.activity_sum[index] / self.updates
            )
        return result


def epoch_ns(values: object) -> np.ndarray:
    return pd.DatetimeIndex(values).as_unit("ns").asi8


def utc_timestamp(ns: int) -> pd.Timestamp:
    return pd.Timestamp(ns, unit="ns", tz="UTC")


def load_prices() -> tuple[np.ndarray, np.ndarray]:
    """Strictly inner-join all canonical close series; never resample/fill."""
    joined: pd.DataFrame | None = None
    for pair in PAIRS:
        path = CANONICAL_DIR / f"{pair}.parquet"
        frame = pd.read_parquet(path, columns=["timestamp", "close"])
        frame = frame.rename(columns={"close": pair})
        joined = frame if joined is None else joined.merge(
            frame, on="timestamp", how="inner", validate="one_to_one"
        )
    assert joined is not None
    joined = joined.sort_values("timestamp", kind="stable").reset_index(drop=True)
    times = epoch_ns(joined["timestamp"])
    closes = joined.loc[:, list(PAIRS)].to_numpy(dtype=np.float64)
    if (len(times) < 4 or not np.all(np.diff(times) > 0)
            or not np.isfinite(closes).all() or np.any(closes <= 0.0)):
        raise ContractError("10-pair canonical input violates timestamp/price contract")
    return times, np.log(closes)


def load_coupling() -> dict[str, np.ndarray]:
    """Load complete causal 10x10 snapshots in the frozen canonical order."""
    path = DERIVED_DIR / "coupling_estimates.parquet"
    frame = pd.read_parquet(path)
    required = {"update_time", "affected_index", "source_index", "c_ij_s_minus_2"}
    if not required.issubset(frame.columns):
        raise ContractError("coupling estimate schema is incomplete")
    expected = pd.MultiIndex.from_product([range(N_SITES), range(N_SITES)])
    wide = frame.pivot(
        index="update_time", columns=["affected_index", "source_index"],
        values="c_ij_s_minus_2",
    ).reindex(columns=expected)
    if wide.empty or wide.isna().any().any():
        raise ContractError("coupling snapshots are incomplete")
    matrices = wide.to_numpy(dtype=np.float64).reshape(-1, N_SITES, N_SITES)
    diagonal = np.diagonal(matrices, axis1=1, axis2=2)
    if not np.isfinite(matrices).all() or not np.array_equal(diagonal, np.zeros_like(diagonal)):
        raise ContractError("coupling finite/zero-diagonal contract failed")
    return {"times": epoch_ns(wide.index), "matrices": matrices}


def spin_one_operators() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Sx, Sy, Sz in the fixed |down>, |neutral>, |up> basis."""
    root2 = math.sqrt(2.0)
    sx = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]) / root2
    sy = np.array([[0.0, -1j, 0.0], [1j, 0.0, -1j], [0.0, 1j, 0.0]]) / root2
    sz = np.diag([-1.0, 0.0, 1.0])
    return sx.astype(np.complex128), sy.astype(np.complex128), sz.astype(np.complex128)


SX, SY, SZ = spin_one_operators()
LOCAL_GENERATOR = LOCAL_PHASE_GAIN * SZ + LOCAL_DRIVE_GAIN * SX
LOCAL_EIGENVALUES, LOCAL_EIGENVECTORS = np.linalg.eigh(LOCAL_GENERATOR)
BOND_GENERATOR = (
    np.kron(SX, SX) + np.kron(SY, SY) + 0.35 * np.kron(SZ, SZ)
)
BOND_EIGENVALUES, BOND_EIGENVECTORS = np.linalg.eigh(BOND_GENERATOR)


def unitary_from_spectrum(values: np.ndarray, vectors: np.ndarray, angle: float) -> np.ndarray:
    return (vectors * np.exp(-1j * angle * values)) @ vectors.conj().T


def local_gate(z_value: float, factor: float) -> np.ndarray:
    angle = factor * float(np.clip(z_value, -MAX_STANDARDIZED_RETURN, MAX_STANDARDIZED_RETURN))
    return unitary_from_spectrum(LOCAL_EIGENVALUES, LOCAL_EIGENVECTORS, angle)


def edge_angle(coupling: np.ndarray, left: int) -> float:
    """Map an adjacent causal C pair into a bounded Hermitian exchange angle.

    C is directed.  Hamiltonian exchange needs a real symmetric coefficient, so
    only (C_ij+C_ji)/2 is represented; the directed remainder is explicitly
    outside this experiment rather than silently treated as Hermitian.
    """
    raw = 0.5 * (float(coupling[left, left + 1]) + float(coupling[left + 1, left]))
    return float(np.clip(COUPLING_GAIN * raw * DT_S * DT_S,
                         -MAX_EDGE_ANGLE, MAX_EDGE_ANGLE))


def bond_gate(angle: float, factor: float) -> np.ndarray:
    return unitary_from_spectrum(BOND_EIGENVALUES, BOND_EIGENVECTORS, factor * angle)


def mps_product_uniform() -> list[np.ndarray]:
    """Return an exactly normalized unentangled 10-qutrit MPS."""
    local = np.full(LOCAL_DIM, 1.0 / math.sqrt(LOCAL_DIM), dtype=np.complex128)
    return [local.reshape(1, LOCAL_DIM, 1).copy() for _ in range(N_SITES)]


def mps_norm_sq(mps: list[np.ndarray]) -> float:
    environment = np.ones((1, 1), dtype=np.complex128)
    for tensor in mps:
        environment = np.einsum("ab,asc,bsd->cd", environment, tensor, tensor.conj(),
                                optimize=True)
    value = complex(environment[0, 0])
    if abs(value.imag) > 1.0e-9 or not math.isfinite(value.real) or value.real <= 0.0:
        raise ContractError(f"non-positive/non-real MPS norm squared: {value}")
    return float(value.real)


def normalize_mps(mps: list[np.ndarray]) -> float:
    """Normalize in place and return the absolute pre-normalization norm error."""
    norm_sq = mps_norm_sq(mps)
    norm = math.sqrt(norm_sq)
    mps[0] = mps[0] / norm
    return abs(norm - 1.0)


def apply_one_site(mps: list[np.ndarray], site: int, gate: np.ndarray) -> None:
    tensor = mps[site]
    mps[site] = np.einsum("uv,lvr->lur", gate, tensor, optimize=True)


def apply_two_site(mps: list[np.ndarray], left: int, gate: np.ndarray,
                   chi: int, cutoff: float) -> tuple[int, float, float]:
    """Apply a 9x9 gate then SVD-split/truncate an adjacent MPS bond.

    Returns retained rank, relative discarded norm-squared, and absolute
    pre-normalization norm error caused by the truncation.
    """
    a = mps[left]
    b = mps[left + 1]
    if a.shape[2] != b.shape[0]:
        raise ContractError("adjacent MPS bond dimensions are incompatible")
    theta = np.einsum("aib,bjc->aijc", a, b, optimize=True)
    gate4 = gate.reshape(LOCAL_DIM, LOCAL_DIM, LOCAL_DIM, LOCAL_DIM)
    theta = np.einsum("IJij,aijb->aIJb", gate4, theta, optimize=True)
    ldim, _, _, rdim = theta.shape
    matrix = theta.reshape(ldim * LOCAL_DIM, LOCAL_DIM * rdim)
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    total_weight = float(np.dot(singular, singular))
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ContractError("two-site SVD has invalid singular spectrum")
    above_cutoff = int(np.count_nonzero(singular > singular[0] * cutoff))
    kept = max(1, min(int(chi), above_cutoff, len(singular)))
    discarded = float(np.dot(singular[kept:], singular[kept:]) / total_weight)
    kept_weight = max(0.0, 1.0 - discarded)
    mps[left] = u[:, :kept].reshape(ldim, LOCAL_DIM, kept)
    mps[left + 1] = (singular[:kept, None] * vh[:kept, :]).reshape(
        kept, LOCAL_DIM, rdim
    )
    return kept, discarded, abs(math.sqrt(kept_weight) - 1.0)


def local_probabilities(mps: list[np.ndarray]) -> np.ndarray:
    """Compute local qutrit Born probabilities without forming 3^10 amplitudes."""
    right: list[np.ndarray] = [np.empty((0, 0), dtype=np.complex128) for _ in range(N_SITES + 1)]
    right[N_SITES] = np.ones((1, 1), dtype=np.complex128)
    for site in range(N_SITES - 1, -1, -1):
        tensor = mps[site]
        right[site] = np.einsum("asc,bsd,cd->ab", tensor, tensor.conj(), right[site + 1],
                                optimize=True)
    left_env = np.ones((1, 1), dtype=np.complex128)
    output = np.empty((N_SITES, LOCAL_DIM), dtype=np.float64)
    for site, tensor in enumerate(mps):
        reduced = np.einsum("ab,asc,btd,cd->st", left_env, tensor, tensor.conj(),
                            right[site + 1], optimize=True)
        diagonal = np.real(np.diag(0.5 * (reduced + reduced.conj().T)))
        if not np.isfinite(diagonal).all() or np.any(diagonal < -1.0e-9):
            raise ContractError("local reduced state has invalid probabilities")
        diagonal = np.clip(diagonal, 0.0, None)
        total = float(diagonal.sum())
        if total <= 0.0:
            raise ContractError("local reduced state has zero trace")
        output[site] = diagonal / total
        left_env = np.einsum("ab,asc,bsd->cd", left_env, tensor, tensor.conj(),
                             optimize=True)
    return output


def schmidt_metrics(mps: list[np.ndarray]) -> tuple[float, float, int]:
    """Compute exact per-cut Schmidt spectra from compact left/right blocks.

    The MPS need not be in a particular canonical gauge for this calculation.
    With ten qutrits the largest side block is 243 by chi, never 3^10 squared.
    """
    max_entropy = 0.0
    entropy_sum = 0.0
    max_rank = 1
    for cut in range(1, N_SITES):
        left = mps[0][0, :, :]
        for site in range(1, cut):
            left = np.tensordot(left, mps[site], axes=(-1, 0))
            left = left.reshape(-1, left.shape[-1])
        right = mps[cut]
        for site in range(cut + 1, N_SITES):
            right = np.tensordot(right, mps[site], axes=(-1, 0))
        right = right.reshape(right.shape[0], -1)
        q_left, r_left = np.linalg.qr(left, mode="reduced")
        q_right, r_right = np.linalg.qr(right.T, mode="reduced")
        del q_left, q_right
        singular = np.linalg.svd(r_left @ r_right.T, compute_uv=False)
        weights = np.square(singular)
        total = float(weights.sum())
        if not math.isfinite(total) or total <= 0.0:
            raise ContractError("invalid Schmidt spectrum")
        weights /= total
        positive = weights[weights > 1.0e-15]
        entropy = -float(np.dot(positive, np.log(positive)))
        entropy_sum += entropy
        max_entropy = max(max_entropy, entropy)
        max_rank = max(max_rank, int(np.count_nonzero(weights > 1.0e-12)))
    return entropy_sum / (N_SITES - 1), max_entropy, max_rank


def mps_dense(mps: list[np.ndarray]) -> np.ndarray:
    """Small-system self-check materialization; never used by the replay."""
    state = mps[0][0, :, :]
    for tensor in mps[1:]:
        state = np.tensordot(state, tensor, axes=(-1, 0))
    return state.reshape(-1)


def quantum_jump_layer(mps: list[np.ndarray], rng: np.random.Generator) -> int:
    """Sample local pure-dephasing Lindblad jumps and renormalize conditionally."""
    jumps = 0
    for site in range(N_SITES):
        if rng.random() < NO_JUMP_PROBABILITY:
            continue
        probabilities = local_probabilities(mps)[site]
        choice = int(rng.choice(LOCAL_DIM, p=probabilities))
        projector = np.zeros((LOCAL_DIM, LOCAL_DIM), dtype=np.complex128)
        projector[choice, choice] = 1.0
        apply_one_site(mps, site, projector)
        normalize_mps(mps)
        jumps += 1
    return jumps


def tebd_step(mps: list[np.ndarray], standardized_return: np.ndarray,
              coupling: np.ndarray, chi: int, cutoff: float,
              rng: np.random.Generator) -> StepDiagnostics:
    """Second-order local + nearest-neighbour TEBD, then a dephasing trajectory layer."""
    result = StepDiagnostics()
    if standardized_return.shape != (N_SITES,) or coupling.shape != (N_SITES, N_SITES):
        raise ContractError("step input shape differs from fixed 10-qutrit contract")
    gates = [bond_gate(edge_angle(coupling, left), 1.0) for left in range(N_SITES - 1)]
    for factor in (0.5,):
        for site, value in enumerate(standardized_return):
            gate = local_gate(float(value), factor)
            result.max_gate_unitarity_error = max(
                result.max_gate_unitarity_error,
                float(np.max(np.abs(gate.conj().T @ gate - np.eye(LOCAL_DIM)))),
            )
            apply_one_site(mps, site, gate)
    # Strang ordering: even half, odd full, even half.
    for parity, factor in ((0, 0.5), (1, 1.0), (0, 0.5)):
        for left in range(parity, N_SITES - 1, 2):
            gate = gates[left] if factor == 1.0 else bond_gate(edge_angle(coupling, left), factor)
            result.max_gate_unitarity_error = max(
                result.max_gate_unitarity_error,
                float(np.max(np.abs(gate.conj().T @ gate - np.eye(LOCAL_DIM ** 2)))),
            )
            _, discarded, pre_norm_error = apply_two_site(mps, left, gate, chi, cutoff)
            result.discarded_weight += discarded
            if discarded > 0.0:
                result.truncation_events += 1
                result.max_pre_normalization_error = max(
                    result.max_pre_normalization_error, pre_norm_error
                )
            if discarded > 0.0:
                normalize_mps(mps)
    for factor in (0.5,):
        for site, value in enumerate(standardized_return):
            apply_one_site(mps, site, local_gate(float(value), factor))
    result.jump_events = quantum_jump_layer(mps, rng)
    result.max_pre_normalization_error = max(result.max_pre_normalization_error, normalize_mps(mps))
    return result


def activity_distribution(local_born: np.ndarray) -> np.ndarray:
    """Map local qutrits to a normalized next-bar *diagnostic* distribution.

    The neutral basis state is not an activity state, so each site's score is
    p(down)+p(up).  This is a model output only, not an executable forecast.
    """
    activity = np.clip(local_born[:, 0] + local_born[:, 2], 0.0, None)
    total = float(activity.sum())
    if not math.isfinite(total) or total <= 1.0e-15:
        return np.full(N_SITES, 1.0 / N_SITES, dtype=np.float64)
    return activity / total


def minute_record(*, timestamp_ns: int, row_index: int, previous_ns: int,
                  next_ns: int, previous_contiguous: bool, next_contiguous: bool,
                  reason: str, phase: str, reset: bool,
                  coupling_time_ns: int | None, coupling_matrix: np.ndarray | None,
                  log_return: np.ndarray | None = None,
                  standardized_return: np.ndarray | None = None,
                  activity: np.ndarray | None = None,
                  local_born: np.ndarray | None = None,
                  next_log_return: np.ndarray | None = None,
                  target_index: int = -1, mean_schmidt_entropy: float = float("nan"),
                  max_schmidt_entropy: float = float("nan"), max_bond_dimension: int = 0,
                  jump_events: int = 0, truncation_events: int = 0,
                  discarded_weight: float = 0.0, norm_error: float = float("nan")) -> dict[str, object]:
    nan_vector = np.full(N_SITES, np.nan, dtype=np.float64)
    log_return = nan_vector if log_return is None else log_return
    standardized_return = nan_vector if standardized_return is None else standardized_return
    activity = nan_vector if activity is None else activity
    next_log_return = nan_vector if next_log_return is None else next_log_return
    if local_born is None:
        local_born = np.full((N_SITES, LOCAL_DIM), np.nan, dtype=np.float64)
    record: dict[str, object] = {
        "timestamp": utc_timestamp(timestamp_ns),
        "row_index": row_index,
        "previous_timestamp": utc_timestamp(previous_ns),
        "next_timestamp": utc_timestamp(next_ns),
        "previous_contiguous_60s": previous_contiguous,
        "next_contiguous_60s": next_contiguous,
        "reason": reason,
        "chronological_phase": phase,
        "mps_reset": reset,
        "coupling_update_time": (utc_timestamp(coupling_time_ns)
                                 if coupling_time_ns is not None else pd.NaT),
        "coupling_age_seconds": (float(timestamp_ns - coupling_time_ns) / 1_000_000_000
                                 if coupling_time_ns is not None else float("nan")),
        "next_target_index": target_index,
        "mean_schmidt_entropy": mean_schmidt_entropy,
        "max_schmidt_entropy": max_schmidt_entropy,
        "max_bond_dimension": max_bond_dimension,
        "jump_events": jump_events,
        "truncation_events": truncation_events,
        "discarded_weight": discarded_weight,
        "norm_error": norm_error,
    }
    for index, pair in enumerate(PAIRS):
        label = pair.lower()
        record[f"log_return_{label}"] = float(log_return[index])
        record[f"z_{label}"] = float(standardized_return[index])
        record[f"activity_{label}"] = float(activity[index])
        record[f"born_down_{label}"] = float(local_born[index, 0])
        record[f"born_neutral_{label}"] = float(local_born[index, 1])
        record[f"born_up_{label}"] = float(local_born[index, 2])
        record[f"next_log_return_{label}"] = float(next_log_return[index])
    if coupling_matrix is not None:
        for affected in range(N_SITES):
            for source in range(N_SITES):
                record[f"c_{affected}{source}_s_minus_2"] = float(coupling_matrix[affected, source])
    return record


def self_check() -> dict[str, object]:
    """Verify quantum gates, MPS splitting, truncation, jumps, and seed replay."""
    local = local_gate(0.8, 1.0)
    interaction = bond_gate(0.22, 1.0)
    mps = mps_product_uniform()
    dense_before = mps_dense(mps)
    _, discarded, _ = apply_two_site(mps, 0, interaction, chi=16, cutoff=1.0e-14)
    normalize_mps(mps)
    dense_after = mps_dense(mps)
    born = local_probabilities(mps)
    entropy_mean, entropy_max, schmidt_rank = schmidt_metrics(mps)

    # A low chi path must quantify discarded amplitude rather than hide it.
    truncated = mps_product_uniform()
    _, forced_discarded, _ = apply_two_site(truncated, 0, interaction, chi=1, cutoff=0.0)
    normalize_mps(truncated)

    z = np.linspace(-0.8, 0.9, N_SITES)
    c = np.zeros((N_SITES, N_SITES), dtype=np.float64)
    c[np.arange(N_SITES - 1), np.arange(1, N_SITES)] = 4.0e-5
    c[np.arange(1, N_SITES), np.arange(N_SITES - 1)] = -1.0e-5
    first = mps_product_uniform()
    second = mps_product_uniform()
    diag_a = tebd_step(first, z, c, 8, 1.0e-10, np.random.default_rng(DEFAULT_SEED))
    diag_b = tebd_step(second, z, c, 8, 1.0e-10, np.random.default_rng(DEFAULT_SEED))
    deterministic = all(np.array_equal(a, b) for a, b in zip(first, second))
    norm_error = abs(mps_norm_sq(first) - 1.0)
    result = {
        "passed": bool(
            np.max(np.abs(LOCAL_GENERATOR - LOCAL_GENERATOR.conj().T)) < 1.0e-12
            and np.max(np.abs(BOND_GENERATOR - BOND_GENERATOR.conj().T)) < 1.0e-12
            and np.max(np.abs(local.conj().T @ local - np.eye(LOCAL_DIM))) < 1.0e-12
            and np.max(np.abs(interaction.conj().T @ interaction - np.eye(LOCAL_DIM ** 2))) < 1.0e-12
            and abs(np.vdot(dense_before, dense_before).real - 1.0) < 1.0e-12
            and abs(np.vdot(dense_after, dense_after).real - 1.0) < 1.0e-12
            and abs(float(born[0].sum()) - 1.0) < 1.0e-12
            and discarded < 1.0e-12
            and forced_discarded > 1.0e-12
            and entropy_max > 1.0e-8
            and schmidt_rank >= 2
            and norm_error < 1.0e-12
            and deterministic
            and diag_a.jump_events == diag_b.jump_events
        ),
        "local_generator_hermitian_error": float(np.max(np.abs(LOCAL_GENERATOR - LOCAL_GENERATOR.conj().T))),
        "bond_generator_hermitian_error": float(np.max(np.abs(BOND_GENERATOR - BOND_GENERATOR.conj().T))),
        "local_unitary_error": float(np.max(np.abs(local.conj().T @ local - np.eye(LOCAL_DIM)))),
        "bond_unitary_error": float(np.max(np.abs(interaction.conj().T @ interaction - np.eye(LOCAL_DIM ** 2)))),
        "exact_split_discarded_weight": discarded,
        "forced_chi_one_discarded_weight": forced_discarded,
        "two_site_max_schmidt_entropy": entropy_max,
        "two_site_schmidt_rank": schmidt_rank,
        "seed_reproducible": deterministic,
        "seed_replay_jump_events": diag_a.jump_events,
        "seed_replay_norm_error": norm_error,
    }
    return result


def run(max_steps: int, chi: int, cutoff: float, seed: int, oos_fraction: float,
        out_dir: Path) -> dict[str, object]:
    if max_steps < 4 or chi < 1 or not (0.0 <= cutoff < 1.0) or not (0.0 < oos_fraction < 1.0):
        raise ContractError("invalid max-steps, chi, cutoff, or OOS fraction")
    times, x = load_prices()
    coupling = load_coupling()
    first = int(np.searchsorted(times, coupling["times"][0], side="left"))
    while first < len(times) - 2 and (first == 0 or times[first] - times[first - 1] != DT_NS):
        first += 1
    if first >= len(times) - 2:
        raise ContractError("no contiguous MPS start at/after first coupling snapshot")
    end = min(len(times) - 1, first + max_steps)
    if end - first < 3:
        raise ContractError("requested run is too short")
    oos_start = first + int((end - first) * oos_fraction)
    cursor = int(np.searchsorted(coupling["times"], times[first], side="right") - 1)
    if cursor < 0:
        raise ContractError("no causal coupling state at MPS start")

    mps = mps_product_uniform()
    rng = np.random.default_rng(seed)
    sigma2 = np.full(N_SITES, 1.0e-8, dtype=np.float64)
    minute_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    daily: DailyAccumulator | None = None
    resets = 0
    updates = 0
    total_jumps = 0
    total_truncations = 0
    discarded_total = 0.0
    max_discarded_step = 0.0
    max_norm_error = 0.0
    max_gate_unitarity_error = 0.0
    max_schmidt_entropy = 0.0
    max_bond_dimension = 1
    train_target_counts = np.zeros(N_SITES, dtype=np.int64)
    frozen_prior: np.ndarray | None = None
    frozen_constant: int | None = None
    oos_samples = 0
    oos_model_top1 = 0
    oos_model_brier = 0.0
    oos_constant_top1 = 0
    oos_constant_brier = 0.0
    oos_prior_top1 = 0
    oos_prior_brier = 0.0

    for i in range(first, end):
        now_ns = int(times[i])
        previous_contiguous = i > 0 and contiguous_60s(times[i - 1], times[i], DT_NS)
        next_contiguous = contiguous_60s(times[i], times[i + 1], DT_NS)
        while cursor + 1 < len(coupling["times"]) and coupling["times"][cursor + 1] <= now_ns:
            cursor += 1
        coupling_time = int(coupling["times"][cursor])
        c_matrix = coupling["matrices"][cursor]
        phase = "oos" if i >= oos_start else "pre_oos"
        if not previous_contiguous:
            mps = mps_product_uniform()
            sigma2.fill(1.0e-8)
            resets += 1
            minute_rows.append(minute_record(
                timestamp_ns=now_ns, row_index=i, previous_ns=int(times[i - 1]),
                next_ns=int(times[i + 1]), previous_contiguous=False,
                next_contiguous=next_contiguous, reason="post_gap_reset_skip", phase=phase,
                reset=True, coupling_time_ns=coupling_time, coupling_matrix=c_matrix,
            ))
            continue
        if not next_contiguous:
            minute_rows.append(minute_record(
                timestamp_ns=now_ns, row_index=i, previous_ns=int(times[i - 1]),
                next_ns=int(times[i + 1]), previous_contiguous=True,
                next_contiguous=False, reason="pre_gap_skip", phase=phase, reset=False,
                coupling_time_ns=coupling_time, coupling_matrix=c_matrix,
            ))
            continue

        log_return = x[i] - x[i - 1]
        if not np.isfinite(log_return).all():
            raise ContractError("canonical log return is non-finite")
        sigma2 = (1.0 - EWMA_ALPHA) * sigma2 + EWMA_ALPHA * np.square(log_return)
        standardized = log_return / np.sqrt(np.maximum(sigma2, 1.0e-16))
        step = tebd_step(mps, standardized, c_matrix, chi, cutoff, rng)
        norm_error = abs(mps_norm_sq(mps) - 1.0)
        if norm_error > 1.0e-10:
            raise ContractError(f"MPS norm contract failed at row {i}: {norm_error}")
        born = local_probabilities(mps)
        activity = activity_distribution(born)
        mean_entropy, step_max_entropy, step_max_rank = schmidt_metrics(mps)
        next_return = x[i + 1] - x[i]
        target = int(np.argmax(np.abs(next_return)))
        if i < oos_start:
            train_target_counts[target] += 1
        elif frozen_prior is None:
            total = int(train_target_counts.sum())
            if total <= 0:
                raise ContractError("OOS begins without an eligible training target")
            frozen_prior = train_target_counts / total
            frozen_constant = int(np.argmax(train_target_counts))
        else:
            assert frozen_constant is not None
            one_hot = np.zeros(N_SITES, dtype=np.float64)
            one_hot[target] = 1.0
            constant = np.zeros(N_SITES, dtype=np.float64)
            constant[frozen_constant] = 1.0
            oos_samples += 1
            oos_model_top1 += int(np.argmax(activity) == target)
            oos_model_brier += float(np.sum(np.square(activity - one_hot)))
            oos_constant_top1 += int(frozen_constant == target)
            oos_constant_brier += float(np.sum(np.square(constant - one_hot)))
            oos_prior_top1 += int(np.argmax(frozen_prior) == target)
            oos_prior_brier += float(np.sum(np.square(frozen_prior - one_hot)))

        day = now_ns // 86_400_000_000_000
        if daily is None or daily.day != day:
            if daily is not None and daily.updates:
                daily_rows.append(daily.as_row())
            daily = DailyAccumulator(day=day)
        daily.add(activity, mean_entropy=mean_entropy, max_entropy=step_max_entropy,
                  max_bond_dim=step_max_rank, step=step, norm_error=norm_error)
        updates += 1
        total_jumps += step.jump_events
        total_truncations += step.truncation_events
        discarded_total += step.discarded_weight
        max_discarded_step = max(max_discarded_step, step.discarded_weight)
        max_norm_error = max(max_norm_error, norm_error)
        max_gate_unitarity_error = max(max_gate_unitarity_error, step.max_gate_unitarity_error)
        max_schmidt_entropy = max(max_schmidt_entropy, step_max_entropy)
        max_bond_dimension = max(max_bond_dimension, step_max_rank)
        minute_rows.append(minute_record(
            timestamp_ns=now_ns, row_index=i, previous_ns=int(times[i - 1]),
            next_ns=int(times[i + 1]), previous_contiguous=True, next_contiguous=True,
            reason="updated", phase=phase, reset=False, coupling_time_ns=coupling_time,
            coupling_matrix=c_matrix, log_return=log_return,
            standardized_return=standardized, activity=activity, local_born=born,
            next_log_return=next_return, target_index=target,
            mean_schmidt_entropy=mean_entropy, max_schmidt_entropy=step_max_entropy,
            max_bond_dimension=step_max_rank, jump_events=step.jump_events,
            truncation_events=step.truncation_events,
            discarded_weight=step.discarded_weight, norm_error=norm_error,
        ))
    if daily is not None and daily.updates:
        daily_rows.append(daily.as_row())
    if updates == 0 or oos_samples == 0 or frozen_prior is None or frozen_constant is None:
        raise ContractError("run produced no eligible updates or OOS diagnostic")

    out_dir.mkdir(parents=True, exist_ok=True)
    minute_path = out_dir / "quantum_mps_minute.parquet"
    daily_path = out_dir / "quantum_mps_daily.parquet"
    summary_path = out_dir / "quantum_mps_summary.json"
    validation_path = out_dir / "quantum_mps_validation.json"
    pd.DataFrame.from_records(minute_rows).to_parquet(minute_path, index=False)
    pd.DataFrame.from_records(daily_rows).to_parquet(daily_path, index=False)
    validation = self_check()
    validation.update({
        "run_norm_contract_passed": max_norm_error < 1.0e-10,
        "run_finite_metrics": bool(np.isfinite([
            discarded_total, max_discarded_step, max_norm_error,
            max_gate_unitarity_error, max_schmidt_entropy,
        ]).all()),
    })
    validation["passed"] = bool(validation["passed"] and validation["run_norm_contract_passed"]
                                and validation["run_finite_metrics"])
    with open(validation_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(validation, handle, indent=2, allow_nan=False)
        handle.write("\n")
    summary: dict[str, object] = {
        "version": VERSION,
        "status": "EXPERIMENTAL_NONPROMOTION",
        "boundary": (
            "Open-quantum-system mathematics in an algorithmic representation; no claim "
            "that FX markets are quantum physical systems and no trading use."
        ),
        "input": {
            "directory": "data/canonical",
            "instrument_order": list(PAIRS),
            "strict_common_rows": int(len(times)),
            "first_processed_timestamp": str(utc_timestamp(int(times[first]))),
            "last_processed_timestamp": str(utc_timestamp(int(times[end - 1]))),
            "coupling_path": "data/derived/coupling_estimates.parquet",
            "causal_alignment": "latest coupling update_time <= current source timestamp",
            "gap_policy": "reset and skip after any non-60-second prior interval; skip pre-gap row",
        },
        "model": {
            "representation": "10-site qutrit MPS pure-state quantum trajectory",
            "local_basis": ["down", "neutral", "up"],
            "chain_edges": [[PAIRS[i], PAIRS[i + 1]] for i in range(N_SITES - 1)],
            "integrator": "second-order Strang TEBD: local half, even half, odd full, even half, local half",
            "max_bond_dimension_chi": chi,
            "relative_svd_cutoff": cutoff,
            "edge_projection": "0.5*(C_ij+C_ji)*60^2, clipped to [-0.35,0.35] radians before gain",
            "per_site_dephasing_jump_probability": DEPHASING_PER_SITE_STEP,
            "seed": seed,
        },
        "replay": {
            "source_rows_requested": max_steps,
            "source_rows_processed": int(end - first),
            "valid_updates": updates,
            "gap_resets": resets,
            "jump_events": total_jumps,
            "truncation_events": total_truncations,
            "discarded_weight_total": discarded_total,
            "discarded_weight_max_step": max_discarded_step,
            "max_norm_error": max_norm_error,
            "max_gate_unitarity_error": max_gate_unitarity_error,
            "max_schmidt_entropy": max_schmidt_entropy,
            "max_observed_bond_dimension": max_bond_dimension,
        },
        "diagnostic": {
            "target": "index of largest absolute valid next-60-second log return across the 10 pairs",
            "model_output": "normalized local Born non-neutral mass",
            "oos_fraction_final_chronological": oos_fraction,
            "oos_samples": oos_samples,
            "model_top1_accuracy": oos_model_top1 / oos_samples,
            "model_brier": oos_model_brier / oos_samples,
            "uniform_expected_top1": 1.0 / N_SITES,
            "uniform_brier": 1.0 - 1.0 / N_SITES,
            "frozen_constant_index": frozen_constant,
            "frozen_constant_pair": PAIRS[frozen_constant],
            "frozen_constant_top1_accuracy": oos_constant_top1 / oos_samples,
            "frozen_constant_brier": oos_constant_brier / oos_samples,
            "frozen_empirical_prior": frozen_prior.tolist(),
            "frozen_prior_top1_accuracy": oos_prior_top1 / oos_samples,
            "frozen_prior_brier": oos_prior_brier / oos_samples,
            "promotion": "REJECTED: diagnostic only; requires a predeclared holdout win against every frozen baseline before any further consideration",
        },
        "outputs": {
            "minute": str(minute_path.relative_to(ROOT)).replace("\\", "/"),
            "daily": str(daily_path.relative_to(ROOT)).replace("\\", "/"),
            "summary": str(summary_path.relative_to(ROOT)).replace("\\", "/"),
            "validation": str(validation_path.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    with open(summary_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"[QUANTUM_MPS] rows={end-first} updates={updates} resets={resets} chi={chi} "
          f"max_bond={max_bond_dimension} discarded={discarded_total:.6e}")
    print(f"[QUANTUM_MPS] oos_top1={summary['diagnostic']['model_top1_accuracy']:.6f} "
          f"oos_brier={summary['diagnostic']['model_brier']:.6f} "
          f"constant_top1={summary['diagnostic']['frozen_constant_top1_accuracy']:.6f}")
    print(f"[QUANTUM_MPS] validation={validation_path} passed={validation['passed']}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help="bounded source rows after first causal coupling snapshot")
    parser.add_argument("--chi", type=int, default=DEFAULT_CHI,
                        help="maximum MPS bond dimension")
    parser.add_argument("--svd-cutoff", type=float, default=DEFAULT_SVD_CUTOFF,
                        help="relative singular-value retention cutoff")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="deterministic quantum-trajectory seed")
    parser.add_argument("--oos-fraction", type=float, default=DEFAULT_OOS_FRACTION,
                        help="chronological pre-OOS fraction for frozen baselines")
    parser.add_argument("--out-dir", type=Path, default=DERIVED_DIR,
                        help="output directory (default: data/derived)")
    parser.add_argument("--self-check", action="store_true",
                        help="only run deterministic numerical checks")
    args = parser.parse_args(argv)
    try:
        if args.self_check:
            result = self_check()
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result["passed"] else 1
        run(args.max_steps, args.chi, args.svd_cutoff, args.seed,
            args.oos_fraction, args.out_dir.resolve())
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

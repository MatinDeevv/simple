"""Monte Carlo wavefunction experiment on the causal three-pair canonical slice.

This is an open-quantum-system *statistical encoding*.  It uses genuine
qutrit quantum trajectories and a Lindblad dephasing unraveling, but makes no
claim that FX markets are physical quantum systems.  It is an isolated
experiment and does not participate in canonical state or trading logic.
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


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "data_canonical"
DERIVED_DIR = ROOT / "data_derived"

PAIRS = ("EURUSD", "USDJPY", "USDCNH")
PAIR_INDICES = (0, 1, 5)
DIMENSION = len(PAIRS)

DT_S = 60.0
DT_NS = int(DT_S * 1_000_000_000)
HALFLIFE_STEPS = 1_440.0
EWMA_ALPHA = 1.0 - math.exp(-math.log(2.0) / HALFLIFE_STEPS)
PHASE_GAIN = 0.08
COUPLING_GAIN = 1.0
MAX_STANDARDIZED_RETURN = 5.0

# For L_j = sqrt(gamma) |j><j|, coherences contract by exp(-gamma * dt).
DEPHASING_PER_STEP = 0.02
NO_JUMP_PROBABILITY = 1.0 - DEPHASING_PER_STEP
DEPHASING_RATE_PER_S = -math.log(NO_JUMP_PROBABILITY) / DT_S

DEFAULT_MAX_STEPS = 250_000
DEFAULT_TRAJECTORIES = 96
DEFAULT_SEED = 20260718
DEFAULT_OOS_FRACTION = 0.70
VERSION = "quantum-trajectories-1.0.0"


class ContractError(RuntimeError):
    """The experiment's causal/physicality contract was violated."""


@dataclass
class DailyAccumulator:
    day: int
    samples: int = 0
    probability_sum: np.ndarray | None = None
    purity_sum: float = 0.0
    jump_events: int = 0
    max_trace_error: float = 0.0
    max_hermitian_error: float = 0.0
    min_eigenvalue: float = float("inf")
    max_state_norm_error: float = 0.0

    def add(self, probabilities: np.ndarray, purity: float, jump_events: int,
            trace_error: float, hermitian_error: float, min_eigenvalue: float,
            state_norm_error: float) -> None:
        if self.probability_sum is None:
            self.probability_sum = np.zeros(DIMENSION, dtype=np.float64)
        self.samples += 1
        self.probability_sum += probabilities
        self.purity_sum += purity
        self.jump_events += jump_events
        self.max_trace_error = max(self.max_trace_error, trace_error)
        self.max_hermitian_error = max(self.max_hermitian_error, hermitian_error)
        self.min_eigenvalue = min(self.min_eigenvalue, min_eigenvalue)
        self.max_state_norm_error = max(self.max_state_norm_error, state_norm_error)

    def as_row(self) -> dict[str, object]:
        if self.samples <= 0 or self.probability_sum is None:
            raise ContractError("cannot materialize an empty daily accumulator")
        p = self.probability_sum / self.samples
        entropy = -float(np.sum(np.where(p > 0.0, p * np.log(p), 0.0)))
        return {
            "timestamp": utc_timestamp(self.day * 86_400_000_000_000),
            "updates": self.samples,
            "p_eurusd_mean": float(p[0]),
            "p_usdjpy_mean": float(p[1]),
            "p_usdcnh_mean": float(p[2]),
            "born_shannon_entropy_mean_probability": entropy,
            "density_purity_mean": self.purity_sum / self.samples,
            "jump_events": self.jump_events,
            "max_trace_error": self.max_trace_error,
            "max_hermitian_error": self.max_hermitian_error,
            "min_eigenvalue": self.min_eigenvalue,
            "max_state_norm_error": self.max_state_norm_error,
        }


def epoch_ns(values: object) -> np.ndarray:
    return pd.DatetimeIndex(values).as_unit("ns").asi8


def utc_timestamp(ns: int) -> pd.Timestamp:
    return pd.Timestamp(ns, unit="ns", tz="UTC")


def load_prices() -> tuple[np.ndarray, np.ndarray]:
    """Load only the three canonical closing-price series and inner-join them."""
    joined: pd.DataFrame | None = None
    for pair in PAIRS:
        frame = pd.read_parquet(CANONICAL_DIR / f"{pair}.parquet",
                                columns=["timestamp", "close"])
        frame = frame.rename(columns={"close": pair})
        joined = frame if joined is None else joined.merge(
            frame, on="timestamp", how="inner", validate="one_to_one"
        )
    assert joined is not None
    joined = joined.sort_values("timestamp", kind="stable").reset_index(drop=True)
    times = epoch_ns(joined["timestamp"])
    close = joined.loc[:, list(PAIRS)].to_numpy(dtype=np.float64)
    if (not np.all(np.diff(times) > 0) or not np.isfinite(close).all()
            or np.any(close <= 0.0)):
        raise ContractError("canonical three-pair input is invalid")
    return times, np.log(close)


def load_coupling() -> dict[str, np.ndarray]:
    """Load the accepted causal coupling process without modifying it."""
    path = DERIVED_DIR / "coupling_estimates.parquet"
    frame = pd.read_parquet(path)
    required = {"update_time", "affected_index", "source_index", "c_ij_s_minus_2"}
    if not required.issubset(frame.columns):
        raise ContractError("coupling estimate schema is incomplete")
    sub = frame[frame["affected_index"].isin(PAIR_INDICES)
                & frame["source_index"].isin(PAIR_INDICES)]
    columns = pd.MultiIndex.from_product([PAIR_INDICES, PAIR_INDICES])
    wide = sub.pivot(index="update_time", columns=["affected_index", "source_index"],
                     values="c_ij_s_minus_2").reindex(columns=columns)
    if wide.empty or wide.isna().any().any():
        raise ContractError("three-pair coupling submatrix is incomplete")
    matrices = wide.to_numpy(dtype=np.float64).reshape(-1, DIMENSION, DIMENSION)
    diagonal = np.diagonal(matrices, axis1=1, axis2=2)
    if not np.isfinite(matrices).all() or not np.array_equal(diagonal, np.zeros_like(diagonal)):
        raise ContractError("coupling matrices violate finite/zero-diagonal contract")
    return {"times": epoch_ns(wide.index), "matrices": matrices}


def hamiltonian(standardized_return: np.ndarray, coupling: np.ndarray) -> np.ndarray:
    """Return a trace-free Hermitian qutrit Hamiltonian, in radians per step."""
    z = np.clip(np.asarray(standardized_return, dtype=np.float64),
                -MAX_STANDARDIZED_RETURN, MAX_STANDARDIZED_RETURN)
    symmetric_coupling = 0.5 * (coupling + coupling.T) * DT_S * DT_S
    H = PHASE_GAIN * (np.diag(z) + COUPLING_GAIN * symmetric_coupling)
    H = 0.5 * (H + H.T)
    H -= np.eye(DIMENSION) * np.trace(H) / DIMENSION
    return H.astype(np.complex128, copy=False)


def unitary_from_hamiltonian(H: np.ndarray) -> np.ndarray:
    """Compute exp(-iH) via the Hermitian eigendecomposition."""
    values, vectors = np.linalg.eigh(H)
    return (vectors * np.exp(-1j * values)) @ vectors.conj().T


def lindblad_jump_operators() -> tuple[np.ndarray, ...]:
    """L_j=sqrt(gamma)|j><j|, an explicit pure-dephasing Lindblad bath."""
    operators: list[np.ndarray] = []
    for j in range(DIMENSION):
        op = np.zeros((DIMENSION, DIMENSION), dtype=np.complex128)
        op[j, j] = math.sqrt(DEPHASING_RATE_PER_S)
        operators.append(op)
    return tuple(operators)


def finite_step_kraus(U: np.ndarray) -> tuple[np.ndarray, ...]:
    """Exact finite-step Kraus operators for unitary evolution plus dephasing.

    K_0=sqrt(q)U and K_j=sqrt(1-q)|j><j|U, q=exp(-gamma*dt).
    They obey sum K_a^dagger K_a=I and unravel the stated Lindblad channel.
    """
    operators = [math.sqrt(NO_JUMP_PROBABILITY) * U]
    for j in range(DIMENSION):
        projector = np.zeros((DIMENSION, DIMENSION), dtype=np.complex128)
        projector[j, j] = 1.0
        operators.append(math.sqrt(1.0 - NO_JUMP_PROBABILITY) * projector @ U)
    return tuple(operators)


def maximally_mixed_ensemble(trajectories: int) -> np.ndarray:
    """Pure normalized vectors whose reconstructed density is exactly I/3."""
    if trajectories <= 0 or trajectories % DIMENSION != 0:
        raise ContractError(f"trajectory count must be a positive multiple of {DIMENSION}")
    states = np.zeros((trajectories, DIMENSION), dtype=np.complex128)
    states[np.arange(trajectories), np.arange(trajectories) % DIMENSION] = 1.0
    return states


def reconstruct_density(states: np.ndarray) -> np.ndarray:
    """rho = mean_n |psi_n><psi_n|, using row-wise state vectors."""
    return states.T @ states.conj() / states.shape[0]


def density_checks(rho: np.ndarray, states: np.ndarray) -> tuple[float, float, float, float]:
    trace_error = abs(float(np.trace(rho).real) - 1.0)
    hermitian_error = float(np.max(np.abs(rho - rho.conj().T)))
    min_eigenvalue = float(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)).min())
    norm_error = float(np.max(np.abs(np.sum(np.abs(states) ** 2, axis=1) - 1.0)))
    return trace_error, hermitian_error, min_eigenvalue, norm_error


def analytic_dephasing_channel(rho: np.ndarray, U: np.ndarray) -> np.ndarray:
    """The exact density map associated with finite_step_kraus()."""
    rotated = U @ rho @ U.conj().T
    return (NO_JUMP_PROBABILITY * rotated
            + (1.0 - NO_JUMP_PROBABILITY) * np.diag(np.diag(rotated)))


def advance_ensemble(states: np.ndarray, U: np.ndarray,
                     rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """One seeded Monte Carlo wavefunction step of the dephasing unraveling."""
    # Column convention is |psi'>=U|psi>; row storage requires multiplication by U.T.
    advanced = states @ U.T
    uniforms = rng.random(advanced.shape[0])
    jump_mask = uniforms >= NO_JUMP_PROBABILITY
    jump_count = int(np.count_nonzero(jump_mask))
    if jump_count:
        born = np.abs(advanced[jump_mask]) ** 2
        born /= born.sum(axis=1, keepdims=True)
        choices = np.sum(
            ((uniforms[jump_mask] - NO_JUMP_PROBABILITY)
             / (1.0 - NO_JUMP_PROBABILITY))[:, None]
            > np.cumsum(born, axis=1),
            axis=1,
        )
        advanced[jump_mask] = 0.0
        advanced[np.flatnonzero(jump_mask), choices] = 1.0

    # The mathematical maps preserve norm.  This only removes floating-point round-off.
    norms = np.sqrt(np.sum(np.abs(advanced) ** 2, axis=1, keepdims=True))
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ContractError("trajectory acquired a non-finite or zero norm")
    advanced /= norms
    return advanced, jump_count


def _lindblad_generator(rho: np.ndarray) -> np.ndarray:
    result = np.zeros_like(rho)
    for L in lindblad_jump_operators():
        LdagL = L.conj().T @ L
        result += L @ rho @ L.conj().T - 0.5 * (LdagL @ rho + rho @ LdagL)
    return result


def minute_record(*, timestamp_ns: int, row_index: int, previous_ns: int,
                  next_ns: int, previous_contiguous: bool, next_contiguous: bool,
                  reason: str, phase: str, reset: bool,
                  coupling_time_ns: int | None, coupling_matrix: np.ndarray | None,
                  log_return: np.ndarray | None = None,
                  standardized_return: np.ndarray | None = None,
                  probabilities: np.ndarray | None = None,
                  next_log_return: np.ndarray | None = None,
                  target_index: int = -1, jump_events: int = 0,
                  density_purity: float = float("nan")) -> dict[str, object]:
    """Persist every run-row input, decision, and skip reason for replay audits."""
    nan_vector = np.full(DIMENSION, np.nan, dtype=np.float64)
    log_return = nan_vector if log_return is None else log_return
    standardized_return = nan_vector if standardized_return is None else standardized_return
    probabilities = nan_vector if probabilities is None else probabilities
    next_log_return = nan_vector if next_log_return is None else next_log_return
    coupling_age_s = (float(timestamp_ns - coupling_time_ns) / 1_000_000_000
                      if coupling_time_ns is not None else float("nan"))
    record: dict[str, object] = {
        "timestamp": utc_timestamp(timestamp_ns),
        "row_index": row_index,
        "previous_timestamp": utc_timestamp(previous_ns),
        "next_timestamp": utc_timestamp(next_ns),
        "previous_contiguous_60s": previous_contiguous,
        "next_contiguous_60s": next_contiguous,
        "reason": reason,
        "chronological_phase": phase,
        "ensemble_reset": reset,
        "coupling_update_time": (utc_timestamp(coupling_time_ns)
                                 if coupling_time_ns is not None else pd.NaT),
        "coupling_age_seconds": coupling_age_s,
        "log_return_eurusd": float(log_return[0]),
        "log_return_usdjpy": float(log_return[1]),
        "log_return_usdcnh": float(log_return[2]),
        "z_eurusd": float(standardized_return[0]),
        "z_usdjpy": float(standardized_return[1]),
        "z_usdcnh": float(standardized_return[2]),
        "p_eurusd": float(probabilities[0]),
        "p_usdjpy": float(probabilities[1]),
        "p_usdcnh": float(probabilities[2]),
        "next_log_return_eurusd": float(next_log_return[0]),
        "next_log_return_usdjpy": float(next_log_return[1]),
        "next_log_return_usdcnh": float(next_log_return[2]),
        "next_target_index": target_index,
        "jump_events": jump_events,
        "density_purity": density_purity,
    }
    for affected in range(DIMENSION):
        for source in range(DIMENSION):
            record[f"c_{affected}{source}_s_minus_2"] = (
                float(coupling_matrix[affected, source])
                if coupling_matrix is not None else float("nan")
            )
    return record


def self_check() -> dict[str, object]:
    """Deterministic numerical checks for quantum and stochastic contracts."""
    C = np.array([[0.0, 1e-5, -2e-5], [1e-5, 0.0, 3e-5],
                  [-2e-5, 3e-5, 0.0]], dtype=np.float64)
    H = hamiltonian(np.array([1.0, -0.5, 0.2]), C)
    U = unitary_from_hamiltonian(H)
    kraus = finite_step_kraus(U)
    kraus_complete = sum((K.conj().T @ K for K in kraus), np.zeros_like(U))
    ket = np.array([1.0, 1j, -1.0], dtype=np.complex128) / math.sqrt(3.0)
    rho0 = np.outer(ket, ket.conj())
    generator_trace_error = abs(float(np.trace(_lindblad_generator(rho0)).real))

    # This verifies the MC unraveling statistically against its exact density channel.
    ensemble = np.repeat(ket[None, :], 30_000, axis=0)
    stepped, jumps = advance_ensemble(ensemble, U, np.random.default_rng(DEFAULT_SEED))
    rho_mc = reconstruct_density(stepped)
    rho_exact = analytic_dephasing_channel(rho0, U)
    mc_channel_error = float(np.max(np.abs(rho_mc - rho_exact)))
    trace_error, hermitian_error, min_eigenvalue, norm_error = density_checks(rho_mc, stepped)

    # Fixed seeds must make full trajectory paths bitwise reproducible.
    deterministic_states_a = maximally_mixed_ensemble(96)
    deterministic_states_b = maximally_mixed_ensemble(96)
    rng_a = np.random.default_rng(DEFAULT_SEED)
    rng_b = np.random.default_rng(DEFAULT_SEED)
    for _ in range(5):
        deterministic_states_a, _ = advance_ensemble(deterministic_states_a, U, rng_a)
        deterministic_states_b, _ = advance_ensemble(deterministic_states_b, U, rng_b)
    deterministic = bool(np.array_equal(deterministic_states_a, deterministic_states_b))

    passed = bool(
        np.max(np.abs(H - H.conj().T)) < 1e-12
        and np.max(np.abs(U.conj().T @ U - np.eye(DIMENSION))) < 1e-12
        and np.max(np.abs(kraus_complete - np.eye(DIMENSION))) < 1e-12
        and generator_trace_error < 1e-12
        and mc_channel_error < 0.004
        and trace_error < 1e-12
        and hermitian_error < 1e-12
        and min_eigenvalue >= -1e-12
        and norm_error < 1e-12
        and deterministic
        and jumps > 0
    )
    return {
        "passed": passed,
        "hamiltonian_hermitian_error": float(np.max(np.abs(H - H.conj().T))),
        "unitary_error": float(np.max(np.abs(U.conj().T @ U - np.eye(DIMENSION)))),
        "kraus_completeness_error": float(
            np.max(np.abs(kraus_complete - np.eye(DIMENSION)))
        ),
        "lindblad_generator_trace_error": generator_trace_error,
        "mc_vs_exact_channel_max_error": mc_channel_error,
        "mc_trace_error": trace_error,
        "mc_hermitian_error": hermitian_error,
        "mc_min_eigenvalue": min_eigenvalue,
        "max_trajectory_norm_error": norm_error,
        "mc_jump_events": jumps,
        "seed_reproducible": deterministic,
    }


def run(max_steps: int, trajectories: int, seed: int, oos_fraction: float,
        out_dir: Path) -> dict[str, object]:
    if not 0.0 < oos_fraction < 1.0:
        raise ContractError("--oos-fraction must lie strictly between zero and one")
    states = maximally_mixed_ensemble(trajectories)
    rng = np.random.default_rng(seed)
    times, x = load_prices()
    coupling = load_coupling()

    first = int(np.searchsorted(times, coupling["times"][0], side="left"))
    while first < len(times) and (first == 0 or times[first] - times[first - 1] != DT_NS):
        first += 1
    if first >= len(times) - 2:
        raise ContractError("no contiguous start after first causal coupling estimate")
    end = min(len(times) - 1, first + max_steps)
    if end - first < 3:
        raise ContractError("requested run is too short")
    oos_start = first + int((end - first) * oos_fraction)

    sigma2 = np.full(DIMENSION, 1e-8, dtype=np.float64)
    coupling_cursor = int(np.searchsorted(coupling["times"], times[first], side="right") - 1)
    if coupling_cursor < 0:
        raise ContractError("no causal coupling state at experiment start")

    daily: list[dict[str, object]] = []
    minute: list[dict[str, object]] = []
    accumulator: DailyAccumulator | None = None
    gap_resets = 0
    continuous_steps = 0
    total_jump_events = 0
    max_hamiltonian_hermitian_error = 0.0
    max_unitary_error = 0.0
    max_trace_error = 0.0
    max_hermitian_error = 0.0
    min_eigenvalue = float("inf")
    max_state_norm_error = 0.0
    oos_samples = 0
    oos_top1_hits = 0
    oos_brier_sum = 0.0
    train_target_counts = np.zeros(DIMENSION, dtype=np.int64)
    frozen_prior: np.ndarray | None = None
    frozen_constant_index: int | None = None
    oos_constant_top1_hits = 0
    oos_constant_brier_sum = 0.0
    oos_prior_top1_hits = 0
    oos_prior_brier_sum = 0.0

    for i in range(first, end):
        now_ns = int(times[i])
        previous_contiguous = i > 0 and times[i] - times[i - 1] == DT_NS
        next_contiguous = times[i + 1] - times[i] == DT_NS
        while (coupling_cursor + 1 < len(coupling["times"])
               and coupling["times"][coupling_cursor + 1] <= now_ns):
            coupling_cursor += 1
        coupling_time_ns = int(coupling["times"][coupling_cursor])
        coupling_matrix = coupling["matrices"][coupling_cursor]
        phase = "oos" if i >= oos_start else "pre_oos"
        if not previous_contiguous:
            # Reset before any return is formed: x[i]-x[i-1] crosses the gap.
            states = maximally_mixed_ensemble(trajectories)
            sigma2.fill(1e-8)
            gap_resets += 1
            minute.append(minute_record(
                timestamp_ns=now_ns, row_index=i, previous_ns=int(times[i - 1]),
                next_ns=int(times[i + 1]), previous_contiguous=previous_contiguous,
                next_contiguous=next_contiguous, reason="post_gap_reset_skip",
                phase=phase, reset=True, coupling_time_ns=coupling_time_ns,
                coupling_matrix=coupling_matrix,
            ))
            continue
        if not next_contiguous:
            # Do not form a state or next-bar target across a missing/session interval.
            minute.append(minute_record(
                timestamp_ns=now_ns, row_index=i, previous_ns=int(times[i - 1]),
                next_ns=int(times[i + 1]), previous_contiguous=previous_contiguous,
                next_contiguous=next_contiguous, reason="pre_gap_skip",
                phase=phase, reset=False, coupling_time_ns=coupling_time_ns,
                coupling_matrix=coupling_matrix,
            ))
            continue

        ret = x[i] - x[i - 1]
        sigma2 = (1.0 - EWMA_ALPHA) * sigma2 + EWMA_ALPHA * ret * ret
        z = ret / np.sqrt(np.maximum(sigma2, 1e-16))
        H = hamiltonian(z, coupling["matrices"][coupling_cursor])
        U = unitary_from_hamiltonian(H)
        states, jump_events = advance_ensemble(states, U, rng)
        rho = reconstruct_density(states)
        trace_error, hermitian_error, eig_min, norm_error = density_checks(rho, states)
        hamiltonian_error = float(np.max(np.abs(H - H.conj().T)))
        unitary_error = float(np.max(np.abs(U.conj().T @ U - np.eye(DIMENSION))))
        max_hamiltonian_hermitian_error = max(max_hamiltonian_hermitian_error, hamiltonian_error)
        max_unitary_error = max(max_unitary_error, unitary_error)
        max_trace_error = max(max_trace_error, trace_error)
        max_hermitian_error = max(max_hermitian_error, hermitian_error)
        min_eigenvalue = min(min_eigenvalue, eig_min)
        max_state_norm_error = max(max_state_norm_error, norm_error)
        total_jump_events += jump_events
        probabilities = np.maximum(np.diag(rho).real, 0.0)
        probabilities /= probabilities.sum()
        purity = float(np.trace(rho @ rho).real)

        next_ret = x[i + 1] - x[i]
        target = int(np.argmax(np.abs(next_ret)))
        minute.append(minute_record(
            timestamp_ns=now_ns, row_index=i, previous_ns=int(times[i - 1]),
            next_ns=int(times[i + 1]), previous_contiguous=previous_contiguous,
            next_contiguous=next_contiguous, reason="updated_and_scored",
            phase=phase, reset=False, coupling_time_ns=coupling_time_ns,
            coupling_matrix=coupling_matrix, log_return=ret, standardized_return=z,
            probabilities=probabilities, next_log_return=next_ret, target_index=target,
            jump_events=jump_events, density_purity=purity,
        ))

        day = int(now_ns // 86_400_000_000_000)
        if accumulator is None:
            accumulator = DailyAccumulator(day)
        elif day != accumulator.day:
            daily.append(accumulator.as_row())
            accumulator = DailyAccumulator(day)
        accumulator.add(probabilities, purity, jump_events, trace_error, hermitian_error,
                        eig_min, norm_error)

        # Targets are never state inputs.  Freeze all non-model baselines before OOS.
        if i < oos_start:
            train_target_counts[target] += 1
        else:
            if frozen_prior is None:
                if int(train_target_counts.sum()) <= 0:
                    raise ContractError("no pre-OOS targets exist for frozen baselines")
                frozen_prior = train_target_counts / train_target_counts.sum()
                frozen_constant_index = int(np.argmax(frozen_prior))
            one_hot = np.zeros(DIMENSION, dtype=np.float64)
            one_hot[target] = 1.0
            oos_top1_hits += int(np.argmax(probabilities) == target)
            oos_brier_sum += float(np.sum((probabilities - one_hot) ** 2))
            constant_prediction = np.zeros(DIMENSION, dtype=np.float64)
            assert frozen_constant_index is not None
            constant_prediction[frozen_constant_index] = 1.0
            oos_constant_top1_hits += int(frozen_constant_index == target)
            oos_constant_brier_sum += float(np.sum((constant_prediction - one_hot) ** 2))
            oos_prior_top1_hits += int(np.argmax(frozen_prior) == target)
            oos_prior_brier_sum += float(np.sum((frozen_prior - one_hot) ** 2))
            oos_samples += 1
        continuous_steps += 1

    if accumulator is not None and accumulator.samples:
        daily.append(accumulator.as_row())
    if not daily or continuous_steps <= 0 or oos_samples <= 0:
        raise ContractError("experiment did not produce sufficient contiguous/OOS samples")
    if frozen_prior is None or frozen_constant_index is None:
        raise ContractError("chronological baselines were not frozen")
    if (max_hamiltonian_hermitian_error >= 1e-10 or max_unitary_error >= 1e-10
            or max_trace_error >= 1e-10 or max_hermitian_error >= 1e-10
            or min_eigenvalue < -1e-10 or max_state_norm_error >= 1e-10):
        raise ContractError("open-quantum-system physicality check failed")

    out_dir.mkdir(parents=True, exist_ok=True)
    daily_path = out_dir / "quantum_trajectories_daily.parquet"
    minute_path = out_dir / "quantum_trajectories_minute.parquet"
    summary_path = out_dir / "quantum_trajectories_summary.json"
    validation_path = out_dir / "quantum_trajectories_validation.json"
    pd.DataFrame(daily).to_parquet(daily_path, index=False, compression="zstd")
    pd.DataFrame(minute).to_parquet(minute_path, index=False, compression="zstd")
    validation = self_check()
    validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    if not validation["passed"]:
        raise ContractError("quantum trajectory self-check failed")

    summary = {
        "version": VERSION,
        "interpretation": "open-quantum-system statistical encoding; not evidence of a physical quantum FX market",
        "pair_scope": list(PAIRS),
        "pair_indices": list(PAIR_INDICES),
        "numerical_backend": "NumPy only for quantum linear algebra and stochastic trajectories; pandas only for local Parquet I/O",
        "span": [utc_timestamp(int(times[first])).isoformat(),
                 utc_timestamp(int(times[end])).isoformat()],
        "rows_processed": int(end - first),
        "continuous_60s_steps": continuous_steps,
        "gap_ensemble_resets": gap_resets,
        "trajectory_ensemble": {
            "trajectories": trajectories,
            "seed": seed,
            "dimension": DIMENSION,
            "initial_density": "I/3 reconstructed from equally allocated normalized basis-state trajectories",
            "jump_operators": "L_j=sqrt(gamma)|j><j|, j=0,1,2",
            "dephasing_per_60s_step": DEPHASING_PER_STEP,
            "dephasing_rate_per_s": DEPHASING_RATE_PER_S,
            "no_jump_probability": NO_JUMP_PROBABILITY,
            "jump_events": total_jump_events,
            "observed_jump_rate": total_jump_events / (continuous_steps * trajectories),
        },
        "physics_checks": {
            "max_hamiltonian_hermitian_error": max_hamiltonian_hermitian_error,
            "max_unitary_error": max_unitary_error,
            "max_density_trace_error": max_trace_error,
            "max_density_hermitian_error": max_hermitian_error,
            "min_reconstructed_density_eigenvalue": min_eigenvalue,
            "max_trajectory_norm_error": max_state_norm_error,
        },
        "next_bar_oos_experimental_metric": {
            "split": f"fixed chronological post-{int(oos_fraction * 100)}% of this run",
            "prediction_time_start": utc_timestamp(int(times[oos_start])).isoformat(),
            "target": "which pair has the largest absolute next valid 60-second return",
            "samples": oos_samples,
            "top1_accuracy": oos_top1_hits / oos_samples,
            "uniform_top1_accuracy": 1.0 / DIMENSION,
            "brier_score": oos_brier_sum / oos_samples,
            "frozen_pre_oos_baselines": {
                "uniform": {
                    "probabilities": [1.0 / DIMENSION] * DIMENSION,
                    "expected_top1_accuracy": 1.0 / DIMENSION,
                    "brier_score": 2.0 / 3.0,
                },
                "constant": {
                    "target_index": frozen_constant_index,
                    "target_pair": PAIRS[frozen_constant_index],
                    "probabilities": [float(j == frozen_constant_index)
                                      for j in range(DIMENSION)],
                    "top1_accuracy": oos_constant_top1_hits / oos_samples,
                    "brier_score": oos_constant_brier_sum / oos_samples,
                },
                "prior": {
                    "pre_oos_target_counts": train_target_counts.tolist(),
                    "probabilities": frozen_prior.tolist(),
                    "top1_accuracy": oos_prior_top1_hits / oos_samples,
                    "brier_score": oos_prior_brier_sum / oos_samples,
                },
            },
            "claim": "prequential diagnostic only; no trading, causation, promotion, or physical-market-quantum claim",
        },
        "gap_policy": "On the first bar after any non-60-second interval, reset all pure trajectories to the I/3 ensemble and reset causal volatility before forming a return; do not update or score across a gap.",
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
    parser.add_argument("--trajectories", type=int, default=DEFAULT_TRAJECTORIES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--oos-fraction", type=float, default=DEFAULT_OOS_FRACTION)
    parser.add_argument("--out-dir", type=Path, default=DERIVED_DIR)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        result = self_check()
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    if args.max_steps < 3:
        raise SystemExit("--max-steps must be at least 3")
    try:
        run(args.max_steps, args.trajectories, args.seed, args.oos_fraction,
            args.out_dir.resolve())
        return 0
    except (ContractError, ValueError, np.linalg.LinAlgError, OSError) as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

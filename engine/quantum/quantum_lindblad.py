"""
quantum_lindblad.py -- open-quantum-system statistical representation for FX data.

The code uses density matrices, a Hermitian Hamiltonian, a complete two-outcome
Kraus instrument, and pure dephasing.  It is not evidence that FX is a
physical quantum system, and its categorical next-bar score is diagnostic only.
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

from engine.core.contracts import ContractError as SharedContractError
from engine.core.contracts import canonical_pair_order, contiguous_60s


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DIR = ROOT / "data" / "canonical"
DERIVED_DIR = ROOT / "data" / "derived"
PAIR_INDICES = (0, 1, 5)
PAIRS = tuple(canonical_pair_order(ROOT)[index] for index in PAIR_INDICES)

DT_S = 60.0
DT_NS = int(DT_S * 1_000_000_000)
HALFLIFE_STEPS = 1_440.0
EWMA_ALPHA = 1.0 - math.exp(-math.log(2.0) / HALFLIFE_STEPS)
PHASE_GAIN = 0.08
COUPLING_GAIN = 1.0
MEASUREMENT_GAIN = 0.20
DEPHASING_PER_STEP = 0.02
MAX_STANDARDIZED_RETURN = 5.0
MAX_COUPLING_AGE_S = 36.0 * 3_600.0
DEFAULT_MAX_STEPS = 250_000
VERSION = "quantum-lindblad-1.2.0"


class ContractError(RuntimeError):
    pass


def epoch_ns(values: object) -> np.ndarray:
    return pd.DatetimeIndex(values).as_unit("ns").asi8


def utc_timestamp(ns: int) -> pd.Timestamp:
    return pd.Timestamp(ns, unit="ns", tz="UTC")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_cross_gap_state_updates(records: dict[str, list[object]]) -> int:
    """Count scored emissions that claim an invalid predecessor edge.

    This is intentionally computed from the persisted minute-record fields,
    rather than maintained as an incidental loop counter.  It makes the audit
    result independently checkable from the emitted replay surface.
    """
    reasons = records.get("reason")
    predecessors = records.get("previous_contiguous_60s")
    if reasons is None or predecessors is None or len(reasons) != len(predecessors):
        raise ContractError("minute records lack aligned gap-audit fields")
    return int(sum(
        reason == "scored_contiguous" and not bool(previous)
        for reason, previous in zip(reasons, predecessors, strict=True)
    ))


def load_prices() -> tuple[np.ndarray, np.ndarray]:
    joined: pd.DataFrame | None = None
    for pair in PAIRS:
        frame = pd.read_parquet(CANONICAL_DIR / f"{pair}.parquet",
                                columns=["timestamp", "close"])
        frame = frame.rename(columns={"close": pair})
        joined = frame if joined is None else joined.merge(
            frame, on="timestamp", how="inner", validate="one_to_one")
    assert joined is not None
    joined = joined.sort_values("timestamp", kind="stable").reset_index(drop=True)
    times = epoch_ns(joined["timestamp"])
    close = joined.loc[:, list(PAIRS)].to_numpy(dtype=np.float64)
    if not np.all(np.diff(times) > 0) or not np.isfinite(close).all() or np.any(close <= 0.0):
        raise ContractError("canonical three-pair input is invalid")
    return times, np.log(close)


def load_coupling() -> dict[str, np.ndarray]:
    path = DERIVED_DIR / "coupling_estimates.parquet"
    frame = pd.read_parquet(path)
    required = {"update_time", "affected_index", "source_index", "c_ij_s_minus_2"}
    if not required.issubset(frame.columns):
        raise ContractError("coupling estimate schema is incomplete")
    sub = frame[frame["affected_index"].isin(PAIR_INDICES)
                & frame["source_index"].isin(PAIR_INDICES)]
    columns = pd.MultiIndex.from_product([PAIR_INDICES, PAIR_INDICES])
    wide = sub.pivot(index="update_time",
                     columns=["affected_index", "source_index"],
                     values="c_ij_s_minus_2").reindex(columns=columns)
    if wide.empty or wide.isna().any().any():
        raise ContractError("three-pair coupling submatrix is incomplete")
    matrices = wide.to_numpy(dtype=np.float64).reshape(-1, 3, 3)
    if not np.isfinite(matrices).all() or not np.array_equal(
            np.diagonal(matrices, axis1=1, axis2=2), np.zeros((len(matrices), 3))):
        raise ContractError("coupling matrices violate finite/zero-diagonal contract")
    return {"times": epoch_ns(wide.index), "matrices": matrices}


def unitary_from_hamiltonian(H: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(H)
    return (vectors * np.exp(-1j * values)) @ vectors.conj().T


def normalize_density(rho: np.ndarray) -> np.ndarray:
    rho = 0.5 * (rho + rho.conj().T)
    trace = float(np.trace(rho).real)
    if not math.isfinite(trace) or trace <= 0.0:
        raise ContractError("density matrix has invalid trace")
    return rho / trace


def measurement_instrument(return_signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a complete, data-conditioned diagonal two-outcome instrument.

    K0 is the conditional evidence branch used by the filter.  K1 is retained
    to make the instrument explicit and to test sum_a Ka^dagger Ka = I.  The
    observed classical return chooses this modelling likelihood; it is not a
    physical state-preparation or measurement correspondence for the market.
    """
    bounded = np.clip(return_signal, -MAX_STANDARDIZED_RETURN,
                      MAX_STANDARDIZED_RETURN)
    log_effect = MEASUREMENT_GAIN * (bounded - float(np.max(bounded)))
    effect = np.exp(log_effect)  # 0 < effect_i <= 1 by construction.
    K0 = np.diag(np.sqrt(effect)).astype(np.complex128)
    K1 = np.diag(np.sqrt(np.maximum(1.0 - effect, 0.0))).astype(np.complex128)
    completeness = K0.conj().T @ K0 + K1.conj().T @ K1
    error = float(np.max(np.abs(completeness - np.eye(3))))
    return K0, K1, error


def quantum_step(rho: np.ndarray, standardized_return: np.ndarray,
                 coupling: np.ndarray) -> tuple[np.ndarray, float]:
    """One conditional CPTP-instrument / dephasing update in a qutrit space."""
    return_signal = np.clip(standardized_return, -MAX_STANDARDIZED_RETURN,
                            MAX_STANDARDIZED_RETURN)
    symmetric_coupling = 0.5 * (coupling + coupling.T) * DT_S * DT_S
    H = PHASE_GAIN * (np.diag(return_signal) + COUPLING_GAIN * symmetric_coupling)
    H = 0.5 * (H + H.T)
    H -= np.eye(3) * np.trace(H) / 3.0
    U = unitary_from_hamiltonian(H)
    rho = U @ rho @ U.conj().T

    K0, _, completeness_error = measurement_instrument(return_signal)
    rho = normalize_density(K0 @ rho @ K0.conj().T)

    # Exact finite-step pure-dephasing channel for diagonal Lindblad operators.
    rho = ((1.0 - DEPHASING_PER_STEP) * rho
           + DEPHASING_PER_STEP * np.diag(np.diag(rho)))
    return normalize_density(rho), completeness_error


def density_checks(rho: np.ndarray) -> tuple[float, float, float]:
    trace_error = abs(float(np.trace(rho).real) - 1.0)
    hermitian_error = float(np.max(np.abs(rho - rho.conj().T)))
    min_eigenvalue = float(np.linalg.eigvalsh(rho).min())
    return trace_error, hermitian_error, min_eigenvalue


def one_hot(index: int) -> np.ndarray:
    result = np.zeros(3, dtype=np.float64)
    result[index] = 1.0
    return result


def brier(probabilities: np.ndarray, target: int) -> float:
    return float(np.sum((probabilities - one_hot(target)) ** 2))


def self_check() -> dict[str, object]:
    rho = np.eye(3, dtype=np.complex128) / 3.0
    C = np.array([[0.0, 1e-5, -2e-5], [1e-5, 0.0, 3e-5], [-2e-5, 3e-5, 0.0]])
    K0, K1, completeness_error = measurement_instrument(np.array([1.0, -0.5, 0.2]))
    rho, step_completeness_error = quantum_step(rho, np.array([1.0, -0.5, 0.2]), C)
    trace_error, hermitian_error, min_eigenvalue = density_checks(rho)
    return {
        "passed": bool(trace_error < 1e-12 and hermitian_error < 1e-12
                       and min_eigenvalue >= -1e-12
                       and completeness_error < 1e-12
                       and step_completeness_error < 1e-12
                       and np.allclose(K0.conj().T @ K0 + K1.conj().T @ K1, np.eye(3))),
        "trace_error": trace_error,
        "hermitian_error": hermitian_error,
        "min_eigenvalue": min_eigenvalue,
        "instrument_completeness_error": max(completeness_error, step_completeness_error),
    }


def append_minute(records: dict[str, list[object]], *, now_ns: int, next_ns: int,
                  reason: str, coupling_ns: int, coupling_age_s: float,
                  previous_contiguous: bool,
                  z: np.ndarray | None = None, probabilities: np.ndarray | None = None,
                  target: int | None = None, next_ret: np.ndarray | None = None,
                  matrix: np.ndarray | None = None, trace_error: float = math.nan,
                  hermitian_error: float = math.nan, min_eigenvalue: float = math.nan,
                  completeness_error: float = math.nan) -> None:
    records["timestamp"].append(utc_timestamp(now_ns))
    records["target_time"].append(utc_timestamp(next_ns))
    records["reason"].append(reason)
    records["previous_contiguous_60s"].append(previous_contiguous)
    records["coupling_update_time"].append(utc_timestamp(coupling_ns))
    records["coupling_age_s"].append(coupling_age_s)
    records["target_index"].append(np.nan if target is None else target)
    for prefix, values in (("z", z), ("p", probabilities), ("next_log_return", next_ret)):
        array = np.full(3, np.nan) if values is None else values
        for pair, value in zip(PAIRS, array, strict=True):
            records[f"{prefix}_{pair.lower()}"].append(float(value))
    flat_matrix = np.full(9, np.nan) if matrix is None else matrix.reshape(-1)
    for row in range(3):
        for col in range(3):
            records[f"c_{row}{col}_s_minus_2"].append(float(flat_matrix[3 * row + col]))
    records["trace_error"].append(trace_error)
    records["hermitian_error"].append(hermitian_error)
    records["min_eigenvalue"].append(min_eigenvalue)
    records["instrument_completeness_error"].append(completeness_error)


def run(max_steps: int, out_dir: Path, max_coupling_age_s: float) -> dict[str, object]:
    if not math.isfinite(max_coupling_age_s) or max_coupling_age_s <= 0.0:
        raise ContractError("max_coupling_age_s must be finite and positive")
    times, x = load_prices()
    coupling = load_coupling()
    first = int(np.searchsorted(times, coupling["times"][0], side="left"))
    while first < len(times) and (first == 0 or not contiguous_60s(times[first - 1], times[first], DT_NS)):
        first += 1
    if first >= len(times) - 2:
        raise ContractError("no contiguous start after first causal coupling estimate")
    end = min(len(times) - 1, first + max_steps)

    rho = np.eye(3, dtype=np.complex128) / 3.0
    sigma2 = np.full(3, 1e-8, dtype=np.float64)
    coupling_cursor = int(np.searchsorted(coupling["times"], times[first], side="right") - 1)
    if coupling_cursor < 0:
        raise ContractError("no causal coupling state at experiment start")

    records: dict[str, list[object]] = {
        key: [] for key in (
            "timestamp", "target_time", "reason", "previous_contiguous_60s", "coupling_update_time", "coupling_age_s",
            "target_index", "trace_error", "hermitian_error", "min_eigenvalue",
            "instrument_completeness_error",
            *[f"{prefix}_{pair.lower()}" for prefix in ("z", "p", "next_log_return") for pair in PAIRS],
            *[f"c_{row}{col}_s_minus_2" for row in range(3) for col in range(3)],
        )
    }
    daily: list[dict[str, object]] = []
    current_day: int | None = None
    gap_resets = 0
    leading_gap_skips = 0
    stale_coupling_skips = 0
    stale_coupling_resets = 0
    stale_active = False
    continuous_steps = 0
    target_count = 0
    top1_hits = 0
    brier_sum = 0.0
    causal_prior_hits = 0
    causal_prior_brier = 0.0
    last_magnitude_hits = 0
    last_magnitude_brier = 0.0
    target_counts = np.zeros(3, dtype=np.int64)
    max_trace_error = 0.0
    max_hermitian_error = 0.0
    min_eigenvalue = float("inf")
    max_instrument_completeness_error = 0.0
    max_coupling_age_seen_s = 0.0

    for i in range(first, end):
        now_ns = int(times[i])
        next_ns = int(times[i + 1])
        while (coupling_cursor + 1 < len(coupling["times"])
               and coupling["times"][coupling_cursor + 1] <= now_ns):
            coupling_cursor += 1
        coupling_ns = int(coupling["times"][coupling_cursor])
        coupling_age_s = (now_ns - coupling_ns) / 1_000_000_000.0
        max_coupling_age_seen_s = max(max_coupling_age_seen_s, coupling_age_s)

        previous_contiguous = i == first or contiguous_60s(int(times[i - 1]), now_ns, DT_NS)
        if not previous_contiguous:
            rho = np.eye(3, dtype=np.complex128) / 3.0
            sigma2.fill(1e-8)
            gap_resets += 1
            append_minute(records, now_ns=now_ns, next_ns=next_ns,
                          reason="cross_gap_return_reset", coupling_ns=coupling_ns,
                          coupling_age_s=coupling_age_s, previous_contiguous=False)
            continue
        if not contiguous_60s(now_ns, next_ns, DT_NS):
            leading_gap_skips += 1
            append_minute(records, now_ns=now_ns, next_ns=next_ns,
                          reason="next_bar_gap_skip", coupling_ns=coupling_ns,
                          coupling_age_s=coupling_age_s, previous_contiguous=True)
            continue
        if coupling_age_s > max_coupling_age_s:
            if not stale_active:
                rho = np.eye(3, dtype=np.complex128) / 3.0
                sigma2.fill(1e-8)
                stale_coupling_resets += 1
                stale_active = True
            stale_coupling_skips += 1
            append_minute(records, now_ns=now_ns, next_ns=next_ns,
                          reason="stale_coupling_skip", coupling_ns=coupling_ns,
                          coupling_age_s=coupling_age_s, previous_contiguous=True)
            continue
        stale_active = False

        ret = x[i] - x[i - 1]
        sigma2 = (1.0 - EWMA_ALPHA) * sigma2 + EWMA_ALPHA * ret * ret
        z = ret / np.sqrt(np.maximum(sigma2, 1e-16))
        matrix = coupling["matrices"][coupling_cursor]
        rho, completeness_error = quantum_step(rho, z, matrix)
        trace_error, hermitian_error, eig_min = density_checks(rho)
        max_trace_error = max(max_trace_error, trace_error)
        max_hermitian_error = max(max_hermitian_error, hermitian_error)
        min_eigenvalue = min(min_eigenvalue, eig_min)
        max_instrument_completeness_error = max(max_instrument_completeness_error,
                                                completeness_error)
        probabilities = np.maximum(np.diag(rho).real, 0.0)
        probabilities /= probabilities.sum()

        next_ret = x[i + 1] - x[i]
        target = int(np.argmax(np.abs(next_ret)))
        causal_prior = (target_counts + 1.0) / (float(target_counts.sum()) + 3.0)
        last_magnitude = one_hot(int(np.argmax(np.abs(ret))))
        top1_hits += int(np.argmax(probabilities) == target)
        brier_sum += brier(probabilities, target)
        causal_prior_hits += int(np.argmax(causal_prior) == target)
        causal_prior_brier += brier(causal_prior, target)
        last_magnitude_hits += int(np.argmax(last_magnitude) == target)
        last_magnitude_brier += brier(last_magnitude, target)
        target_counts[target] += 1
        target_count += 1
        continuous_steps += 1

        append_minute(records, now_ns=now_ns, next_ns=next_ns,
                      reason="scored_contiguous", coupling_ns=coupling_ns,
                      previous_contiguous=True,
                      coupling_age_s=coupling_age_s, z=z, probabilities=probabilities,
                      target=target, next_ret=next_ret, matrix=matrix,
                      trace_error=trace_error, hermitian_error=hermitian_error,
                      min_eigenvalue=eig_min, completeness_error=completeness_error)

        day = now_ns // 86_400_000_000_000
        if current_day is None or day != current_day:
            entropy = -float(np.sum(np.where(probabilities > 0.0,
                                              probabilities * np.log(probabilities), 0.0)))
            daily.append({
                "timestamp": utc_timestamp(now_ns),
                "p_eurusd": float(probabilities[0]),
                "p_usdjpy": float(probabilities[1]),
                "p_usdcnh": float(probabilities[2]),
                "measurement_shannon_entropy": entropy,
                "trace_error": trace_error,
                "hermitian_error": hermitian_error,
                "min_eigenvalue": eig_min,
                "instrument_completeness_error": completeness_error,
            })
            current_day = int(day)

    if target_count == 0 or not daily:
        raise ContractError("experiment produced no scored contiguous samples")
    cross_gap_state_updates = count_cross_gap_state_updates(records)
    if cross_gap_state_updates != 0:
        raise ContractError("cross-gap return reached a state update")
    if (max_trace_error >= 1e-10 or max_hermitian_error >= 1e-10
            or min_eigenvalue < -1e-10 or max_instrument_completeness_error >= 1e-12):
        raise ContractError("density-matrix or instrument physicality check failed")

    out_dir.mkdir(parents=True, exist_ok=True)
    daily_path = out_dir / "quantum_lindblad_daily.parquet"
    minute_path = out_dir / "quantum_lindblad_minutes.parquet"
    summary_path = out_dir / "quantum_lindblad_summary.json"
    pd.DataFrame(daily).to_parquet(daily_path, index=False, compression="zstd")
    pd.DataFrame(records).to_parquet(minute_path, index=False, compression="zstd")

    class_prior = target_counts / target_count
    best_constant = int(np.argmax(target_counts))
    uniform = np.full(3, 1.0 / 3.0)
    diagnostic_passes_causal_baselines = bool(
        brier_sum / target_count < causal_prior_brier / target_count
        and brier_sum / target_count < last_magnitude_brier / target_count
        and top1_hits / target_count > causal_prior_hits / target_count
        and top1_hits / target_count > last_magnitude_hits / target_count)
    source_hashes = {
        **{f"data/canonical/{pair}.parquet": sha256_file(CANONICAL_DIR / f"{pair}.parquet")
           for pair in PAIRS},
        "data/derived/coupling_estimates.parquet": sha256_file(
            DERIVED_DIR / "coupling_estimates.parquet"),
        "engine/quantum/quantum_lindblad.py": sha256_file(Path(__file__).resolve()),
    }
    summary = {
        "version": VERSION,
        "interpretation": (
            "data-conditioned quantum-inspired likelihood filter using valid open-quantum "
            "system mathematics; not evidence of a physical quantum FX market"),
        "pair_scope": list(PAIRS),
        "pair_indices": list(PAIR_INDICES),
        "span": [utc_timestamp(int(times[first])).isoformat(),
                 utc_timestamp(int(times[end])).isoformat()],
        "configuration": {
            "dt_s": DT_S,
            "ewma_half_life_steps": HALFLIFE_STEPS,
            "phase_gain": PHASE_GAIN,
            "coupling_gain": COUPLING_GAIN,
            "measurement_gain": MEASUREMENT_GAIN,
            "dephasing_per_step": DEPHASING_PER_STEP,
            "max_standardized_return": MAX_STANDARDIZED_RETURN,
            "max_coupling_age_s": max_coupling_age_s,
        },
        "source_hashes": source_hashes,
        "rows_processed": int(end - first),
        "scored_continuous_60s_steps": continuous_steps,
        "gap_density_resets": gap_resets,
        "leading_gap_skips": leading_gap_skips,
        "cross_gap_state_updates": cross_gap_state_updates,
        "stale_coupling_skips": stale_coupling_skips,
        "stale_coupling_resets": stale_coupling_resets,
        "max_coupling_age_seen_s": max_coupling_age_seen_s,
        "physics_checks": {
            "max_trace_error": max_trace_error,
            "max_hermitian_error": max_hermitian_error,
            "min_eigenvalue": min_eigenvalue,
            "max_instrument_completeness_error": max_instrument_completeness_error,
        },
        "next_bar_experimental_metric": {
            "target": "which pair has the largest absolute next valid 60-second log return",
            "samples": target_count,
            "top1_accuracy": top1_hits / target_count,
            "brier_score": brier_sum / target_count,
            "baselines": {
                "uniform": {"top1_accuracy": 1.0 / 3.0, "brier_score": brier(uniform, 0)},
                "causal_laplace_class_prior": {
                    "top1_accuracy": causal_prior_hits / target_count,
                    "brier_score": causal_prior_brier / target_count,
                },
                "causal_last_bar_largest_magnitude": {
                    "top1_accuracy": last_magnitude_hits / target_count,
                    "brier_score": last_magnitude_brier / target_count,
                },
                "in_sample_best_constant": {
                    "pair": PAIRS[best_constant],
                    "class_frequency": float(class_prior[best_constant]),
                    "top1_accuracy": float(class_prior[best_constant]),
                    "brier_score": 2.0 * (1.0 - float(class_prior[best_constant])),
                },
                "empirical_class_prior": {
                    "probabilities": {pair: float(value) for pair, value in zip(PAIRS, class_prior, strict=True)},
                    "brier_score": 1.0 - float(np.sum(class_prior ** 2)),
                },
            },
            "passes_causal_baselines": diagnostic_passes_causal_baselines,
            "promotion_status": (
                "rejected: first-segment diagnostic is not a frozen chronological OOS test; "
                "it cannot promote the quantum branch regardless of this score"),
            "claim": "diagnostic only; no trading, causation, quantum advantage, or physical-quantum claim",
        },
        "outputs": [
            str(daily_path.relative_to(ROOT)).replace("\\", "/"),
            str(minute_path.relative_to(ROOT)).replace("\\", "/"),
        ],
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--out-dir", type=Path, default=DERIVED_DIR)
    parser.add_argument("--max-coupling-age-s", type=float, default=MAX_COUPLING_AGE_S)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        result = self_check()
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    if args.max_steps < 2:
        raise SystemExit("--max-steps must be at least 2")
    try:
        run(args.max_steps, args.out_dir.resolve(), args.max_coupling_age_s)
        return 0
    except (ContractError, SharedContractError, ValueError, np.linalg.LinAlgError, OSError) as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

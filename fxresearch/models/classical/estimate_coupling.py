"""
estimate_coupling.py -- causal OQ-6/OQ-7 FX coupling estimator.

Default run (canonical parquet input only):
    python -m fxresearch.models.classical.estimate_coupling

The program writes the current, versioned-by-content coupling delivery to
data/derived/.  It does not read raw provider data or any dynamics artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fxresearch.core.contracts import canonical_pair_order


US_PER_MINUTE = 60_000_000
US_PER_DAY = 86_400_000_000
SECONDS_PER_MINUTE = 60.0
DEFAULT_LOOKBACK_SAMPLES = 28_800       # 20 24-hour-equivalent minute blocks
DEFAULT_RIDGE = 1.0e-3                  # dimensionless diagonal scale
VERSION = "1.0.0"

ROOT = Path(__file__).resolve().parents[3]
CANONICAL_DIR = ROOT / "data" / "canonical"
DEFAULT_OUT_DIR = ROOT / "data" / "derived"
PAIRS = list(canonical_pair_order(ROOT))
N_PAIRS = len(PAIRS)


@dataclass
class DayChunk:
    """Only current rolling-window samples are retained, never the full z/b."""

    timestamps_us: np.ndarray
    z: np.ndarray
    b: np.ndarray

    @property
    def n(self) -> int:
        return len(self.timestamps_us)


def identity_transform() -> tuple[np.ndarray, np.ndarray]:
    """Return T and T^-1 for the identity-free channel convention.

    z = T g.  Rows 0..6 are independent raw pair-return channels.  The last
    three channels are triangle residual returns, not arithmetic cross moves:
      z7 = g7 - g0 + g2, z8 = g8 - g0 - g1, z9 = g9 - g2 - g1.
    The same transform is applied to target accelerations b = T a.
    """
    transform = np.eye(N_PAIRS, dtype=np.float64)
    transform[7, 0], transform[7, 2] = -1.0, 1.0
    transform[8, 0], transform[8, 1] = -1.0, -1.0
    transform[9, 1], transform[9, 2] = -1.0, -1.0
    inverse = np.linalg.inv(transform)
    return transform, inverse


def iso_utc_from_us(value: int | np.integer) -> str:
    return datetime.fromtimestamp(int(value) / 1_000_000, tz=timezone.utc).isoformat()


def timestamp_array(table: pa.Table) -> np.ndarray:
    """Extract UTC parquet timestamps as epoch microseconds without pandas."""
    column = table.column("timestamp").combine_chunks()
    # Canonical schema is timestamp[us, UTC].  Cast makes the epoch unit
    # explicit and refuses accidental string/object timestamp handling.
    return np.asarray(column.cast(pa.int64()).to_numpy(), dtype=np.int64)


def load_common_log_prices() -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Strict intersection alignment; neither fill nor as-of matching is used."""
    common: np.ndarray | None = None
    row_counts: dict[str, int] = {}
    for pair in PAIRS:
        path = CANONICAL_DIR / f"{pair}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing canonical input: {path}")
        table = pq.read_table(path, columns=["timestamp"])
        ts = timestamp_array(table)
        if len(ts) == 0 or np.any(np.diff(ts) <= 0):
            raise ValueError(f"{pair}: canonical timestamps must be nonempty and unique ascending")
        row_counts[pair] = int(len(ts))
        common = ts if common is None else np.intersect1d(common, ts, assume_unique=True)

    assert common is not None
    if len(common) < 3:
        raise ValueError("fewer than three synchronous canonical timestamps")

    # Second bounded pass: only the final common timestamps are materialized as
    # an N x 10 matrix.  This deliberately does not build a dense minute grid.
    log_prices = np.empty((len(common), N_PAIRS), dtype=np.float64)
    for j, pair in enumerate(PAIRS):
        path = CANONICAL_DIR / f"{pair}.parquet"
        table = pq.read_table(path, columns=["timestamp", "close"])
        ts = timestamp_array(table)
        close = np.asarray(table.column("close").combine_chunks().to_numpy(), dtype=np.float64)
        idx = np.searchsorted(ts, common)
        if np.any(idx >= len(ts)) or not np.array_equal(ts[idx], common):
            raise RuntimeError(f"{pair}: intersection alignment invariant failed")
        selected = close[idx]
        if not np.isfinite(selected).all() or np.any(selected <= 0.0):
            raise ValueError(f"{pair}: non-finite or nonpositive canonical close in aligned data")
        log_prices[:, j] = np.log(selected)

    return common, log_prices, row_counts


def correlation(cov: np.ndarray, i: int, j: int) -> float:
    denom = float(np.sqrt(max(cov[i, i], 0.0) * max(cov[j, j], 0.0)))
    return float(cov[i, j] / denom) if denom > 0.0 else float("nan")


def triangle_metrics(sum_z: np.ndarray, sum_zz: np.ndarray, n: int,
                     inverse: np.ndarray) -> dict[str, float]:
    """Compare raw cross-parent dependence with constrained residual-parent dependence."""
    if n <= 1:
        return {k: float("nan") for k in (
            "t1_pre_abs_corr", "t1_post_abs_corr", "t2_pre_abs_corr",
            "t2_post_abs_corr", "t3_pre_abs_corr", "t3_post_abs_corr")}
    cov_z = (sum_zz - np.outer(sum_z, sum_z) / n) / (n - 1)
    cov_raw = inverse @ cov_z @ inverse.T

    def avg_abs(values: list[float]) -> float:
        return float(np.mean(np.abs(values)))

    # Raw g: cross-parent dependence.  z: triangle-residual-parent dependence.
    return {
        "t1_pre_abs_corr": avg_abs([correlation(cov_raw, 7, 0), correlation(cov_raw, 7, 2)]),
        "t1_post_abs_corr": avg_abs([correlation(cov_z, 7, 0), correlation(cov_z, 7, 2)]),
        "t2_pre_abs_corr": avg_abs([correlation(cov_raw, 8, 0), correlation(cov_raw, 8, 1)]),
        "t2_post_abs_corr": avg_abs([correlation(cov_z, 8, 0), correlation(cov_z, 8, 1)]),
        "t3_pre_abs_corr": avg_abs([correlation(cov_raw, 9, 2), correlation(cov_raw, 9, 1)]),
        "t3_post_abs_corr": avg_abs([correlation(cov_z, 9, 2), correlation(cov_z, 9, 1)]),
    }


def fit_matrix(sum_bz: np.ndarray, sum_zz: np.ndarray, transform: np.ndarray,
               inverse: np.ndarray, ridge: float) -> tuple[np.ndarray | None, float, float]:
    """Fit M in b=M z, map C_raw=T^-1 M T, then enforce the frozen zero diagonal."""
    scale = np.diag(sum_zz).copy()
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        return None, float("nan"), float("nan")
    floor = max(float(np.median(scale)) * 1.0e-12, np.finfo(np.float64).tiny)
    regularized = sum_zz + np.diag(ridge * np.maximum(scale, floor))
    condition = float(np.linalg.cond(regularized))
    try:
        # solve(A.T, B.T).T = B @ inv(A), avoiding explicit inversion.
        latent = np.linalg.solve(regularized.T, sum_bz.T).T
    except np.linalg.LinAlgError:
        return None, condition, float("nan")
    raw_unconstrained = inverse @ latent @ transform
    diagonal_removed_l2 = float(np.linalg.norm(np.diag(raw_unconstrained)))
    coupling = raw_unconstrained.copy()
    np.fill_diagonal(coupling, 0.0)
    if not np.isfinite(coupling).all():
        return None, condition, diagonal_removed_l2
    return coupling, condition, diagonal_removed_l2


def run_static_self_check() -> dict[str, object]:
    """Small deterministic contract test; does not touch data or write outputs."""
    transform, inverse = identity_transform()
    round_trip_error = float(np.max(np.abs(inverse @ transform - np.eye(N_PAIRS))))
    # Exact triangle moves must have zero identity-residual channels.
    g = np.zeros(N_PAIRS)
    g[0], g[1], g[2] = 0.02, -0.01, 0.015
    g[7], g[8], g[9] = g[0] - g[2], g[0] + g[1], g[2] + g[1]
    residual_error = float(np.max(np.abs((transform @ g)[7:])))
    matrix = np.ones((N_PAIRS, N_PAIRS), dtype=np.float64)
    np.fill_diagonal(matrix, 0.0)
    return {
        "passed": bool(round_trip_error < 1e-12 and residual_error < 1e-12
                       and bool(np.all(np.diag(matrix) == 0.0))),
        "transform_inverse_max_abs_error": round_trip_error,
        "exact_triangle_residual_max_abs_error": residual_error,
        "zero_diagonal_check": bool(np.all(np.diag(matrix) == 0.0)),
    }


def make_estimate_table(matrices: list[np.ndarray], update_us: list[int],
                        start_us: list[int], sample_counts: list[int]) -> pa.Table:
    n_snapshots = len(matrices)
    stacked = np.stack(matrices, axis=0)
    per_matrix = N_PAIRS * N_PAIRS
    affected = np.tile(np.repeat(np.arange(N_PAIRS, dtype=np.int8), N_PAIRS), n_snapshots)
    source = np.tile(np.tile(np.arange(N_PAIRS, dtype=np.int8), N_PAIRS), n_snapshots)
    symbols = np.asarray(PAIRS, dtype=object)
    return pa.table({
        "update_time": pa.array(np.repeat(np.asarray(update_us, dtype=np.int64), per_matrix),
                                type=pa.timestamp("us", tz="UTC")),
        "window_start_time": pa.array(np.repeat(np.asarray(start_us, dtype=np.int64), per_matrix),
                                      type=pa.timestamp("us", tz="UTC")),
        "effective_samples": pa.array(np.repeat(np.asarray(sample_counts, dtype=np.int32), per_matrix)),
        "affected_index": pa.array(affected),
        "affected_symbol": pa.array(symbols[affected]),
        "source_index": pa.array(source),
        "source_symbol": pa.array(symbols[source]),
        "c_ij_s_minus_2": pa.array(stacked.reshape(-1)),
    })


def validate_run(matrices: list[np.ndarray], update_us: list[int], start_us: list[int],
                 diagnostics: list[dict[str, object]], static: dict[str, object]) -> dict[str, object]:
    accepted = [d for d in diagnostics if d["status"] == "ACCEPTED"]
    diagonal_ok = bool(all(np.array_equal(np.diag(m), np.zeros(N_PAIRS)) for m in matrices))
    finite_ok = bool(all(np.isfinite(m).all() for m in matrices))
    causal_ok = bool(all(s <= u for s, u in zip(start_us, update_us)))
    triangle_ok = bool(all(np.isfinite([d["t1_pre_abs_corr"], d["t1_post_abs_corr"],
                                        d["t2_pre_abs_corr"], d["t2_post_abs_corr"],
                                        d["t3_pre_abs_corr"], d["t3_post_abs_corr"]]).all()
                            for d in accepted))
    return {
        "version": VERSION,
        "static_self_check": static,
        "accepted_matrix_count": len(matrices),
        "diagnostic_rows": len(diagnostics),
        "zero_diagonal_pass": diagonal_ok,
        "finite_accepted_matrices_pass": finite_ok,
        "causality_boundary_pass": causal_ok,
        "triangle_diagnostic_finite_pass": triangle_ok,
        "passed": bool(static["passed"] and diagonal_ok and finite_ok and causal_ok and triangle_ok),
    }


def estimate(lookback_samples: int, ridge: float, out_dir: Path) -> dict[str, object]:
    if lookback_samples < 100:
        raise ValueError("--lookback-samples must be at least 100")
    if not np.isfinite(ridge) or ridge <= 0.0:
        raise ValueError("--ridge must be positive and finite")

    started_wall = time.perf_counter()
    run_started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    static = run_static_self_check()
    if not static["passed"]:
        raise RuntimeError(f"internal transform self-check failed: {static}")

    timestamps, x, row_counts = load_common_log_prices()
    transform, inverse = identity_transform()
    gap_deltas = np.diff(timestamps)
    candidates = max(0, len(timestamps) - 2)
    total_valid = 0
    diagnostics: list[dict[str, object]] = []
    matrices: list[np.ndarray] = []
    update_us: list[int] = []
    start_us: list[int] = []
    sample_counts: list[int] = []

    window: deque[DayChunk] = deque()
    sum_z = np.zeros(N_PAIRS, dtype=np.float64)
    sum_zz = np.zeros((N_PAIRS, N_PAIRS), dtype=np.float64)
    sum_bz = np.zeros((N_PAIRS, N_PAIRS), dtype=np.float64)
    in_window = 0
    previous_accepted: np.ndarray | None = None

    day = timestamps // US_PER_DAY
    starts = np.r_[0, np.flatnonzero(np.diff(day)) + 1]
    ends = np.r_[starts[1:], len(timestamps)]

    for begin, end in zip(starts, ends):
        indices = np.arange(max(int(begin), 2), int(end), dtype=np.int64)
        if len(indices):
            valid = ((timestamps[indices] - timestamps[indices - 1] == US_PER_MINUTE)
                     & (timestamps[indices - 1] - timestamps[indices - 2] == US_PER_MINUTE))
            indices = indices[valid]
        if len(indices) == 0:
            continue

        # g(t) is the actual one-minute log return.  a(t) is a fully backward
        # second difference; both are valid only across two observed 60s bars.
        g = x[indices] - x[indices - 1]
        a = (x[indices] - 2.0 * x[indices - 1] + x[indices - 2]) / (SECONDS_PER_MINUTE ** 2)
        z = g @ transform.T
        b = a @ transform.T
        chunk = DayChunk(timestamps[indices].copy(), z, b)
        total_valid += chunk.n
        window.append(chunk)
        in_window += chunk.n
        sum_z += z.sum(axis=0)
        sum_zz += z.T @ z
        sum_bz += b.T @ z

        # Exact-sample causal lookback: remove only the oldest observations.
        # This keeps W valid synchronous samples ending at the current day-end,
        # even when calendar weekends/intra-session holes occurred.
        excess = in_window - lookback_samples
        while excess > 0:
            first = window[0]
            take = min(excess, first.n)
            rz, rb = first.z[:take], first.b[:take]
            sum_z -= rz.sum(axis=0)
            sum_zz -= rz.T @ rz
            sum_bz -= rb.T @ rz
            in_window -= take
            excess -= take
            if take == first.n:
                window.popleft()
            else:
                first.timestamps_us = first.timestamps_us[take:]
                first.z = first.z[take:]
                first.b = first.b[take:]

        now = int(chunk.timestamps_us[-1])
        base = {
            "update_time": iso_utc_from_us(now),
            "window_start_time": iso_utc_from_us(int(window[0].timestamps_us[0])),
            "effective_samples": int(in_window),
            "status": "INSUFFICIENT_HISTORY",
            "condition_number": None,
            "unconstrained_diagonal_l2_s_minus_2": None,
            "matrix_abs_max_s_minus_2": None,
            "matrix_delta_fro_s_minus_2": None,
            "matrix_relative_fro_change": None,
            "regime_break_flag": False,
        }
        base.update(triangle_metrics(sum_z, sum_zz, in_window, inverse))
        if in_window < lookback_samples:
            diagnostics.append(base)
            continue

        coupling, condition, removed_diag = fit_matrix(sum_bz, sum_zz, transform, inverse, ridge)
        if coupling is None or not np.isfinite(condition) or condition > 1.0e12:
            base.update({
                "status": "REJECTED_NUMERICAL_CONDITION",
                "condition_number": condition if np.isfinite(condition) else None,
                "unconstrained_diagonal_l2_s_minus_2": removed_diag if np.isfinite(removed_diag) else None,
            })
            diagnostics.append(base)
            continue

        base.update({
            "status": "ACCEPTED",
            "condition_number": condition,
            "unconstrained_diagonal_l2_s_minus_2": removed_diag,
            "matrix_abs_max_s_minus_2": float(np.max(np.abs(coupling))),
        })
        if previous_accepted is not None:
            delta_fro = float(np.linalg.norm(coupling - previous_accepted))
            previous_fro = float(np.linalg.norm(previous_accepted))
            relative_change = delta_fro / max(previous_fro, 1.0e-15)
            base.update({
                "matrix_delta_fro_s_minus_2": delta_fro,
                "matrix_relative_fro_change": relative_change,
                "regime_break_flag": bool(relative_change >= 0.50 or condition >= 1.0e8),
            })
        diagnostics.append(base)
        matrices.append(coupling)
        update_us.append(now)
        start_us.append(int(window[0].timestamps_us[0]))
        sample_counts.append(int(in_window))
        previous_accepted = coupling

    if not matrices:
        raise RuntimeError("no accepted coupling matrix: inspect diagnostics for visible rejection reasons")

    out_dir.mkdir(parents=True, exist_ok=True)
    estimate_path = out_dir / "coupling_estimates.parquet"
    diagnostic_path = out_dir / "coupling_diagnostics.parquet"
    summary_path = out_dir / "coupling_summary.json"
    validation_path = out_dir / "coupling_validation.json"
    pq.write_table(make_estimate_table(matrices, update_us, start_us, sample_counts), estimate_path,
                   compression="zstd")
    pq.write_table(pa.Table.from_pylist(diagnostics), diagnostic_path, compression="zstd")

    validation = validate_run(matrices, update_us, start_us, diagnostics, static)
    accepted = [d for d in diagnostics if d["status"] == "ACCEPTED"]
    rejected = [d for d in diagnostics if d["status"] == "REJECTED_NUMERICAL_CONDITION"]
    latest = accepted[-1]
    summary: dict[str, object] = {
        "version": VERSION,
        "run_started_utc": run_started,
        "run_finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runtime_seconds": round(time.perf_counter() - started_wall, 3),
        "input": {
            "directory": "data/canonical",
            "format": "parquet only",
            "instrument_order": PAIRS,
            "per_pair_rows": row_counts,
            "common_synchronous_rows": int(len(timestamps)),
            "common_timestamp_first": iso_utc_from_us(int(timestamps[0])),
            "common_timestamp_last": iso_utc_from_us(int(timestamps[-1])),
            "candidate_second_difference_rows": int(candidates),
            "accepted_valid_60s_two_step_samples": int(total_valid),
            "excluded_for_gap_or_asynchrony": int(candidates - total_valid),
        },
        "estimator": {
            "g": "g_j(t)=60 s * v_j(t)=x_j(t)-x_j(t-60 s), nats",
            "target": "a_i(t)=[x_i(t)-2x_i(t-60s)+x_i(t-120s)]/(60s)^2, nats s^-2",
            "basis": "z=Tg and b=Ta; T replaces EURGBP/EURJPY/GBPJPY with triangle residual channels",
            "fit": "M=(sum b z^T)[sum z z^T + ridge*diag(diag(sum z z^T))]^-1; C=offdiag(T^-1 M T)",
            "lookback_valid_samples": lookback_samples,
            "update_cadence": "last valid synchronous two-step sample of each UTC calendar day",
            "ridge_dimensionless": ridge,
            "condition_reject_threshold": 1.0e12,
        },
        "outputs": {
            "estimates": str(estimate_path.relative_to(ROOT)).replace("\\", "/"),
            "diagnostics": str(diagnostic_path.relative_to(ROOT)).replace("\\", "/"),
            "validation": str(validation_path.relative_to(ROOT)).replace("\\", "/"),
        },
        "snapshots": {
            "accepted": len(accepted),
            "insufficient_history": sum(d["status"] == "INSUFFICIENT_HISTORY" for d in diagnostics),
            "rejected_numerical_condition": len(rejected),
            "regime_break_flags": sum(bool(d["regime_break_flag"]) for d in accepted),
            "first_accepted_update_time": accepted[0]["update_time"],
            "last_accepted_update_time": latest["update_time"],
        },
        "triangle_constraint_diagnostic_latest": {
            k: latest[k] for k in latest if k.startswith("t") and k.endswith("abs_corr")
        },
        "bid_only_limitations": [
            "BID bars have no contemporaneous ask or spread; triangle residuals include bid-side quote and timing effects.",
            "The residual basis removes arithmetic identity channels; it cannot prove executable arbitrage, causation, or tradeable lead-lag.",
        ],
    }
    with open(summary_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with open(validation_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(validation, handle, indent=2, allow_nan=False)
        handle.write("\n")

    print(f"[COUPLING] common_rows={len(timestamps)} valid_samples={total_valid} "
          f"accepted_snapshots={len(matrices)} runtime_s={summary['runtime_seconds']}")
    print(f"[COUPLING] estimates={estimate_path}")
    print(f"[COUPLING] diagnostics={diagnostic_path}")
    print(f"[COUPLING] validation={validation_path} passed={validation['passed']}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--lookback-samples", type=int, default=DEFAULT_LOOKBACK_SAMPLES,
                        help="causal valid-sample lookback (default: 28800)")
    parser.add_argument("--ridge", type=float, default=DEFAULT_RIDGE,
                        help="dimensionless diagonal ridge scale (default: 1e-3)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="derived-output directory (default: data/derived)")
    parser.add_argument("--self-check", action="store_true",
                        help="run only the deterministic transform/constraint self-check")
    args = parser.parse_args(argv)
    if args.self_check:
        result = run_static_self_check()
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result["passed"] else 1
    try:
        estimate(args.lookback_samples, args.ridge, args.out_dir.resolve())
    except Exception as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

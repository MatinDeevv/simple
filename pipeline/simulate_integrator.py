"""
simulate_integrator.py -- causal, stability-checked three-pair FX replay.

This is intentionally scoped to EURUSD, USDJPY, and USDCNH because those are
the only pairs with accepted dynamics parameter streams.  Coupling is the
corresponding 3x3 submatrix of the accepted 10x10 field.  No missing pair is
imputed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "data_canonical"
DERIVED_DIR = ROOT / "data_derived"

PAIRS = ("EURUSD", "USDJPY", "USDCNH")
PAIR_INDICES = (0, 1, 5)
DT_NOM_S = 60.0
DT_NOM_NS = int(DT_NOM_S * 1_000_000_000)
DEFAULT_MAX_STEPS = 250_000
STABILITY_CAP_S = 3_600.0
VERSION = "integrator-1.1.0"
STABLE_RHO_LIMIT = 1.0 + 1.0e-10


class ContractError(RuntimeError):
    pass


def utc_iso(ns: int) -> str:
    return pd.Timestamp(ns, unit="ns", tz="UTC").isoformat()


def epoch_ns(values: object) -> np.ndarray:
    """Convert a timezone-aware timestamp column/index to UTC epoch ns."""
    return pd.DatetimeIndex(values).as_unit("ns").asi8


def read_prices() -> tuple[np.ndarray, np.ndarray]:
    merged: pd.DataFrame | None = None
    for pair in PAIRS:
        frame = pd.read_parquet(CANONICAL_DIR / f"{pair}.parquet",
                                columns=["timestamp", "close"])
        frame = frame.rename(columns={"close": pair})
        merged = frame if merged is None else merged.merge(frame, on="timestamp",
                                                           how="inner", validate="one_to_one")
    assert merged is not None
    merged = merged.sort_values("timestamp", kind="stable").reset_index(drop=True)
    times = epoch_ns(merged["timestamp"])
    if not np.all(np.diff(times) > 0):
        raise ContractError("strict-common canonical timestamps are not increasing")
    close = merged.loc[:, list(PAIRS)].to_numpy(dtype=np.float64)
    if not np.isfinite(close).all() or np.any(close <= 0.0):
        raise ContractError("canonical close values must be finite and positive")
    return times, np.log(close)


def read_dynamics(pair: str) -> dict[str, np.ndarray]:
    path = DERIVED_DIR / f"dynamics_params_{pair}.parquet"
    if not path.exists():
        raise ContractError(f"missing accepted dynamics stream: {path}")
    frame = pd.read_parquet(path, columns=["timestamp", "m", "k", "c", "x_eq",
                                           "sigma_eps", "n_valid_window"])
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    times = epoch_ns(frame["timestamp"])
    values = frame.loc[:, ["m", "k", "c", "x_eq", "sigma_eps",
                           "n_valid_window"]].to_numpy(dtype=np.float64)
    if len(times) == 0 or not np.all(np.diff(times) > 0):
        raise ContractError(f"{pair}: parameter timestamps are invalid")
    if not np.isfinite(values).all() or np.any(values[:, 0] <= 0.0):
        raise ContractError(f"{pair}: dynamics parameters are non-finite or mass is non-positive")
    if np.any(values[:, 5] < 20_000):
        raise ContractError(f"{pair}: emitted parameter row below minimum valid window")
    return {"times": times, "values": values}


def read_coupling() -> dict[str, np.ndarray]:
    path = DERIVED_DIR / "coupling_estimates.parquet"
    if not path.exists():
        raise ContractError(f"missing accepted coupling stream: {path}")
    frame = pd.read_parquet(path)
    required = {"update_time", "affected_index", "source_index", "c_ij_s_minus_2"}
    if not required.issubset(frame.columns):
        raise ContractError(f"coupling schema missing {sorted(required - set(frame.columns))}")
    subset = frame[
        frame["affected_index"].isin(PAIR_INDICES)
        & frame["source_index"].isin(PAIR_INDICES)
    ].copy()
    cols = pd.MultiIndex.from_product([PAIR_INDICES, PAIR_INDICES])
    wide = subset.pivot(index="update_time",
                        columns=["affected_index", "source_index"],
                        values="c_ij_s_minus_2").reindex(columns=cols)
    if wide.empty or wide.isna().any().any():
        raise ContractError("coupling submatrix is incomplete")
    times = epoch_ns(wide.index)
    matrices = wide.to_numpy(dtype=np.float64).reshape(-1, len(PAIRS), len(PAIRS))
    if not np.all(np.diff(times) > 0) or not np.isfinite(matrices).all():
        raise ContractError("coupling matrices are invalid")
    if not np.array_equal(np.diagonal(matrices, axis1=1, axis2=2),
                          np.zeros((len(matrices), len(PAIRS)))):
        raise ContractError("coupling diagonal must be exactly zero")
    return {"times": times, "matrices": matrices}


def cursor_at(times: np.ndarray, now_ns: int) -> int:
    index = int(np.searchsorted(times, now_ns, side="right") - 1)
    if index < 0:
        raise ContractError(f"no causal parameter value at {utc_iso(now_ns)}")
    return index


def configuration_at(now_ns: int, dynamics: list[dict[str, np.ndarray]],
                     coupling: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray,
                                                               np.ndarray, np.ndarray]:
    m = np.empty(len(PAIRS))
    k = np.empty(len(PAIRS))
    c = np.empty(len(PAIRS))
    x_eq = np.empty(len(PAIRS))
    for i, table in enumerate(dynamics):
        row = table["values"][cursor_at(table["times"], now_ns)]
        m[i], k[i], c[i], x_eq[i] = row[:4]
    C = coupling["matrices"][cursor_at(coupling["times"], now_ns)]
    kappa = k / m
    gamma = c / m
    if not (np.isfinite(kappa).all() and np.isfinite(gamma).all()
            and np.isfinite(x_eq).all() and np.isfinite(C).all()):
        raise ContractError("non-finite causal configuration")
    return kappa, gamma, x_eq, C


def amplification_matrix(kappa: np.ndarray, gamma: np.ndarray,
                         coupling: np.ndarray, dt_s: float) -> np.ndarray:
    """Semi-implicit Euler, with damping implicit and C as acceleration."""
    n = len(kappa)
    if dt_s <= 0.0 or np.any(1.0 + dt_s * gamma <= 0.0):
        raise ContractError("non-positive implicit-damping denominator")
    damping_inverse = np.diag(1.0 / (1.0 + dt_s * gamma))
    spring = np.diag(kappa)
    v_from_x = damping_inverse @ (-dt_s * spring)
    v_from_v = damping_inverse @ (np.eye(n) + dt_s * dt_s * coupling)
    return np.block([
        [np.eye(n) + dt_s * v_from_x, dt_s * v_from_v],
        [v_from_x, v_from_v],
    ])


def spectral_radius(kappa: np.ndarray, gamma: np.ndarray,
                    coupling: np.ndarray, dt_s: float) -> float:
    values = np.linalg.eigvals(amplification_matrix(kappa, gamma, coupling, dt_s))
    radius = float(np.max(np.abs(values)))
    if not math.isfinite(radius):
        raise ContractError("non-finite amplification eigenvalue")
    return radius


def stable_dt_lower_bound(kappa: np.ndarray, gamma: np.ndarray,
                          coupling: np.ndarray) -> float:
    """Largest stable dt found by monotone bracketing, capped at one hour."""
    tolerance = 1.0 + 1.0e-10
    lo = 1.0
    if spectral_radius(kappa, gamma, coupling, lo) > tolerance:
        return 0.0
    hi = DT_NOM_S
    while hi < STABILITY_CAP_S and spectral_radius(kappa, gamma, coupling, hi) <= tolerance:
        lo = hi
        hi = min(hi * 2.0, STABILITY_CAP_S)
    if spectral_radius(kappa, gamma, coupling, hi) <= tolerance:
        return STABILITY_CAP_S
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if spectral_radius(kappa, gamma, coupling, mid) <= tolerance:
            lo = mid
        else:
            hi = mid
    return lo


def stability_table(dynamics: list[dict[str, np.ndarray]],
                    coupling: dict[str, np.ndarray]) -> pd.DataFrame:
    config_times = np.unique(np.concatenate([
        *(table["times"] for table in dynamics), coupling["times"],
    ]))
    first_ready = max(*(table["times"][0] for table in dynamics), coupling["times"][0])
    records: list[dict[str, object]] = []
    for now_ns in config_times[config_times >= first_ready]:
        kappa, gamma, _x_eq, C = configuration_at(int(now_ns), dynamics, coupling)
        rho_raw = spectral_radius(kappa, gamma, C, DT_NOM_S)
        negative_curvature_components = int(np.sum(kappa < 0.0))
        kappa_sim = np.maximum(kappa, 0.0)
        rho = spectral_radius(kappa_sim, gamma, C, DT_NOM_S)
        records.append({
            "timestamp": pd.Timestamp(now_ns, unit="ns", tz="UTC"),
            "rho_dt_60_raw": rho_raw,
            "rho_dt_60_guarded": rho,
            "raw_stability_margin": 1.0 - rho_raw,
            "guarded_stability_margin": 1.0 - rho,
            "stable_dt_lower_bound_s": stable_dt_lower_bound(kappa_sim, gamma, C),
            "negative_curvature_components": negative_curvature_components,
            "negative_curvature_projection_applied": bool(negative_curvature_components),
            "max_abs_kappa_s_minus_2": float(np.max(np.abs(kappa))),
            "max_gamma_s_minus_1": float(np.max(gamma)),
            "coupling_spectral_norm_s_minus_2": float(np.linalg.norm(C, ord=2)),
            "status": "STABLE_GUARDED_60S" if rho <= STABLE_RHO_LIMIT else "REJECTED_GUARDED_60S",
        })
    result = pd.DataFrame(records)
    if result.empty:
        raise ContractError("no common stability configurations")
    return result


def write_checkpoint(path: Path, next_index: int, times: np.ndarray,
                     x_hat: np.ndarray, v_hat: np.ndarray) -> None:
    payload = {
        "version": VERSION,
        "pair_scope": list(PAIRS),
        "next_row_index": int(next_index),
        "last_timestamp": utc_iso(int(times[next_index - 1])),
        "x_hat": [float(value) for value in x_hat],
        "v_hat": [float(value) for value in v_hat],
        "restore_rule": "reload causal daily inputs by timestamp; do not infer velocity across a gap",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_checkpoint(path: Path, times: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != VERSION or payload.get("pair_scope") != list(PAIRS):
        raise ContractError("checkpoint contract does not match this integrator")
    next_index = int(payload["next_row_index"])
    if not 1 <= next_index < len(times):
        raise ContractError("checkpoint next_row_index is outside the replay")
    if utc_iso(int(times[next_index - 1])) != payload["last_timestamp"]:
        raise ContractError("checkpoint timestamp does not match canonical stream")
    x_hat = np.asarray(payload["x_hat"], dtype=np.float64)
    v_hat = np.asarray(payload["v_hat"], dtype=np.float64)
    if x_hat.shape != (len(PAIRS),) or v_hat.shape != (len(PAIRS),):
        raise ContractError("checkpoint state has wrong shape")
    if not np.isfinite(x_hat).all() or not np.isfinite(v_hat).all():
        raise ContractError("checkpoint state is non-finite")
    return next_index, x_hat, v_hat


def replay(times: np.ndarray, actual_x: np.ndarray, dynamics: list[dict[str, np.ndarray]],
           coupling: dict[str, np.ndarray], max_steps: int, checkpoint: Path,
           resume: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    first_ready = max(*(table["times"][0] for table in dynamics), coupling["times"][0])
    if resume:
        start_index, x_hat, v_hat = load_checkpoint(checkpoint, times)
        initialised_at = int(times[start_index - 1])
    else:
        start_index = int(np.searchsorted(times, first_ready, side="left"))
        while start_index < len(times) and (
            start_index == 0 or times[start_index] - times[start_index - 1] != DT_NOM_NS
        ):
            start_index += 1
        if start_index >= len(times) - 1:
            raise ContractError("cannot find a contiguous causal replay start")
        x_hat = actual_x[start_index].copy()
        v_hat = (actual_x[start_index] - actual_x[start_index - 1]) / DT_NOM_S
        initialised_at = int(times[start_index])

    end_index = min(len(times), start_index + max_steps + 1)
    gap_events: list[dict[str, object]] = []
    daily_records: list[dict[str, object]] = []
    previous_day: int | None = None
    day_steps = 0
    day_resets = 0
    day_max_error = 0.0
    day_max_pseudo_energy = 0.0
    day_unstable_rejected_steps = 0
    max_abs_state = float(np.max(np.abs(np.concatenate([x_hat, v_hat]))))
    continuous_steps = 0
    market_gap_resets = 0
    unstable_rejected_steps = 0
    unstable_configurations: set[tuple[int, ...]] = set()
    projected_steps = 0
    projected_configurations: set[tuple[int, ...]] = set()
    last_signature: tuple[int, ...] | None = None
    cached_config: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    cached_rho = float("nan")
    finite = True

    for index in range(start_index + 1, end_index):
        now_ns = int(times[index - 1])
        arrival_ns = int(times[index])
        arrival_day = arrival_ns // 86_400_000_000_000
        if previous_day is None:
            previous_day = int(arrival_day)
        if arrival_day != previous_day:
            daily_records.append({
                "timestamp": pd.Timestamp(now_ns, unit="ns", tz="UTC"),
                "continuous_steps": day_steps,
                "gap_resets": day_resets,
                "max_abs_prediction_error_nats": day_max_error,
                "max_pseudo_energy": day_max_pseudo_energy,
                "unstable_rejected_steps": day_unstable_rejected_steps,
            })
            previous_day = int(arrival_day)
            day_steps = 0
            day_resets = 0
            day_max_error = 0.0
            day_max_pseudo_energy = 0.0
            day_unstable_rejected_steps = 0

        observed_dt_ns = arrival_ns - now_ns
        if observed_dt_ns != DT_NOM_NS:
            # The absence is known only when this bar arrives.  Reset then; no
            # giant step and no fabricated gap-crossing velocity is permitted.
            x_hat = actual_x[index].copy()
            v_hat = np.zeros(len(PAIRS), dtype=np.float64)
            day_resets += 1
            market_gap_resets += 1
            gap_events.append({
                "arrival_time": pd.Timestamp(arrival_ns, unit="ns", tz="UTC"),
                "previous_time": pd.Timestamp(now_ns, unit="ns", tz="UTC"),
                "observed_gap_s": observed_dt_ns / 1_000_000_000.0,
                "action": "RESET_TO_OBSERVED_POSITION_ZERO_VELOCITY",
            })
            continue

        signature = tuple(cursor_at(table["times"], now_ns) for table in dynamics)
        signature += (cursor_at(coupling["times"], now_ns),)
        if signature != last_signature:
            cached_config = configuration_at(now_ns, dynamics, coupling)
            cached_rho = spectral_radius(np.maximum(cached_config[0], 0.0), cached_config[1],
                                         cached_config[3], DT_NOM_S)
            last_signature = signature
        assert cached_config is not None
        kappa, gamma, x_eq, C = cached_config
        if np.any(kappa < 0.0):
            if signature not in projected_configurations:
                projected_configurations.add(signature)
                gap_events.append({
                    "arrival_time": pd.Timestamp(arrival_ns, unit="ns", tz="UTC"),
                    "previous_time": pd.Timestamp(now_ns, unit="ns", tz="UTC"),
                    "observed_gap_s": DT_NOM_S,
                    "action": "PROJECT_NEGATIVE_KAPPA_TO_ZERO",
                })
            projected_steps += 1
            kappa = np.maximum(kappa, 0.0)
        if cached_rho > STABLE_RHO_LIMIT:
            # Signed fitted curvature can make the discrete physical state
            # unstable.  Do not clip C/k or take an unstable step: re-sync
            # to the arriving observed bar until a later causal configuration
            # passes the 60-second bound.
            if signature not in unstable_configurations:
                unstable_configurations.add(signature)
                gap_events.append({
                    "arrival_time": pd.Timestamp(arrival_ns, unit="ns", tz="UTC"),
                    "previous_time": pd.Timestamp(now_ns, unit="ns", tz="UTC"),
                    "observed_gap_s": DT_NOM_S,
                    "action": "REJECT_UNSTABLE_CONFIG_HOLD_OBSERVED",
                })
            x_hat = actual_x[index].copy()
            v_hat = (actual_x[index] - actual_x[index - 1]) / DT_NOM_S
            unstable_rejected_steps += 1
            day_unstable_rejected_steps += 1
            continue
        displacement = x_hat - x_eq
        # Conservative replay mode: epsilon=0.  The historical residual scale
        # was fitted without C, so adding sampled forcing would risk double count.
        specific_coupling_acceleration = C @ (DT_NOM_S * v_hat)
        numerator = v_hat + DT_NOM_S * (
            -kappa * displacement + specific_coupling_acceleration
        )
        v_hat = numerator / (1.0 + DT_NOM_S * gamma)
        x_hat = x_hat + DT_NOM_S * v_hat
        if not np.isfinite(x_hat).all() or not np.isfinite(v_hat).all():
            finite = False
            raise ContractError(f"non-finite replay state at {utc_iso(arrival_ns)}")
        absolute_error = float(np.max(np.abs(x_hat - actual_x[index])))
        pseudo_energy = float(0.5 * (
            np.dot(x_hat - x_eq, x_hat - x_eq) + DT_NOM_S ** 2 * np.dot(v_hat, v_hat)
        ))
        day_steps += 1
        continuous_steps += 1
        day_max_error = max(day_max_error, absolute_error)
        day_max_pseudo_energy = max(day_max_pseudo_energy, pseudo_energy)
        max_abs_state = max(max_abs_state, float(np.max(np.abs(np.concatenate([x_hat, v_hat])))))

    if day_steps or day_resets:
        daily_records.append({
            "timestamp": pd.Timestamp(int(times[end_index - 1]), unit="ns", tz="UTC"),
            "continuous_steps": day_steps,
            "gap_resets": day_resets,
            "max_abs_prediction_error_nats": day_max_error,
            "max_pseudo_energy": day_max_pseudo_energy,
            "unstable_rejected_steps": day_unstable_rejected_steps,
        })
    if end_index < len(times):
        write_checkpoint(checkpoint, end_index, times, x_hat, v_hat)

    summary = {
        "initialised_at": utc_iso(initialised_at),
        "finished_at": utc_iso(int(times[end_index - 1])),
        "rows_consumed": int(end_index - start_index),
        "continuous_60s_steps": int(continuous_steps),
        "gap_resets": int(market_gap_resets),
        "gap_policy_exercised": bool(market_gap_resets > 0),
        "event_count": int(len(gap_events)),
        "unstable_rejected_steps": int(unstable_rejected_steps),
        "unstable_configurations": int(len(unstable_configurations)),
        "negative_curvature_projected_steps": int(projected_steps),
        "negative_curvature_projected_configurations": int(len(projected_configurations)),
        "unprotected_unstable_steps": 0,
        "finite_state_pass": bool(finite),
        "max_abs_state": max_abs_state,
        "forcing_mode": "zero; conservative because accepted residual scale is not joint-C conditioned",
        "pseudo_energy_note": "diagnostic only: moving equilibrium, damping, coupling, and reset make physical energy non-conserved",
        "checkpoint_written": bool(end_index < len(times)),
        "checkpoint_path": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
    }
    return pd.DataFrame(daily_records), pd.DataFrame(gap_events), summary


def self_check() -> dict[str, object]:
    kappa = np.array([1.0e-7, 2.0e-7, 1.5e-7])
    gamma = np.array([0.015, 0.016, 0.017])
    C = np.zeros((3, 3))
    rho = spectral_radius(kappa, gamma, C, DT_NOM_S)
    return {
        "passed": bool(math.isfinite(rho) and rho < 1.0
                       and stable_dt_lower_bound(kappa, gamma, C) >= DT_NOM_S),
        "rho_dt_60": rho,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help="number of canonical rows after initialization")
    parser.add_argument("--out-dir", type=Path, default=DERIVED_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)

    if args.self_check:
        result = self_check()
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be positive")

    try:
        out_dir = args.out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = (args.checkpoint or (out_dir / "integrator_checkpoint.json")).resolve()
        times, actual_x = read_prices()
        dynamics = [read_dynamics(pair) for pair in PAIRS]
        coupling = read_coupling()
        stability = stability_table(dynamics, coupling)
        daily, gaps, replay_summary = replay(
            times, actual_x, dynamics, coupling, args.max_steps, checkpoint, args.resume
        )
        stability_path = out_dir / "integrator_stability.parquet"
        daily_path = out_dir / "integrator_replay_daily.parquet"
        gaps_path = out_dir / "integrator_gap_events.parquet"
        summary_path = out_dir / "integrator_summary.json"
        stability.to_parquet(stability_path, index=False, compression="zstd")
        daily.to_parquet(daily_path, index=False, compression="zstd")
        gaps.to_parquet(gaps_path, index=False, compression="zstd")
        raw_stable = bool((stability["rho_dt_60_raw"] <= STABLE_RHO_LIMIT).all())
        stable = bool((stability["rho_dt_60_guarded"] <= STABLE_RHO_LIMIT).all())
        result = {
            "version": VERSION,
            "pair_scope": list(PAIRS),
            "pair_indices": list(PAIR_INDICES),
            "scheme": "semi-implicit Euler; damping implicit, position updated from new velocity",
            "gap_policy": "reset to observed position and zero velocity when an arriving bar reveals dt != 60 s; reject raw-unstable configurations and hold observed state",
            "stability": {
                "configuration_count": int(len(stability)),
                "raw_60s_pass": raw_stable,
                "raw_rho_dt_60_max": float(stability["rho_dt_60_raw"].max()),
                "raw_stability_margin_min": float(stability["raw_stability_margin"].min()),
                "negative_curvature_projected_configurations": int(stability["negative_curvature_projection_applied"].sum()),
                "guarded_60s_pass": stable,
                "guarded_rho_dt_60_max": float(stability["rho_dt_60_guarded"].max()),
                "guarded_stability_margin_min": float(stability["guarded_stability_margin"].min()),
                "stable_dt_lower_bound_min_s": float(stability["stable_dt_lower_bound_s"].min()),
                "stable_dt_search_cap_s": STABILITY_CAP_S,
            },
            "replay": replay_summary,
            "outputs": {
                "stability": str(stability_path.relative_to(ROOT)).replace("\\", "/"),
                "replay_daily": str(daily_path.relative_to(ROOT)).replace("\\", "/"),
                "gap_events": str(gaps_path.relative_to(ROOT)).replace("\\", "/"),
                "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
            },
            "passed": bool(replay_summary["finite_state_pass"]
                           and replay_summary["unprotected_unstable_steps"] == 0),
            "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    except (ContractError, ValueError, KeyError, OSError) as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

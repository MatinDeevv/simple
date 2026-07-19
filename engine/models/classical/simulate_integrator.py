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

from engine.core.contracts import canonical_pair_order


ROOT = Path(__file__).resolve().parents[3]
CANONICAL_DIR = ROOT / "data" / "canonical"
DERIVED_DIR = ROOT / "data" / "derived"

PAIR_INDICES = (0, 1, 5)
PAIRS = tuple(canonical_pair_order(ROOT)[index] for index in PAIR_INDICES)
DT_NOM_S = 60.0
DT_NOM_NS = int(DT_NOM_S * 1_000_000_000)
DEFAULT_MAX_STEPS = 250_000
STABILITY_CAP_S = 3_600.0
VERSION = "integrator-1.2.0"
STABLE_RHO_LIMIT = 1.0 + 1.0e-10
TRANSIENT_HORIZON_STEPS = 60
TRANSIENT_GROWTH_LIMIT = 10.0
PSEUDOSPECTRAL_UNIT_CIRCLE_SAMPLES = 16
STABILITY_DT_GRID_S = np.unique(np.concatenate((
    np.arange(1.0, 121.0, 1.0),
    np.arange(150.0, STABILITY_CAP_S + 1.0, 30.0),
)))


class ContractError(RuntimeError):
    pass


def utc_iso(ns: int) -> str:
    return pd.Timestamp(ns, unit="ns", tz="UTC").isoformat()


def epoch_ns(values: object) -> np.ndarray:
    """Convert a timezone-aware timestamp column/index to UTC epoch ns."""
    return pd.DatetimeIndex(values).as_unit("ns").asi8


def display_path(path: Path) -> str:
    """Prefer repo-relative artifact paths without rejecting valid external paths."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


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


def amplification_metrics(kappa: np.ndarray, gamma: np.ndarray,
                          coupling: np.ndarray, dt_s: float,
                          horizon_steps: int = TRANSIENT_HORIZON_STEPS) -> dict[str, float]:
    """Measure eigenvalue *and* non-normal transient stability at a fixed dt."""
    if horizon_steps < 1:
        raise ContractError("transient-growth horizon must be positive")
    A = amplification_matrix(kappa, gamma, coupling, dt_s)
    # Norms of [x, v] are dimensionally meaningless because x is in log-price
    # units and v is log-price/s.  Assess non-normal growth in [x, dt*v], where
    # both blocks are one-bar log-price increments.  Eigenvalues are invariant
    # under this similarity transform.
    n = len(kappa)
    scale = np.block([
        [np.eye(n), np.zeros((n, n))],
        [np.zeros((n, n)), dt_s * np.eye(n)],
    ])
    inverse_scale = np.block([
        [np.eye(n), np.zeros((n, n))],
        [np.zeros((n, n)), np.eye(n) / dt_s],
    ])
    balanced_A = scale @ A @ inverse_scale
    eigenvalues, eigenvectors = np.linalg.eig(balanced_A)
    rho = float(np.max(np.abs(eigenvalues)))
    sigma_max = float(np.linalg.svd(balanced_A, compute_uv=False)[0])
    condition = float(np.linalg.cond(eigenvectors))
    power = np.eye(balanced_A.shape[0])
    transient_peak = 1.0
    for _ in range(horizon_steps):
        power = balanced_A @ power
        transient_peak = max(transient_peak, float(np.linalg.svd(power, compute_uv=False)[0]))

    # For a discrete map, the reciprocal of the largest unit-circle resolvent
    # norm estimates the perturbation needed to push an eigenvalue to |z|=1.
    # It is a sensitivity diagnostic, not a claim of an exact pseudospectrum.
    resolvent_peak = 0.0
    identity = np.eye(balanced_A.shape[0], dtype=np.complex128)
    for angle in np.linspace(0.0, 2.0 * math.pi,
                             PSEUDOSPECTRAL_UNIT_CIRCLE_SAMPLES, endpoint=False):
        z = complex(math.cos(angle), math.sin(angle))
        smallest = float(np.linalg.svd(z * identity - balanced_A, compute_uv=False)[-1])
        resolvent_peak = math.inf if smallest == 0.0 else max(resolvent_peak, 1.0 / smallest)
    pseudospectral_distance = 0.0 if math.isinf(resolvent_peak) else 1.0 / resolvent_peak
    values = (rho, sigma_max, condition, transient_peak, pseudospectral_distance)
    if not all(math.isfinite(value) or math.isinf(value) for value in values):
        raise ContractError("non-finite amplification stability diagnostic")
    return {
        "rho": rho,
        "sigma_max": sigma_max,
        "eigenvector_condition_number": condition,
        "transient_growth_max": transient_peak,
        "unit_circle_pseudospectral_distance_estimate": pseudospectral_distance,
    }


def sampled_dt_stability(kappa: np.ndarray, gamma: np.ndarray,
                         coupling: np.ndarray) -> dict[str, float | int | bool]:
    """Sample stability across dt without assuming stable regions are monotone."""
    radii = np.asarray([spectral_radius(kappa, gamma, coupling, float(dt))
                        for dt in STABILITY_DT_GRID_S])
    stable = radii <= STABLE_RHO_LIMIT
    starts = np.flatnonzero(stable & np.r_[True, ~stable[:-1]])
    return {
        "dt_grid_samples": int(len(STABILITY_DT_GRID_S)),
        "dt_stable_interval_count": int(len(starts)),
        "dt_stable_sample_max_s": float(STABILITY_DT_GRID_S[stable].max()) if stable.any() else 0.0,
        "dt_60s_spectral_pass": bool(spectral_radius(kappa, gamma, coupling, DT_NOM_S)
                                      <= STABLE_RHO_LIMIT),
    }


def stability_table(dynamics: list[dict[str, np.ndarray]],
                    coupling: dict[str, np.ndarray]) -> pd.DataFrame:
    config_times = np.unique(np.concatenate([
        *(table["times"] for table in dynamics), coupling["times"],
    ]))
    first_ready = max(*(table["times"][0] for table in dynamics), coupling["times"][0])
    records: list[dict[str, object]] = []
    for now_ns in config_times[config_times >= first_ready]:
        kappa, gamma, _x_eq, C = configuration_at(int(now_ns), dynamics, coupling)
        raw_metrics = amplification_metrics(kappa, gamma, C, DT_NOM_S)
        negative_curvature_components = int(np.sum(kappa < 0.0))
        kappa_sim = np.maximum(kappa, 0.0)
        guarded_metrics = amplification_metrics(kappa_sim, gamma, C, DT_NOM_S)
        dt_metrics = sampled_dt_stability(kappa_sim, gamma, C)
        transient_pass = guarded_metrics["transient_growth_max"] <= TRANSIENT_GROWTH_LIMIT
        records.append({
            "timestamp": pd.Timestamp(now_ns, unit="ns", tz="UTC"),
            "rho_dt_60_raw": raw_metrics["rho"],
            "rho_dt_60_guarded": guarded_metrics["rho"],
            "raw_stability_margin": 1.0 - raw_metrics["rho"],
            "guarded_stability_margin": 1.0 - guarded_metrics["rho"],
            "sigma_max_dt_60_guarded": guarded_metrics["sigma_max"],
            "eigenvector_condition_number_guarded": guarded_metrics["eigenvector_condition_number"],
            "transient_growth_max_60s_guarded": guarded_metrics["transient_growth_max"],
            "unit_circle_pseudospectral_distance_estimate_guarded": (
                guarded_metrics["unit_circle_pseudospectral_distance_estimate"]
            ),
            "transient_growth_horizon_steps": TRANSIENT_HORIZON_STEPS,
            "transient_growth_pass": transient_pass,
            **dt_metrics,
            "negative_curvature_components": negative_curvature_components,
            "negative_curvature_projection_applied": bool(negative_curvature_components),
            "max_abs_kappa_s_minus_2": float(np.max(np.abs(kappa))),
            "max_gamma_s_minus_1": float(np.max(gamma)),
            "coupling_spectral_norm_s_minus_2": float(np.linalg.norm(C, ord=2)),
            "status": (
                "STABLE_GUARDED_60S_TRANSIENT_BOUNDED"
                if guarded_metrics["rho"] <= STABLE_RHO_LIMIT and transient_pass
                else "REJECTED_GUARDED_NONNORMAL_TRANSIENT"
            ),
        })
    result = pd.DataFrame(records)
    if result.empty:
        raise ContractError("no common stability configurations")
    return result


def write_checkpoint(path: Path, next_arrival_index: int, times: np.ndarray,
                     x_hat: np.ndarray, v_hat: np.ndarray) -> None:
    state_index = next_arrival_index - 1
    if not 1 <= state_index < len(times) - 1:
        raise ContractError("checkpoint state/next-arrival indices are outside the replay")
    payload = {
        "version": VERSION,
        "pair_scope": list(PAIRS),
        "state_index": int(state_index),
        "next_arrival_index": int(next_arrival_index),
        "state_timestamp": utc_iso(int(times[state_index])),
        "x_hat": [float(value) for value in x_hat],
        "v_hat": [float(value) for value in v_hat],
        "restore_rule": (
            "state is at state_index; resume must process next_arrival_index exactly once; "
            "reload causal inputs by timestamp and never infer velocity across a gap"
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_checkpoint(path: Path, times: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") not in {"integrator-1.1.0", VERSION} or payload.get("pair_scope") != list(PAIRS):
        raise ContractError("checkpoint contract does not match this integrator")
    # v1.1 stored next_row_index plus state at next_row_index-1.  Its data is
    # unambiguous; only the old replay loop skipped the next arrival.
    next_arrival_index = int(payload.get("next_arrival_index", payload.get("next_row_index")))
    state_index = int(payload.get("state_index", next_arrival_index - 1))
    if state_index != next_arrival_index - 1 or not 1 <= state_index < len(times):
        raise ContractError("checkpoint state_index/next_arrival_index contract is invalid")
    expected_timestamp = payload.get("state_timestamp", payload.get("last_timestamp"))
    if utc_iso(int(times[state_index])) != expected_timestamp:
        raise ContractError("checkpoint timestamp does not match canonical stream")
    x_hat = np.asarray(payload["x_hat"], dtype=np.float64)
    v_hat = np.asarray(payload["v_hat"], dtype=np.float64)
    if x_hat.shape != (len(PAIRS),) or v_hat.shape != (len(PAIRS),):
        raise ContractError("checkpoint state has wrong shape")
    if not np.isfinite(x_hat).all() or not np.isfinite(v_hat).all():
        raise ContractError("checkpoint state is non-finite")
    return next_arrival_index, x_hat, v_hat


def replay(times: np.ndarray, actual_x: np.ndarray, dynamics: list[dict[str, np.ndarray]],
           coupling: dict[str, np.ndarray], max_steps: int, checkpoint: Path,
           resume: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    first_ready = max(*(table["times"][0] for table in dynamics), coupling["times"][0])
    if resume:
        next_arrival_index, x_hat, v_hat = load_checkpoint(checkpoint, times)
        initialised_at = int(times[next_arrival_index - 1])
    else:
        state_index = int(np.searchsorted(times, first_ready, side="left"))
        while state_index < len(times) and (
            state_index == 0 or times[state_index] - times[state_index - 1] != DT_NOM_NS
        ):
            state_index += 1
        if state_index >= len(times) - 1:
            raise ContractError("cannot find a contiguous causal replay start")
        x_hat = actual_x[state_index].copy()
        v_hat = (actual_x[state_index] - actual_x[state_index - 1]) / DT_NOM_S
        initialised_at = int(times[state_index])
        next_arrival_index = state_index + 1

    end_index = min(len(times), next_arrival_index + max_steps)
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
    cached_guarded_metrics: dict[str, float] | None = None
    finite = True

    for index in range(next_arrival_index, end_index):
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
            cached_guarded_metrics = amplification_metrics(
                np.maximum(cached_config[0], 0.0), cached_config[1],
                cached_config[3], DT_NOM_S,
            )
            last_signature = signature
        assert cached_config is not None and cached_guarded_metrics is not None
        kappa, gamma, x_eq, C = cached_config
        if np.any(kappa < 0.0):
            if signature not in projected_configurations:
                projected_configurations.add(signature)
                gap_events.append({
                    "arrival_time": pd.Timestamp(arrival_ns, unit="ns", tz="UTC"),
                    "previous_time": pd.Timestamp(now_ns, unit="ns", tz="UTC"),
                    "observed_gap_s": DT_NOM_S,
                    "action": "MODEL_PROJECT_NEGATIVE_KAPPA_TO_ZERO",
                })
            projected_steps += 1
            kappa = np.maximum(kappa, 0.0)
        if (cached_guarded_metrics["rho"] > STABLE_RHO_LIMIT
                or cached_guarded_metrics["transient_growth_max"] > TRANSIENT_GROWTH_LIMIT):
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
                    "action": "REJECT_UNSTABLE_OR_TRANSIENT_CONFIG_HOLD_OBSERVED",
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
        "arrivals_processed": int(end_index - next_arrival_index),
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
        "checkpoint_path": display_path(checkpoint),
        "final_state": {
            "timestamp": utc_iso(int(times[end_index - 1])),
            "x_hat": [float(value) for value in x_hat],
            "v_hat": [float(value) for value in v_hat],
        },
    }
    return pd.DataFrame(daily_records), pd.DataFrame(gap_events), summary


def self_check() -> dict[str, object]:
    kappa = np.array([1.0e-7, 2.0e-7, 1.5e-7])
    gamma = np.array([0.015, 0.016, 0.017])
    C = np.zeros((3, 3))
    metrics = amplification_metrics(kappa, gamma, C, DT_NOM_S, horizon_steps=12)
    dt_metrics = sampled_dt_stability(kappa, gamma, C)
    return {
        "passed": bool(metrics["rho"] < 1.0
                       and metrics["transient_growth_max"] <= TRANSIENT_GROWTH_LIMIT
                       and dt_metrics["dt_60s_spectral_pass"]),
        "rho_dt_60": metrics["rho"],
        "sigma_max_dt_60": metrics["sigma_max"],
        "transient_growth_max": metrics["transient_growth_max"],
        "dt_stable_interval_count": dt_metrics["dt_stable_interval_count"],
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
        stable = bool((stability["rho_dt_60_guarded"] <= STABLE_RHO_LIMIT).all()
                      and stability["transient_growth_pass"].all())
        result = {
            "version": VERSION,
            "pair_scope": list(PAIRS),
            "pair_indices": list(PAIR_INDICES),
            "scheme": "semi-implicit Euler; damping implicit, position updated from new velocity",
            "gap_policy": "reset to observed position and zero velocity when an arriving bar reveals dt != 60 s; reject guarded spectral/transient-unstable configurations and hold observed state",
            "stability": {
                "configuration_count": int(len(stability)),
                "raw_60s_pass": raw_stable,
                "raw_rho_dt_60_max": float(stability["rho_dt_60_raw"].max()),
                "raw_stability_margin_min": float(stability["raw_stability_margin"].min()),
                "negative_curvature_projected_configurations": int(stability["negative_curvature_projection_applied"].sum()),
                "guarded_60s_and_transient_pass": stable,
                "guarded_rho_dt_60_max": float(stability["rho_dt_60_guarded"].max()),
                "guarded_stability_margin_min": float(stability["guarded_stability_margin"].min()),
                "guarded_sigma_max_dt_60_max": float(stability["sigma_max_dt_60_guarded"].max()),
                "guarded_transient_growth_max": float(stability["transient_growth_max_60s_guarded"].max()),
                "transient_growth_horizon_steps": TRANSIENT_HORIZON_STEPS,
                "transient_growth_limit": TRANSIENT_GROWTH_LIMIT,
                "sampled_dt_stable_interval_count_max": int(stability["dt_stable_interval_count"].max()),
                "sampled_dt_stable_max_s_min": float(stability["dt_stable_sample_max_s"].min()),
                "stable_dt_scan_cap_s": STABILITY_CAP_S,
            },
            "replay": replay_summary,
            "outputs": {
                "stability": str(stability_path.relative_to(ROOT)).replace("\\", "/"),
                "replay_daily": str(daily_path.relative_to(ROOT)).replace("\\", "/"),
                "gap_events": str(gaps_path.relative_to(ROOT)).replace("\\", "/"),
                "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
            },
            "numerical_safety_pass": bool(replay_summary["finite_state_pass"]
                                            and replay_summary["unprotected_unstable_steps"] == 0
                                            and stable),
            "harmonic_model_identification_pass": bool(
                replay_summary["negative_curvature_projected_steps"] == 0
            ),
            "model_status": (
                "REJECTED_HARMONIC_IDENTIFICATION_NEGATIVE_CURVATURE_PROJECTED"
                if replay_summary["negative_curvature_projected_steps"] else "UNASSESSED"
            ),
            # CLI success reports a completed numerically safe diagnostic. It
            # does not imply the fitted harmonic model is scientifically accepted.
            "passed": bool(replay_summary["finite_state_pass"]
                           and replay_summary["unprotected_unstable_steps"] == 0
                           and stable),
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

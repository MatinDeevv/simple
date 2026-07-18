"""Causal FX regime/stat-arbitrage research arena.

This is a research and forecast-evaluation system, not an execution engine.
It consumes canonical one-minute BID closes, discovers dynamic factors in an
identity-free FX basis, filters a two-state volatility regime, ranks residual
dislocations, creates factor-neutral *diagnostic* allocations, and evaluates a
predeclared residual-convergence probability target.  Bid-only bars contain no
spread, fill, queue, impact, or capacity data, so this program never reports
P&L, tradability, or a promotion decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from contracts import (ContractError as SharedContractError, canonical_pair_order,
                       contiguous_60s, validate_generated_manifest)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = ROOT / "data_canonical"
DERIVED_DIR = ROOT / "data_derived"
PAIRS = canonical_pair_order(ROOT)
N_PAIRS = len(PAIRS)
DT_NS = 60_000_000_000
VERSION = "stat-arb-arena-0.1.0"
OUTER_TEST_YEARS = (2022, 2023, 2024)


class ContractError(RuntimeError):
    """Raised when a causal-research contract is violated."""


@dataclass(frozen=True)
class ArenaConfig:
    """Predeclared, non-tuned settings for the initial OQ-14 arena."""

    factor_count: int = 3
    covariance_half_life_steps: float = 1_440.0
    residual_half_life_steps: float = 1_440.0
    factor_refresh_steps: int = 60
    # The synchronized ten-pair stream has legitimate session/data gaps, so
    # every reset needs a bounded observed-contiguous warmup rather than a
    # fictional 24-hour return path across missing arrivals.
    warmup_steps: int = 60
    outcome_horizon_steps: int = 30
    graph_partial_correlation_threshold: float = 0.15
    covariance_shrinkage: float = 0.20
    bootstrap_samples: int = 250
    bootstrap_block_steps: int = 1_440
    bootstrap_seed: int = 20260718


@dataclass
class ArenaResult:
    emissions: pd.DataFrame
    graph: pd.DataFrame
    summary: dict[str, Any]


def epoch_ns(values: object) -> np.ndarray:
    return pd.DatetimeIndex(values).as_unit("ns").asi8


def sigmoid(value: float) -> float:
    bounded = float(np.clip(value, -40.0, 40.0))
    return 1.0 / (1.0 + math.exp(-bounded))


def log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1.0e-12, 1.0 - 1.0e-12)
    return float(np.mean(-(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))))


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probabilities - labels) ** 2))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_free_transform(pairs: tuple[str, ...] = PAIRS) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Return the coupling-compatible FX identity-free return basis.

    EURGBP, EURJPY, and GBPJPY carry near-arithmetic cross identities.  Their
    transformed channels are residuals, so factor/graph estimates do not treat
    those identities as independent economic relationships.
    """
    required = {"EURUSD", "USDJPY", "GBPUSD", "EURGBP", "EURJPY", "GBPJPY"}
    if len(pairs) != N_PAIRS or not required.issubset(pairs):
        raise ContractError("tracked pair order cannot support the identity-free FX transform")
    index = {pair: position for position, pair in enumerate(pairs)}
    transform = np.eye(len(pairs), dtype=np.float64)
    transform[index["EURGBP"]] = 0.0
    transform[index["EURGBP"], index["EURGBP"]] = 1.0
    transform[index["EURGBP"], index["EURUSD"]] = -1.0
    transform[index["EURGBP"], index["GBPUSD"]] = 1.0
    transform[index["EURJPY"]] = 0.0
    transform[index["EURJPY"], index["EURJPY"]] = 1.0
    transform[index["EURJPY"], index["EURUSD"]] = -1.0
    transform[index["EURJPY"], index["USDJPY"]] = -1.0
    transform[index["GBPJPY"]] = 0.0
    transform[index["GBPJPY"], index["GBPJPY"]] = 1.0
    transform[index["GBPJPY"], index["GBPUSD"]] = -1.0
    transform[index["GBPJPY"], index["USDJPY"]] = -1.0
    inverse = np.linalg.inv(transform)
    labels = list(pairs)
    labels[index["EURGBP"]] = "EURGBP_triangle_residual"
    labels[index["EURJPY"]] = "EURJPY_triangle_residual"
    labels[index["GBPJPY"]] = "GBPJPY_triangle_residual"
    return transform, inverse, tuple(labels)


def factor_neutral_weights(signal_basis: np.ndarray, inverse_transform: np.ndarray,
                           loadings_basis: np.ndarray) -> np.ndarray:
    """Project a residual signal off the current factor span and normalize L1.

    This is a diagnostic allocation only.  It omits every execution prerequisite
    (spread, borrow, limits, impact, and fill probability) and must not be sent
    to an order-management system.
    """
    raw_signal = inverse_transform @ signal_basis
    raw_loadings = inverse_transform @ loadings_basis
    # Solve both constraints simultaneously.  Demeaning *after* removing the
    # factor span can reintroduce factor exposure when the all-ones vector has
    # nonzero factor loadings.
    constraints = np.column_stack((raw_loadings, np.ones(N_PAIRS)))
    projection = constraints @ (np.linalg.pinv(constraints) @ raw_signal)
    weights = raw_signal - projection
    scale = float(np.sum(np.abs(weights)))
    return weights / scale if scale > 1.0e-12 else np.zeros_like(weights)


def moving_block_bootstrap_lower_bound(loss_improvement: np.ndarray, segment_ids: np.ndarray,
                                       block_steps: int, samples: int, seed: int) -> float:
    """One-sided 95% moving-block-bootstrap lower bound for score improvement."""
    if len(loss_improvement) < 2 or len(loss_improvement) != len(segment_ids):
        return float("nan")
    if block_steps < 1 or samples < 1:
        raise ContractError("bootstrap block_steps and samples must be positive")
    chunks: list[np.ndarray] = []
    start = 0
    while start < len(loss_improvement):
        end = start + 1
        while end < len(loss_improvement) and segment_ids[end] == segment_ids[start]:
            end += 1
        if end - start > 0:
            chunks.append(loss_improvement[start:end])
        start = end
    candidates = [(chunk, offset) for chunk in chunks for offset in range(len(chunk))]
    if not candidates:
        return float("nan")
    rng = np.random.default_rng(seed)
    draw_blocks = max(1, math.ceil(len(loss_improvement) / block_steps))
    estimates = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        drawn: list[np.ndarray] = []
        remaining = len(loss_improvement)
        for _ in range(draw_blocks):
            chunk, offset = candidates[int(rng.integers(len(candidates)))]
            width = min(block_steps, len(chunk) - offset, remaining)
            if width > 0:
                drawn.append(chunk[offset:offset + width])
                remaining -= width
            if remaining <= 0:
                break
        values = np.concatenate(drawn) if drawn else loss_improvement
        estimates[sample] = float(np.mean(values))
    return float(np.quantile(estimates, 0.05))


class CausalFactorState:
    """Online dynamic-factor, graph, AR, and two-regime state.

    Every field is updated with the current observed return before an emission
    at the same bar close.  No outcome or future return enters this object.
    """

    def __init__(self, config: ArenaConfig, transform: np.ndarray) -> None:
        if not 1 <= config.factor_count < N_PAIRS:
            raise ContractError("factor_count must be between one and nine")
        self.config = config
        self.transform = transform
        self.alpha_covariance = 1.0 - math.exp(-math.log(2.0) / config.covariance_half_life_steps)
        self.alpha_residual = 1.0 - math.exp(-math.log(2.0) / config.residual_half_life_steps)
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.mean = np.zeros(N_PAIRS, dtype=np.float64)
        self.covariance = np.eye(N_PAIRS, dtype=np.float64) * 1.0e-8
        self.loadings = np.eye(N_PAIRS, self.config.factor_count, dtype=np.float64)
        self.residual_variance = np.full(N_PAIRS, 1.0e-8, dtype=np.float64)
        self.ar_numerator = np.zeros(N_PAIRS, dtype=np.float64)
        self.ar_denominator = np.full(N_PAIRS, 1.0e-12, dtype=np.float64)
        self.previous_residual: np.ndarray | None = None
        self.log_vol_mean = 0.0
        self.log_vol_variance = 1.0
        self.regime_high_probability = 0.5
        self.previous_weights = np.zeros(N_PAIRS, dtype=np.float64)
        self.partial_correlation = np.zeros((N_PAIRS, N_PAIRS), dtype=np.float64)

    def _refresh_factor_and_graph(self) -> list[tuple[int, int, float]]:
        diagonal_scale = max(float(np.trace(self.covariance) / N_PAIRS), 1.0e-12)
        regularized = ((1.0 - self.config.covariance_shrinkage) * self.covariance
                       + self.config.covariance_shrinkage * np.eye(N_PAIRS) * diagonal_scale)
        values, vectors = np.linalg.eigh(regularized)
        order = np.argsort(values)[::-1]
        self.loadings = vectors[:, order[:self.config.factor_count]]
        precision = np.linalg.pinv(regularized)
        precision_diagonal = np.maximum(np.diag(precision), 1.0e-18)
        partial = -precision / np.sqrt(np.outer(precision_diagonal, precision_diagonal))
        np.fill_diagonal(partial, 0.0)
        self.partial_correlation = np.clip(partial, -1.0, 1.0)
        edges: list[tuple[int, int, float]] = []
        for row in range(N_PAIRS):
            for column in range(row + 1, N_PAIRS):
                value = float(self.partial_correlation[row, column])
                if abs(value) >= self.config.graph_partial_correlation_threshold:
                    edges.append((row, column, value))
        return edges

    def update(self, raw_return: np.ndarray) -> tuple[dict[str, Any], list[tuple[int, int, float]]]:
        if raw_return.shape != (N_PAIRS,) or not np.isfinite(raw_return).all():
            raise ContractError("factor state requires finite ten-pair return vector")
        transformed_return = self.transform @ raw_return
        delta = transformed_return - self.mean
        self.mean += self.alpha_covariance * delta
        self.covariance = ((1.0 - self.alpha_covariance) * self.covariance
                           + self.alpha_covariance * np.outer(delta, transformed_return - self.mean))
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.steps += 1
        graph_edges = []
        if self.steps == 1 or self.steps % self.config.factor_refresh_steps == 0:
            graph_edges = self._refresh_factor_and_graph()
        centered = transformed_return - self.mean
        factors = self.loadings.T @ centered
        residual = centered - self.loadings @ factors
        self.residual_variance = ((1.0 - self.alpha_residual) * self.residual_variance
                                  + self.alpha_residual * residual * residual)
        z = residual / np.sqrt(np.maximum(self.residual_variance, 1.0e-16))
        if self.previous_residual is not None:
            self.ar_numerator = ((1.0 - self.alpha_residual) * self.ar_numerator
                                 + self.alpha_residual * self.previous_residual * residual)
            self.ar_denominator = ((1.0 - self.alpha_residual) * self.ar_denominator
                                   + self.alpha_residual * self.previous_residual * self.previous_residual)
        beta = self.ar_numerator / np.maximum(self.ar_denominator, 1.0e-16)
        beta = np.clip(beta, -0.9999, 0.9999)
        half_life = np.full(N_PAIRS, np.inf, dtype=np.float64)
        positive = (beta > 1.0e-6) & (beta < 0.999)
        half_life[positive] = -math.log(2.0) / np.log(beta[positive])
        factor_volatility = float(np.sqrt(np.mean(factors * factors)))
        residual_volatility = float(np.sqrt(np.mean(residual * residual)))
        log_volatility = math.log(max(factor_volatility + residual_volatility, 1.0e-12))
        prior_mean = self.log_vol_mean
        prior_scale = math.sqrt(max(self.log_vol_variance, 1.0e-8))
        volatility_z = float(np.clip((log_volatility - prior_mean) / prior_scale, -8.0, 8.0))
        transition_high = 0.985 * self.regime_high_probability + 0.010 * (1.0 - self.regime_high_probability)
        low_likelihood = math.exp(-0.5 * (volatility_z + 0.55) ** 2)
        high_likelihood = math.exp(-0.5 * (volatility_z - 0.55) ** 2)
        normalization = ((1.0 - transition_high) * low_likelihood
                         + transition_high * high_likelihood)
        self.regime_high_probability = (transition_high * high_likelihood / normalization
                                        if normalization > 0.0 else 0.5)
        log_delta = log_volatility - self.log_vol_mean
        self.log_vol_mean += self.alpha_residual * log_delta
        self.log_vol_variance = ((1.0 - self.alpha_residual) * self.log_vol_variance
                                 + self.alpha_residual * log_delta * (log_volatility - self.log_vol_mean))
        self.previous_residual = residual.copy()
        return {
            "residual": residual,
            "z": z,
            "beta": beta,
            "half_life": half_life,
            "factors": factors,
            "factor_volatility": factor_volatility,
            "residual_volatility": residual_volatility,
            "volatility_z": volatility_z,
            "regime_high_probability": self.regime_high_probability,
            "loadings": self.loadings.copy(),
        }, graph_edges


def prediction_from_features(features: dict[str, Any], horizon_steps: int) -> tuple[int, float, float, np.ndarray]:
    """Produce a fixed, predeclared residual-convergence probability.

    The score is deliberately deterministic and untuned.  It is a baseline
    arena candidate, not a learned alpha model.
    """
    z = np.asarray(features["z"], dtype=np.float64)
    half_life = np.asarray(features["half_life"], dtype=np.float64)
    beta = np.asarray(features["beta"], dtype=np.float64)
    selected = int(np.argmax(np.abs(z)))
    selected_abs_z = float(abs(z[selected]))
    selected_half_life = float(half_life[selected])
    speed = (math.log1p(horizon_steps / selected_half_life)
             if math.isfinite(selected_half_life) and selected_half_life > 0.0 else 0.0)
    unstable = float(np.mean((beta <= 0.0) | (beta >= 0.995)))
    breakdown = sigmoid(-2.0 + 1.40 * float(features["regime_high_probability"])
                        + 0.45 * min(selected_abs_z, 6.0) + 2.25 * unstable)
    probability = sigmoid(-0.80 + 0.65 * min(selected_abs_z, 6.0)
                          + 0.35 * speed - 2.50 * breakdown
                          - 0.80 * float(features["regime_high_probability"]))
    signal_basis = np.zeros(N_PAIRS, dtype=np.float64)
    signal_basis[selected] = -float(np.sign(z[selected]) or 1.0)
    return selected, probability, breakdown, signal_basis


def _append_feature_record(records: list[dict[str, Any]], *, index: int, timestamp_ns: int,
                           segment_id: int, features: dict[str, Any], selected: int,
                           probability: float, breakdown: float, weights: np.ndarray,
                           turnover: float, component_names: tuple[str, ...]) -> None:
    record: dict[str, Any] = {
        "source_index": index,
        "timestamp": pd.Timestamp(timestamp_ns, unit="ns", tz="UTC"),
        "segment_id": segment_id,
        "selected_component_index": selected,
        "selected_component": component_names[selected],
        "p_convergence": probability,
        "breakdown_probability": breakdown,
        "high_volatility_regime_probability": float(features["regime_high_probability"]),
        "factor_volatility": float(features["factor_volatility"]),
        "residual_volatility": float(features["residual_volatility"]),
        "volatility_z": float(features["volatility_z"]),
        "turnover_l1_diagnostic": turnover,
        "target_time": pd.NaT,
        "convergence_label": np.nan,
    }
    for pair, value in zip(PAIRS, weights, strict=True):
        record[f"diagnostic_weight_{pair.lower()}"] = float(value)
    for name, value in zip(component_names, np.asarray(features["z"]), strict=True):
        record[f"residual_z_{name.lower()}"] = float(value)
    for name, value in zip(component_names, np.asarray(features["half_life"]), strict=True):
        record[f"residual_half_life_steps_{name.lower()}"] = float(value)
    records.append(record)


def run_arrays(times: np.ndarray, log_prices: np.ndarray, test_start_index: int,
               config: ArenaConfig = ArenaConfig()) -> ArenaResult:
    """Run the causal arena over supplied synchronous log-close observations."""
    if times.ndim != 1 or log_prices.shape != (len(times), N_PAIRS):
        raise ContractError("times/log_prices do not match the ten-pair arena contract")
    if len(times) <= config.warmup_steps + config.outcome_horizon_steps + 2:
        raise ContractError("research window is too short for warmup and target horizon")
    if not np.all(np.diff(times) > 0) or not np.isfinite(log_prices).all():
        raise ContractError("research input must be strictly increasing and finite")
    if not config.warmup_steps < test_start_index < len(times) - config.outcome_horizon_steps:
        raise ContractError("test_start_index does not leave train and OOS target observations")
    transform, inverse_transform, component_names = identity_free_transform()
    state = CausalFactorState(config, transform)
    records: list[dict[str, Any]] = []
    graph_records: list[dict[str, Any]] = []
    segment_id = 0
    gap_resets = 0
    max_contiguous_state_steps = 0
    feature_by_source: dict[int, int] = {}
    for index in range(1, len(times)):
        if not contiguous_60s(int(times[index - 1]), int(times[index]), DT_NS):
            state.reset()
            segment_id += 1
            gap_resets += 1
            continue
        raw_return = log_prices[index] - log_prices[index - 1]
        features, edges = state.update(raw_return)
        max_contiguous_state_steps = max(max_contiguous_state_steps, state.steps)
        for left, right, partial in edges:
            graph_records.append({
                "timestamp": pd.Timestamp(int(times[index]), unit="ns", tz="UTC"),
                "segment_id": segment_id,
                "left_component": component_names[left],
                "right_component": component_names[right],
                "partial_correlation": partial,
            })
        if state.steps < config.warmup_steps:
            continue
        selected, probability, breakdown, signal_basis = prediction_from_features(
            features, config.outcome_horizon_steps)
        weights = factor_neutral_weights(signal_basis, inverse_transform, features["loadings"])
        turnover = float(np.sum(np.abs(weights - state.previous_weights)))
        state.previous_weights = weights
        _append_feature_record(
            records, index=index, timestamp_ns=int(times[index]), segment_id=segment_id,
            features=features, selected=selected, probability=probability, breakdown=breakdown,
            weights=weights, turnover=turnover, component_names=component_names,
        )
        feature_by_source[index] = len(records) - 1
    if not records:
        raise ContractError("no causal feature emissions were produced")
    for record in records:
        source = int(record["source_index"])
        target = source + config.outcome_horizon_steps
        target_record_index = feature_by_source.get(target)
        if target_record_index is None or target >= len(times):
            continue
        target_record = records[target_record_index]
        if int(record["segment_id"]) != int(target_record["segment_id"]):
            continue
        selected = int(record["selected_component_index"])
        source_z = abs(float(record[f"residual_z_{component_names[selected].lower()}"]))
        target_z = abs(float(target_record[f"residual_z_{component_names[selected].lower()}"]))
        record["target_time"] = pd.Timestamp(int(times[target]), unit="ns", tz="UTC")
        record["convergence_label"] = int(target_z < source_z)
    emissions = pd.DataFrame(records)
    labelled = emissions[emissions["convergence_label"].notna()].copy()
    labelled["convergence_label"] = labelled["convergence_label"].astype(np.int8)
    train = labelled[labelled["source_index"] + config.outcome_horizon_steps < test_start_index]
    oos = labelled[labelled["source_index"] >= test_start_index]
    if len(train) < 100 or len(oos) < 100:
        raise ContractError("train/OOS target partitions require at least 100 valid emissions each")
    frozen_train_prior = float(train["convergence_label"].mean())
    oos_probability = oos["p_convergence"].to_numpy(dtype=np.float64)
    oos_label = oos["convergence_label"].to_numpy(dtype=np.float64)
    prior_probability = np.full(len(oos), frozen_train_prior, dtype=np.float64)
    model_brier = brier_score(oos_probability, oos_label)
    prior_brier = brier_score(prior_probability, oos_label)
    improvement = (prior_probability - oos_label) ** 2 - (oos_probability - oos_label) ** 2
    brier_lower_bound = moving_block_bootstrap_lower_bound(
        improvement, oos["segment_id"].to_numpy(dtype=np.int64), config.bootstrap_block_steps,
        config.bootstrap_samples, config.bootstrap_seed,
    )
    prediction_pass = bool(model_brier < prior_brier and math.isfinite(brier_lower_bound)
                           and brier_lower_bound > 0.0)
    summary: dict[str, Any] = {
        "version": VERSION,
        "interpretation": (
            "causal classical FX factor/residual research arena; no execution, PnL, "
            "capacity, market-impact, market-making, options, macro-vintage, or legal-event claim"
        ),
        "pair_scope": list(PAIRS),
        "target_contract": {
            "target": "selected identity-free residual has lower absolute standardized magnitude after the fixed horizon",
            "horizon_steps": config.outcome_horizon_steps,
            "target_available_only_after": f"t+{config.outcome_horizon_steps} contiguous observed minutes",
            "target_excluded_from_feature_state": True,
            "primary_score": "Brier score",
            "baseline": "frozen training convergence frequency",
        },
        "causality": {
            "input_at_t": "synchronous canonical BID log returns through bar t only",
            "gap_policy": "reset all factor/regime/residual state on observed non-60-second arrival",
            "gap_resets": gap_resets,
            "max_contiguous_state_steps": max_contiguous_state_steps,
            "post_gap_warmup_steps": config.warmup_steps,
            "identity_control": "EURGBP/EURJPY/GBPJPY transformed to arithmetic-residual channels before factor/graph estimation",
        },
        "components": {
            "dynamic_factor_model": "EW covariance with hourly eigenspace refresh",
            "sparse_interaction_graph": "shrunken covariance precision partial correlations",
            "regime_filter": "fixed two-state causal volatility HMM filter",
            "cointegration_baskets": "structural FX triangle residual channels only; not claimed as discovered trade baskets",
            "breakdown_probability": "fixed function of regime probability, residual magnitude, and causal AR stability",
            "neutrality": "joint factor-and-net-neutral projection with L1-normalized diagnostic weights",
            "execution_cost_status": "BLOCKED: BID-only source has no spread, fill, impact, borrow, capacity, or latency inputs",
        },
        "window": {
            "start": pd.Timestamp(int(times[0]), unit="ns", tz="UTC").isoformat(),
            "end": pd.Timestamp(int(times[-1]), unit="ns", tz="UTC").isoformat(),
            "raw_rows": int(len(times)),
            "test_start": pd.Timestamp(int(times[test_start_index]), unit="ns", tz="UTC").isoformat(),
            "warmup_steps": config.warmup_steps,
        },
        "evaluation": {
            "train_samples": int(len(train)),
            "oos_samples": int(len(oos)),
            "model": {
                "brier_score": model_brier,
                "log_loss": log_loss(oos_probability, oos_label),
                "accuracy": float(np.mean((oos_probability >= 0.5) == oos_label)),
            },
            "frozen_train_prior": {
                "probability": frozen_train_prior,
                "brier_score": prior_brier,
                "log_loss": log_loss(prior_probability, oos_label),
                "accuracy": float(np.mean((prior_probability >= 0.5) == oos_label)),
            },
            "brier_improvement_lower_95_moving_block_bootstrap": brier_lower_bound,
            "passes_frozen_prior_prediction_gate": prediction_pass,
        },
        "promotion_status": (
            "REJECTED_NO_EXECUTION_DATA_AND_INCOMPLETE_OUTER_FOLDS: forecast score cannot "
            "establish tradability; run each required 2022/2023/2024 fold before any research revisit"
        ),
    }
    return ArenaResult(emissions=emissions, graph=pd.DataFrame(graph_records), summary=summary)


def load_common_log_prices(max_rows: int, test_year: int | None) -> tuple[np.ndarray, np.ndarray, int]:
    """Load either a bounded latest window or a predeclared outer-year window."""
    if max_rows < 2_000:
        raise ContractError("max_rows must be at least 2,000")
    validate_generated_manifest(ROOT)
    if test_year is not None and test_year not in OUTER_TEST_YEARS:
        raise ContractError(f"test_year must be one of {OUTER_TEST_YEARS}")
    filters: list[tuple[str, str, Any]] | None = None
    test_start: pd.Timestamp | None = None
    if test_year is not None:
        start = pd.Timestamp(f"{test_year - 2}-01-01", tz="UTC")
        end = pd.Timestamp(f"{test_year + 1}-01-01", tz="UTC")
        filters = [("timestamp", ">=", start), ("timestamp", "<", end)]
        test_start = pd.Timestamp(f"{test_year}-01-01", tz="UTC")
    else:
        reference = pd.read_parquet(CANONICAL_DIR / f"{PAIRS[0]}.parquet", columns=["timestamp"])
        if len(reference) < max_rows + 2:
            raise ContractError("canonical reference series is shorter than max_rows")
        start = reference.iloc[-max_rows - 1]["timestamp"]
        filters = [("timestamp", ">=", start)]
    joined: pd.DataFrame | None = None
    for pair in PAIRS:
        frame = pd.read_parquet(CANONICAL_DIR / f"{pair}.parquet", columns=["timestamp", "close"],
                                filters=filters)
        frame = frame.rename(columns={"close": pair})
        joined = frame if joined is None else joined.merge(frame, on="timestamp", how="inner",
                                                            validate="one_to_one")
    assert joined is not None
    joined = joined.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if test_year is None and len(joined) > max_rows:
        joined = joined.iloc[-max_rows:].reset_index(drop=True)
    times = epoch_ns(joined["timestamp"])
    prices = joined.loc[:, list(PAIRS)].to_numpy(dtype=np.float64)
    if len(times) < 2_000 or not np.all(np.diff(times) > 0) or not np.isfinite(prices).all() or np.any(prices <= 0.0):
        raise ContractError("joined canonical stat-arb input is invalid")
    if test_start is None:
        test_start_index = int(len(times) * 0.70)
    else:
        test_start_index = int(np.searchsorted(times, test_start.value, side="left"))
    return times, np.log(prices), test_start_index


def write_result(result: ArenaResult, out_dir: Path, label: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    emissions_path = out_dir / f"stat_arb_{label}_minute.parquet"
    graph_path = out_dir / f"stat_arb_{label}_graph.parquet"
    daily_path = out_dir / f"stat_arb_{label}_daily.parquet"
    summary_path = out_dir / f"stat_arb_{label}_summary.json"
    result.emissions.to_parquet(emissions_path, index=False, compression="zstd")
    result.graph.to_parquet(graph_path, index=False, compression="zstd")
    daily = (result.emissions.assign(day=result.emissions["timestamp"].dt.floor("D"))
             .groupby("day", as_index=False)
             .agg(emissions=("source_index", "size"),
                  mean_p_convergence=("p_convergence", "mean"),
                  mean_breakdown_probability=("breakdown_probability", "mean"),
                  diagnostic_turnover_l1=("turnover_l1_diagnostic", "sum")))
    daily.to_parquet(daily_path, index=False, compression="zstd")
    result.summary["outputs"] = {
        "minute": str(emissions_path.relative_to(ROOT)).replace("\\", "/"),
        "graph": str(graph_path.relative_to(ROOT)).replace("\\", "/"),
        "daily": str(daily_path.relative_to(ROOT)).replace("\\", "/"),
    }
    result.summary["source_hashes"] = {
        "pipeline/stat_arb.py": sha256_file(Path(__file__).resolve()),
        **{f"data_canonical/{pair}.parquet": sha256_file(CANONICAL_DIR / f"{pair}.parquet")
           for pair in PAIRS},
    }
    summary_path.write_text(json.dumps(result.summary, indent=2) + "\n", encoding="utf-8")
    return {"minute": str(emissions_path), "graph": str(graph_path), "daily": str(daily_path),
            "summary": str(summary_path)}


def synthetic_input(rows: int = 900) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12345)
    returns = rng.normal(0.0, 2.0e-4, size=(rows, N_PAIRS))
    common = 0.00015 * np.sin(np.arange(rows) / 19.0)
    returns += common[:, None]
    # Preserve the three cross relationships up to small residual noise.
    returns[:, PAIRS.index("EURGBP")] = (returns[:, PAIRS.index("EURUSD")]
                                          - returns[:, PAIRS.index("GBPUSD")]
                                          + rng.normal(0.0, 1e-5, rows))
    returns[:, PAIRS.index("EURJPY")] = (returns[:, PAIRS.index("EURUSD")]
                                          + returns[:, PAIRS.index("USDJPY")]
                                          + rng.normal(0.0, 1e-5, rows))
    returns[:, PAIRS.index("GBPJPY")] = (returns[:, PAIRS.index("GBPUSD")]
                                          + returns[:, PAIRS.index("USDJPY")]
                                          + rng.normal(0.0, 1e-5, rows))
    times = np.arange(rows, dtype=np.int64) * DT_NS
    times[500:] += 2 * DT_NS  # Explicit gap exercises reset semantics.
    return times, np.cumsum(returns, axis=0)


def self_check() -> dict[str, Any]:
    config = ArenaConfig(warmup_steps=64, outcome_horizon_steps=8, factor_refresh_steps=8,
                         covariance_half_life_steps=32.0, residual_half_life_steps=32.0,
                         bootstrap_samples=32, bootstrap_block_steps=32)
    times, log_prices = synthetic_input()
    result = run_arrays(times, log_prices, test_start_index=500, config=config)
    altered = log_prices.copy()
    altered[700:] += 0.02
    altered_result = run_arrays(times, altered, test_start_index=500, config=config)
    comparable = result.emissions[result.emissions["source_index"] < 680].set_index("source_index")
    altered_comparable = altered_result.emissions[altered_result.emissions["source_index"] < 680].set_index("source_index")
    common = comparable.index.intersection(altered_comparable.index)
    prefix_equal = bool(np.allclose(comparable.loc[common, "p_convergence"],
                                    altered_comparable.loc[common, "p_convergence"], atol=0.0, rtol=0.0))
    transform, inverse, _labels = identity_free_transform()
    return {
        "passed": bool(prefix_equal and np.allclose(transform @ inverse, np.eye(N_PAIRS), atol=1e-12)
                       and result.summary["causality"]["gap_resets"] == 1
                       and result.summary["evaluation"]["train_samples"] >= 100
                       and result.summary["evaluation"]["oos_samples"] >= 100),
        "prefix_emissions_bitwise_equal": prefix_equal,
        "transform_inverse_max_error": float(np.max(np.abs(transform @ inverse - np.eye(N_PAIRS)))),
        "gap_resets": result.summary["causality"]["gap_resets"],
        "train_samples": result.summary["evaluation"]["train_samples"],
        "oos_samples": result.summary["evaluation"]["oos_samples"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--max-rows", type=int, default=50_000,
                        help="latest bounded synchronous rows when --test-year is absent")
    parser.add_argument("--test-year", type=int, choices=OUTER_TEST_YEARS,
                        help="run one predeclared calendar outer fold with two prior years of state warmup")
    parser.add_argument("--out-dir", type=Path, default=DERIVED_DIR)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        result = self_check()
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    try:
        times, prices, test_start = load_common_log_prices(args.max_rows, args.test_year)
        result = run_arrays(times, prices, test_start)
        label = f"outer_{args.test_year}" if args.test_year is not None else "bounded_latest"
        outputs = write_result(result, args.out_dir.resolve(), label)
        result.summary["artifact_paths"] = outputs
        print(json.dumps(result.summary, indent=2))
        return 0
    except (ContractError, SharedContractError, ValueError, OSError, np.linalg.LinAlgError) as exc:
        print(f"[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

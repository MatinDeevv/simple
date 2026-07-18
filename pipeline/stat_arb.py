"""Causal FX residual-level research arena.

This module is deliberately a forecast-evaluation system, not an execution
engine.  It consumes canonical one-minute BID closes only.  It has no ask,
spread, fill, queue, impact, borrow, or capacity data and therefore cannot
produce a PnL, tradability, or promotion claim.

Version 0.2 replaces the invalid v0.1 return-extreme target with a frozen
residual-*level* outcome.  Every eligible entry freezes its factor basis,
mean, volatility scale, selected residual, diagnostic basket, stop, and
horizon.  Future observations are evaluated against that frozen definition;
they never use a later factor basis or normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
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
VERSION = "stat-arb-arena-0.2.0-frozen"
OUTER_TEST_YEARS = (2022, 2023, 2024)


class ContractError(RuntimeError):
    """Raised when a causal-research contract is violated."""


@dataclass(frozen=True)
class RegimeParameters:
    """Predeclared model and entry parameters for one latent volatility state."""

    covariance_half_life_steps: float
    residual_half_life_steps: float
    factor_refresh_steps: int
    level_ar: float
    entry_abs_level: float
    holding_horizon_steps: int
    stop_multiple: float
    neutral_factor_count: int


def _low_regime() -> RegimeParameters:
    return RegimeParameters(
        covariance_half_life_steps=1_440.0,
        residual_half_life_steps=1_440.0,
        factor_refresh_steps=60,
        level_ar=0.995,
        entry_abs_level=1.50,
        holding_horizon_steps=30,
        stop_multiple=2.00,
        neutral_factor_count=2,
    )


def _high_regime() -> RegimeParameters:
    return RegimeParameters(
        covariance_half_life_steps=240.0,
        residual_half_life_steps=240.0,
        factor_refresh_steps=15,
        level_ar=0.970,
        entry_abs_level=2.25,
        holding_horizon_steps=15,
        stop_multiple=1.35,
        neutral_factor_count=1,
    )


@dataclass(frozen=True)
class ArenaConfig:
    """Frozen v0.2 settings; they are not selected on an outer-fold result."""

    factor_count: int = 3
    warmup_steps: int = 60
    graph_partial_correlation_threshold: float = 0.15
    covariance_shrinkage: float = 0.20
    bootstrap_samples: int = 2_000
    bootstrap_block_sensitivity_steps: tuple[int, ...] = (30, 240, 1_440)
    bootstrap_seed: int = 20260718
    low_regime: RegimeParameters = field(default_factory=_low_regime)
    high_regime: RegimeParameters = field(default_factory=_high_regime)

    @property
    def max_horizon_steps(self) -> int:
        return max(self.low_regime.holding_horizon_steps, self.high_regime.holding_horizon_steps)


@dataclass
class ArenaResult:
    emissions: pd.DataFrame
    graph: pd.DataFrame
    summary: dict[str, Any]


@dataclass(frozen=True)
class FrozenResidualTarget:
    """t-measurable definition of a single residual-level outcome."""

    source_index: int
    segment_id: int
    selected: int
    entry_level: float
    mean_basis: np.ndarray
    loadings_basis: np.ndarray
    residual_scale: np.ndarray
    level_ar: float
    horizon_steps: int
    stop_multiple: float
    raw_basket_weights: np.ndarray


def epoch_ns(values: object) -> np.ndarray:
    return pd.DatetimeIndex(values).as_unit("ns").asi8


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(np.clip(value, -40.0, 40.0))))


def _alpha(half_life_steps: float) -> float:
    if not math.isfinite(half_life_steps) or half_life_steps <= 0.0:
        raise ContractError("EW half-life must be finite and positive")
    return 1.0 - math.exp(-math.log(2.0) / half_life_steps)


def log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1.0e-12, 1.0 - 1.0e-12)
    return float(np.mean(-(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))))


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probabilities - labels) ** 2))


def calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Equal-width expected calibration error; lower is better."""
    if len(probabilities) == 0 or len(probabilities) != len(labels):
        return float("nan")
    membership = np.minimum((np.clip(probabilities, 0.0, 1.0) * bins).astype(int), bins - 1)
    error = 0.0
    for bucket in range(bins):
        mask = membership == bucket
        if mask.any():
            error += float(mask.mean()) * abs(float(probabilities[mask].mean() - labels[mask].mean()))
    return error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_free_transform(pairs: tuple[str, ...] = PAIRS) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Return the invertible FX identity-free return basis and its inverse."""
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


def basis_signal_to_raw_weights(signal_basis: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Map a dual signal from z=T r coordinates to raw-return pair weights.

    For every raw return r, ``s @ (T @ r) == (T.T @ s) @ r``.  Using
    ``inverse(T) @ s`` here is a category error: that maps a *primal state*,
    not a linear signal/portfolio functional.
    """
    signal = np.asarray(signal_basis, dtype=np.float64)
    if signal.shape != (N_PAIRS,) or transform.shape != (N_PAIRS, N_PAIRS):
        raise ContractError("basis signal and transform must use the ten-pair contract")
    return transform.T @ signal


def currency_incidence(pairs: tuple[str, ...] = PAIRS) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return D where a pair has +1 base and -1 quote currency incidence."""
    currencies = tuple(sorted({pair[:3] for pair in pairs} | {pair[3:] for pair in pairs}))
    index = {currency: column for column, currency in enumerate(currencies)}
    incidence = np.zeros((len(pairs), len(currencies)), dtype=np.float64)
    for row, pair in enumerate(pairs):
        if len(pair) != 6:
            raise ContractError(f"invalid FX pair symbol: {pair}")
        incidence[row, index[pair[:3]]] = 1.0
        incidence[row, index[pair[3:]]] = -1.0
    return incidence, currencies


def factor_neutral_weights(signal_basis: np.ndarray, transform: np.ndarray,
                           inverse_transform: np.ndarray, loadings_basis: np.ndarray,
                           currency_matrix: np.ndarray | None = None,
                           neutral_factor_count: int = 2) -> np.ndarray:
    """Construct a currency- and selected-factor-neutral diagnostic basket.

    ``T.T`` maps the residual signal correctly as a dual vector.  In contrast,
    factor loading columns represent primal directions, so their raw mapping is
    ``inverse(T) @ B``.  The D constraint is ``D.T @ w = 0`` in pair-coefficient
    units; without contract notionals and conversion prices it is explicitly
    not a dollar-risk or execution-neutrality claim.
    """
    signal = np.asarray(signal_basis, dtype=np.float64)
    loadings = np.asarray(loadings_basis, dtype=np.float64)
    if signal.shape != (N_PAIRS,) or transform.shape != (N_PAIRS, N_PAIRS):
        raise ContractError("invalid residual signal transform")
    if inverse_transform.shape != (N_PAIRS, N_PAIRS) or loadings.shape[0] != N_PAIRS:
        raise ContractError("invalid factor basis for diagnostic weights")
    if not 0 <= neutral_factor_count <= loadings.shape[1]:
        raise ContractError("neutral_factor_count is outside the factor basis")
    incidence = currency_incidence()[0] if currency_matrix is None else np.asarray(currency_matrix, dtype=np.float64)
    if incidence.shape[0] != N_PAIRS:
        raise ContractError("currency incidence does not match pair weights")
    raw_signal = basis_signal_to_raw_weights(signal, transform)
    raw_factor_directions = inverse_transform @ loadings
    selected_factors = raw_factor_directions[:, :neutral_factor_count]
    constraints = np.column_stack((incidence, selected_factors))
    if np.linalg.matrix_rank(constraints, tol=1.0e-12) >= N_PAIRS:
        return np.zeros(N_PAIRS, dtype=np.float64)
    projection = constraints @ (np.linalg.pinv(constraints, rcond=1.0e-12) @ raw_signal)
    weights = raw_signal - projection
    scale = float(np.sum(np.abs(weights)))
    return weights / scale if scale > 1.0e-12 else np.zeros_like(weights)


def _segment_positions(segment_ids: np.ndarray) -> list[np.ndarray]:
    if len(segment_ids) == 0:
        return []
    chunks: list[np.ndarray] = []
    start = 0
    for end in range(1, len(segment_ids) + 1):
        if end == len(segment_ids) or segment_ids[end] != segment_ids[start]:
            chunks.append(np.arange(start, end, dtype=np.int64))
            start = end
    return chunks


def circular_block_resample_indices(segment_ids: np.ndarray, block_steps: int,
                                    rng: np.random.Generator) -> np.ndarray:
    """Sample exactly n observations with circular blocks inside one segment."""
    if block_steps < 1:
        raise ContractError("bootstrap block_steps must be positive")
    chunks = _segment_positions(np.asarray(segment_ids))
    if not chunks:
        return np.empty(0, dtype=np.int64)
    lengths = np.asarray([len(chunk) for chunk in chunks], dtype=np.float64)
    probabilities = lengths / lengths.sum()
    drawn: list[np.ndarray] = []
    remaining = int(lengths.sum())
    while remaining > 0:
        chunk = chunks[int(rng.choice(len(chunks), p=probabilities))]
        start = int(rng.integers(len(chunk)))
        width = min(block_steps, remaining)
        offsets = (start + np.arange(width, dtype=np.int64)) % len(chunk)
        drawn.append(chunk[offsets])
        remaining -= width
    return np.concatenate(drawn)


def circular_block_bootstrap_comparison(model_probability: np.ndarray, baseline_probability: np.ndarray,
                                        labels: np.ndarray, segment_ids: np.ndarray, block_steps: int,
                                        samples: int, seed: int) -> dict[str, dict[str, float]]:
    """Exact-length circular-block CIs for Brier, log loss, and calibration."""
    model = np.asarray(model_probability, dtype=np.float64)
    baseline = np.asarray(baseline_probability, dtype=np.float64)
    outcome = np.asarray(labels, dtype=np.float64)
    if len(model) < 2 or len(model) != len(baseline) or len(model) != len(outcome) or len(model) != len(segment_ids):
        return {name: {"point": float("nan"), "lower_95": float("nan"), "upper_95": float("nan")}
                for name in ("brier_improvement", "log_loss_improvement", "calibration_improvement")}
    if samples < 1:
        raise ContractError("bootstrap samples must be positive")
    point = np.array([
        brier_score(baseline, outcome) - brier_score(model, outcome),
        log_loss(baseline, outcome) - log_loss(model, outcome),
        calibration_error(baseline, outcome) - calibration_error(model, outcome),
    ], dtype=np.float64)
    estimates = np.empty((samples, 3), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for draw in range(samples):
        indices = circular_block_resample_indices(segment_ids, block_steps, rng)
        sampled_labels = outcome[indices]
        estimates[draw] = (
            brier_score(baseline[indices], sampled_labels) - brier_score(model[indices], sampled_labels),
            log_loss(baseline[indices], sampled_labels) - log_loss(model[indices], sampled_labels),
            calibration_error(baseline[indices], sampled_labels) - calibration_error(model[indices], sampled_labels),
        )
    result: dict[str, dict[str, float]] = {}
    for column, name in enumerate(("brier_improvement", "log_loss_improvement", "calibration_improvement")):
        result[name] = {
            "point": float(point[column]),
            "lower_95": float(np.quantile(estimates[:, column], 0.05)),
            "upper_95": float(np.quantile(estimates[:, column], 0.95)),
        }
    return result


class CausalFactorState:
    """Online two-regime factor, graph, AR, and residual-level state."""

    def __init__(self, config: ArenaConfig, transform: np.ndarray) -> None:
        if not 1 <= config.factor_count < N_PAIRS:
            raise ContractError("factor_count must be between one and nine")
        for parameters in (config.low_regime, config.high_regime):
            _alpha(parameters.covariance_half_life_steps)
            _alpha(parameters.residual_half_life_steps)
            if not 0.0 < parameters.level_ar < 1.0 or parameters.factor_refresh_steps < 1:
                raise ContractError("regime AR and refresh parameters are invalid")
            if not 0 <= parameters.neutral_factor_count <= config.factor_count:
                raise ContractError("regime neutral factor count is invalid")
        self.config = config
        self.transform = transform
        self.parameters = (config.low_regime, config.high_regime)
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.means = [np.zeros(N_PAIRS, dtype=np.float64) for _ in range(2)]
        self.covariances = [np.eye(N_PAIRS, dtype=np.float64) * 1.0e-8 for _ in range(2)]
        self.loadings_by_regime = [np.eye(N_PAIRS, self.config.factor_count, dtype=np.float64) for _ in range(2)]
        self.residual_variances = [np.full(N_PAIRS, 1.0e-8, dtype=np.float64) for _ in range(2)]
        self.ar_numerators = [np.zeros(N_PAIRS, dtype=np.float64) for _ in range(2)]
        self.ar_denominators = [np.full(N_PAIRS, 1.0e-12, dtype=np.float64) for _ in range(2)]
        self.previous_residuals: list[np.ndarray | None] = [None, None]
        self.partial_correlations = [np.zeros((N_PAIRS, N_PAIRS), dtype=np.float64) for _ in range(2)]
        self.graph_pressures = [np.zeros(N_PAIRS, dtype=np.float64) for _ in range(2)]
        self.graph_cluster_sizes = [np.ones(N_PAIRS, dtype=np.int64) for _ in range(2)]
        self.graph_edges: list[list[tuple[int, int, float]]] = [[], []]
        self.residual_level = np.zeros(N_PAIRS, dtype=np.float64)
        self.log_vol_mean = 0.0
        self.log_vol_variance = 1.0
        self.regime_high_probability = 0.5
        self.previous_weights = np.zeros(N_PAIRS, dtype=np.float64)

    def _update_regime_probability(self, transformed_return: np.ndarray) -> float:
        log_volatility = math.log(max(float(np.sqrt(np.mean(transformed_return * transformed_return))), 1.0e-12))
        prior_scale = math.sqrt(max(self.log_vol_variance, 1.0e-8))
        volatility_z = float(np.clip((log_volatility - self.log_vol_mean) / prior_scale, -8.0, 8.0))
        transition_high = 0.985 * self.regime_high_probability + 0.010 * (1.0 - self.regime_high_probability)
        low_likelihood = math.exp(-0.5 * (volatility_z + 0.55) ** 2)
        high_likelihood = math.exp(-0.5 * (volatility_z - 0.55) ** 2)
        normalization = (1.0 - transition_high) * low_likelihood + transition_high * high_likelihood
        self.regime_high_probability = (transition_high * high_likelihood / normalization
                                        if normalization > 1.0e-300 else 0.5)
        alpha = _alpha(self.parameters[0].residual_half_life_steps)
        delta = log_volatility - self.log_vol_mean
        self.log_vol_mean += alpha * delta
        self.log_vol_variance = ((1.0 - alpha) * self.log_vol_variance
                                 + alpha * delta * (log_volatility - self.log_vol_mean))
        return volatility_z

    def _refresh_factor_and_graph(self, regime: int) -> None:
        covariance = self.covariances[regime]
        diagonal_scale = max(float(np.trace(covariance) / N_PAIRS), 1.0e-12)
        regularized = ((1.0 - self.config.covariance_shrinkage) * covariance
                       + self.config.covariance_shrinkage * np.eye(N_PAIRS) * diagonal_scale)
        values, vectors = np.linalg.eigh(regularized)
        self.loadings_by_regime[regime] = vectors[:, np.argsort(values)[::-1][:self.config.factor_count]]
        precision = np.linalg.pinv(regularized, rcond=1.0e-12)
        precision_diagonal = np.maximum(np.diag(precision), 1.0e-18)
        partial = -precision / np.sqrt(np.outer(precision_diagonal, precision_diagonal))
        np.fill_diagonal(partial, 0.0)
        partial = np.clip(partial, -1.0, 1.0)
        self.partial_correlations[regime] = partial
        self.graph_pressures[regime] = np.minimum(np.sum(np.abs(partial), axis=1) / max(N_PAIRS - 1, 1), 1.0)
        parent = list(range(N_PAIRS))

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        edges: list[tuple[int, int, float]] = []
        for left in range(N_PAIRS):
            for right in range(left + 1, N_PAIRS):
                value = float(partial[left, right])
                if abs(value) >= self.config.graph_partial_correlation_threshold:
                    edges.append((left, right, value))
                    union(left, right)
        roots = [find(node) for node in range(N_PAIRS)]
        self.graph_cluster_sizes[regime] = np.asarray([roots.count(root) for root in roots], dtype=np.int64)
        self.graph_edges[regime] = edges

    def update(self, raw_return: np.ndarray) -> tuple[dict[str, Any], list[tuple[int, int, float]]]:
        if raw_return.shape != (N_PAIRS,) or not np.isfinite(raw_return).all():
            raise ContractError("factor state requires a finite ten-pair return vector")
        transformed_return = self.transform @ raw_return
        volatility_z = self._update_regime_probability(transformed_return)
        posterior = (1.0 - self.regime_high_probability, self.regime_high_probability)
        self.steps += 1
        residuals: list[np.ndarray] = []
        factors_by_regime: list[np.ndarray] = []
        for regime, parameters in enumerate(self.parameters):
            alpha_covariance = _alpha(parameters.covariance_half_life_steps) * posterior[regime]
            delta = transformed_return - self.means[regime]
            self.means[regime] += alpha_covariance * delta
            self.covariances[regime] = ((1.0 - alpha_covariance) * self.covariances[regime]
                                        + alpha_covariance * np.outer(delta, transformed_return - self.means[regime]))
            self.covariances[regime] = 0.5 * (self.covariances[regime] + self.covariances[regime].T)
            if self.steps == 1 or self.steps % parameters.factor_refresh_steps == 0:
                self._refresh_factor_and_graph(regime)
            centered = transformed_return - self.means[regime]
            factors = self.loadings_by_regime[regime].T @ centered
            residual = centered - self.loadings_by_regime[regime] @ factors
            alpha_residual = _alpha(parameters.residual_half_life_steps) * posterior[regime]
            self.residual_variances[regime] = ((1.0 - alpha_residual) * self.residual_variances[regime]
                                               + alpha_residual * residual * residual)
            previous = self.previous_residuals[regime]
            if previous is not None:
                self.ar_numerators[regime] = ((1.0 - alpha_residual) * self.ar_numerators[regime]
                                               + alpha_residual * previous * residual)
                self.ar_denominators[regime] = ((1.0 - alpha_residual) * self.ar_denominators[regime]
                                                 + alpha_residual * previous * previous)
            self.previous_residuals[regime] = residual.copy()
            residuals.append(residual)
            factors_by_regime.append(factors)
        active_regime = int(self.regime_high_probability >= 0.5)
        parameters = self.parameters[active_regime]
        residual = residuals[active_regime]
        residual_scale = np.sqrt(np.maximum(self.residual_variances[active_regime], 1.0e-16))
        residual_innovation = residual / residual_scale
        self.residual_level = parameters.level_ar * self.residual_level + residual_innovation
        beta = np.clip(self.ar_numerators[active_regime] / np.maximum(self.ar_denominators[active_regime], 1.0e-16),
                       -0.9999, 0.9999)
        half_life = np.full(N_PAIRS, np.inf, dtype=np.float64)
        positive = (beta > 1.0e-6) & (beta < 0.999)
        half_life[positive] = -math.log(2.0) / np.log(beta[positive])
        edges = self.graph_edges[active_regime] if self.steps == 1 or self.steps % parameters.factor_refresh_steps == 0 else []
        return {
            "residual": residual,
            "residual_scale": residual_scale,
            "residual_innovation": residual_innovation,
            "residual_level": self.residual_level.copy(),
            "beta": beta,
            "half_life": half_life,
            "factors": factors_by_regime[active_regime],
            "factor_volatility": float(np.sqrt(np.mean(factors_by_regime[active_regime] ** 2))),
            "residual_volatility": float(np.sqrt(np.mean(residual ** 2))),
            "volatility_z": volatility_z,
            "regime_high_probability": self.regime_high_probability,
            "active_regime": "high" if active_regime else "low",
            "active_regime_index": active_regime,
            "parameters": parameters,
            "mean": self.means[active_regime].copy(),
            "loadings": self.loadings_by_regime[active_regime].copy(),
            "graph_pressure": self.graph_pressures[active_regime].copy(),
            "graph_cluster_size": self.graph_cluster_sizes[active_regime].copy(),
        }, edges


def prediction_from_features(features: dict[str, Any]) -> dict[str, Any]:
    """Predeclared regime/graph-aware residual-level forecast and entry contract."""
    level = np.asarray(features["residual_level"], dtype=np.float64)
    pressure = np.asarray(features["graph_pressure"], dtype=np.float64)
    cluster_size = np.asarray(features["graph_cluster_size"], dtype=np.float64)
    score = np.abs(level) / (1.0 + 0.75 * pressure + 0.10 * np.maximum(cluster_size - 1.0, 0.0))
    selected = int(np.argmax(score))
    parameters: RegimeParameters = features["parameters"]
    selected_level = float(level[selected])
    selected_pressure = float(pressure[selected])
    selected_cluster = int(cluster_size[selected])
    entry_eligible = bool(abs(selected_level) >= parameters.entry_abs_level)
    beta = np.asarray(features["beta"], dtype=np.float64)
    unstable = float(np.mean((beta <= 0.0) | (beta >= 0.995)))
    breakdown = sigmoid(-2.10 + 1.20 * float(features["regime_high_probability"])
                        + 0.30 * min(abs(selected_level), 8.0) + 1.00 * selected_pressure
                        + 0.15 * max(selected_cluster - 1, 0) + 1.50 * unstable)
    half_life = float(np.asarray(features["half_life"])[selected])
    speed = math.log1p(parameters.holding_horizon_steps / half_life) if math.isfinite(half_life) and half_life > 0 else 0.0
    probability = sigmoid(-0.40 + 0.28 * min(abs(selected_level), 8.0) + 0.25 * speed
                          - 2.25 * breakdown - 0.65 * float(features["regime_high_probability"]))
    signal_basis = np.zeros(N_PAIRS, dtype=np.float64)
    signal_basis[selected] = -float(np.sign(selected_level) or 1.0)
    scale = max(0.0, min(1.0, (abs(selected_level) - parameters.entry_abs_level) /
                         max(parameters.stop_multiple * parameters.entry_abs_level, 1.0e-12)))
    return {
        "selected": selected,
        "probability": probability,
        "breakdown": breakdown,
        "signal_basis": signal_basis,
        "entry_eligible": entry_eligible,
        "position_scale": scale * (1.0 - breakdown) if entry_eligible else 0.0,
        "selected_level": selected_level,
        "selected_graph_pressure": selected_pressure,
        "selected_cluster_size": selected_cluster,
    }


def evaluate_frozen_residual_target(times: np.ndarray, log_prices: np.ndarray, transform: np.ndarray,
                                    entry: FrozenResidualTarget) -> dict[str, Any] | None:
    """Observe a fixed-horizon residual-level path using only the frozen entry model."""
    target_index = entry.source_index + entry.horizon_steps
    if target_index >= len(times):
        return None
    for index in range(entry.source_index + 1, target_index + 1):
        if not contiguous_60s(int(times[index - 1]), int(times[index]), DT_NS):
            return None
    level = entry.entry_level
    levels: list[float] = []
    innovations: list[float] = []
    basket_returns: list[float] = []
    for index in range(entry.source_index + 1, target_index + 1):
        raw_return = log_prices[index] - log_prices[index - 1]
        basket_returns.append(float(entry.raw_basket_weights @ raw_return))
        centered = transform @ raw_return - entry.mean_basis
        residual = centered - entry.loadings_basis @ (entry.loadings_basis.T @ centered)
        innovation = float(residual[entry.selected] / max(entry.residual_scale[entry.selected], 1.0e-12))
        level = entry.level_ar * level + innovation
        innovations.append(innovation)
        levels.append(level)
    path = np.asarray(levels, dtype=np.float64)
    entry_sign = float(np.sign(entry.entry_level))
    gross_convergence = -entry_sign * (float(path[-1]) - entry.entry_level)
    path_gross = -entry_sign * (path - entry.entry_level)
    time_to_zero = next((offset + 1 for offset, value in enumerate(path) if value * entry_sign <= 0.0), None)
    removed = 100.0 * (1.0 - abs(float(path[-1])) / max(abs(entry.entry_level), 1.0e-12))
    breakdown = bool(np.max(np.abs(path)) > entry.stop_multiple * abs(entry.entry_level))
    return {
        "target_time": pd.Timestamp(int(times[target_index]), unit="ns", tz="UTC"),
        "convergence_label": int(gross_convergence > 0.0 and abs(float(path[-1])) < abs(entry.entry_level)),
        "gross_convergence": gross_convergence,
        "mae_level": float(max(0.0, -np.min(path_gross))),
        "time_to_zero_steps": float(time_to_zero) if time_to_zero is not None else float("nan"),
        "percentage_displacement_removed": removed,
        "breakdown_label": int(breakdown),
        "target_path_volatility": float(np.std(innovations, ddof=0)),
        "frozen_target_level": float(path[-1]),
        "frozen_basket_cumulative_log_return": float(np.sum(basket_returns)),
        "frozen_basket_path_volatility": float(np.std(basket_returns, ddof=0)),
    }


def _level_bin(values: pd.Series) -> pd.Series:
    return pd.Series(np.digitize(np.abs(values.to_numpy(dtype=np.float64)), [0.75, 1.50, 2.50, 4.00]),
                     index=values.index, dtype=np.int8)


def frozen_conditional_climatology(train: pd.DataFrame, target: pd.DataFrame,
                                   minimum_cell_count: int = 20) -> tuple[np.ndarray, list[str]]:
    """Fit only on train labels; predict conditional convergence climatology."""
    if train.empty or target.empty:
        return np.full(len(target), float("nan")), []
    source = train.copy()
    source["_level_bin"] = _level_bin(source["entry_residual_level"])
    destination = target.copy()
    destination["_level_bin"] = _level_bin(destination["entry_residual_level"])

    def table(columns: list[str]) -> dict[tuple[Any, ...], tuple[int, int]]:
        grouped = source.groupby(columns, dropna=False)["convergence_label"].agg(["sum", "count"])
        return {key if isinstance(key, tuple) else (key,): (int(row["sum"]), int(row["count"]))
                for key, row in grouped.iterrows()}

    detailed_columns = ["_level_bin", "selected_component_index", "session_bucket", "active_regime"]
    component_columns = ["_level_bin", "selected_component_index", "active_regime"]
    regime_columns = ["_level_bin", "active_regime"]
    detailed, component, regime = (table(detailed_columns), table(component_columns), table(regime_columns))
    global_probability = float(source["convergence_label"].mean())
    predictions: list[float] = []
    tiers: list[str] = []
    for _, row in destination.iterrows():
        key_sets = (
            (tuple(row[column] for column in detailed_columns), detailed, "component_session_regime"),
            (tuple(row[column] for column in component_columns), component, "component_regime"),
            (tuple(row[column] for column in regime_columns), regime, "level_regime"),
        )
        selected = next(((wins / count, tier) for key, values, tier in key_sets
                         for wins, count in [values.get(key, (0, 0))] if count >= minimum_cell_count),
                        (global_probability, "global"))
        predictions.append(float(selected[0]))
        tiers.append(str(selected[1]))
    return np.asarray(predictions, dtype=np.float64), tiers


def run_arrays(times: np.ndarray, log_prices: np.ndarray, test_start_index: int,
               config: ArenaConfig = ArenaConfig()) -> ArenaResult:
    """Run the frozen v0.2 arena over supplied synchronous log-close observations."""
    if times.ndim != 1 or log_prices.shape != (len(times), N_PAIRS):
        raise ContractError("times/log_prices do not match the ten-pair arena contract")
    if len(times) <= config.warmup_steps + config.max_horizon_steps + 2:
        raise ContractError("research window is too short for warmup and target horizon")
    if not np.all(np.diff(times) > 0) or not np.isfinite(log_prices).all():
        raise ContractError("research input must be strictly increasing and finite")
    if not config.warmup_steps < test_start_index < len(times) - config.max_horizon_steps:
        raise ContractError("test_start_index does not leave train and OOS target observations")
    transform, inverse_transform, component_names = identity_free_transform()
    incidence, currencies = currency_incidence()
    state = CausalFactorState(config, transform)
    records: list[dict[str, Any]] = []
    frozen_targets: list[FrozenResidualTarget | None] = []
    graph_records: list[dict[str, Any]] = []
    segment_id = 0
    gap_resets = 0
    max_contiguous_state_steps = 0
    for index in range(1, len(times)):
        if not contiguous_60s(int(times[index - 1]), int(times[index]), DT_NS):
            state.reset()
            segment_id += 1
            gap_resets += 1
            continue
        features, edges = state.update(log_prices[index] - log_prices[index - 1])
        max_contiguous_state_steps = max(max_contiguous_state_steps, state.steps)
        for left, right, partial in edges:
            graph_records.append({
                "timestamp": pd.Timestamp(int(times[index]), unit="ns", tz="UTC"),
                "segment_id": segment_id,
                "active_regime": features["active_regime"],
                "left_component": component_names[left],
                "right_component": component_names[right],
                "partial_correlation": partial,
            })
        if state.steps < config.warmup_steps:
            continue
        decision = prediction_from_features(features)
        parameters: RegimeParameters = features["parameters"]
        unit_weights = factor_neutral_weights(
            decision["signal_basis"], transform, inverse_transform, features["loadings"], incidence,
            parameters.neutral_factor_count,
        )
        weights = unit_weights * float(decision["position_scale"])
        turnover = float(np.sum(np.abs(weights - state.previous_weights)))
        state.previous_weights = weights
        raw_factor_directions = inverse_transform @ np.asarray(features["loadings"])
        factor_exposure = raw_factor_directions[:, :parameters.neutral_factor_count].T @ weights
        currency_exposure = incidence.T @ weights
        selected = int(decision["selected"])
        entry_eligible = bool(decision["entry_eligible"] and np.sum(np.abs(unit_weights)) > 0.0)
        record: dict[str, Any] = {
            "source_index": index,
            "timestamp": pd.Timestamp(int(times[index]), unit="ns", tz="UTC"),
            "segment_id": segment_id,
            "selected_component_index": selected,
            "selected_component": component_names[selected],
            "active_regime": features["active_regime"],
            "high_volatility_regime_probability": float(features["regime_high_probability"]),
            "session_bucket": int(pd.Timestamp(int(times[index]), unit="ns", tz="UTC").hour // 6),
            "p_convergence": float(decision["probability"]),
            "breakdown_probability": float(decision["breakdown"]),
            "entry_eligible": entry_eligible,
            "entry_residual_level": float(decision["selected_level"]),
            "entry_abs_level_threshold": parameters.entry_abs_level,
            "holding_horizon_steps": parameters.holding_horizon_steps,
            "stop_multiple": parameters.stop_multiple,
            "neutral_factor_count": parameters.neutral_factor_count,
            "diagnostic_position_scale": float(decision["position_scale"] if entry_eligible else 0.0),
            "diagnostic_gross_exposure_l1": float(np.sum(np.abs(weights))),
            "turnover_l1_diagnostic": turnover,
            "selected_graph_pressure": float(decision["selected_graph_pressure"]),
            "selected_graph_cluster_size": int(decision["selected_cluster_size"]),
            "factor_volatility": float(features["factor_volatility"]),
            "residual_volatility": float(features["residual_volatility"]),
            "volatility_z": float(features["volatility_z"]),
            "target_time": pd.NaT,
            "convergence_label": np.nan,
            "gross_convergence": np.nan,
            "mae_level": np.nan,
            "time_to_zero_steps": np.nan,
            "percentage_displacement_removed": np.nan,
            "breakdown_label": np.nan,
            "target_path_volatility": np.nan,
            "frozen_target_level": np.nan,
            "frozen_basket_cumulative_log_return": np.nan,
            "frozen_basket_path_volatility": np.nan,
        }
        for pair, value in zip(PAIRS, weights, strict=True):
            record[f"diagnostic_weight_{pair.lower()}"] = float(value)
        for currency, value in zip(currencies, currency_exposure, strict=True):
            record[f"currency_incidence_exposure_{currency.lower()}"] = float(value)
        for factor, value in enumerate(factor_exposure):
            record[f"selected_factor_exposure_{factor}"] = float(value)
        for name, value in zip(component_names, np.asarray(features["residual_level"]), strict=True):
            record[f"residual_level_{name.lower()}"] = float(value)
        records.append(record)
        frozen_targets.append(FrozenResidualTarget(
            source_index=index,
            segment_id=segment_id,
            selected=selected,
            entry_level=float(decision["selected_level"]),
            mean_basis=np.asarray(features["mean"]).copy(),
            loadings_basis=np.asarray(features["loadings"]).copy(),
            residual_scale=np.asarray(features["residual_scale"]).copy(),
            level_ar=parameters.level_ar,
            horizon_steps=parameters.holding_horizon_steps,
            stop_multiple=parameters.stop_multiple,
            raw_basket_weights=weights.copy(),
        ) if entry_eligible else None)
    if not records:
        raise ContractError("no causal feature emissions were produced")
    for record, target in zip(records, frozen_targets, strict=True):
        if target is None:
            continue
        outcome = evaluate_frozen_residual_target(times, log_prices, transform, target)
        if outcome is not None:
            record.update(outcome)
    emissions = pd.DataFrame(records)
    labelled = emissions[emissions["convergence_label"].notna()].copy()
    labelled["convergence_label"] = labelled["convergence_label"].astype(np.int8)
    train = labelled[labelled["source_index"] + labelled["holding_horizon_steps"] < test_start_index].copy()
    oos = labelled[labelled["source_index"] >= test_start_index].copy()
    if len(train) < 100 or len(oos) < 100:
        raise ContractError("train/OOS target partitions require at least 100 valid frozen entries each")
    frozen_train_prior = float(train["convergence_label"].mean())
    conditional_probability, conditional_tiers = frozen_conditional_climatology(train, oos)
    oos_probability = oos["p_convergence"].to_numpy(dtype=np.float64)
    oos_label = oos["convergence_label"].to_numpy(dtype=np.float64)
    prior_probability = np.full(len(oos), frozen_train_prior, dtype=np.float64)
    emissions["p_conditional_climatology"] = np.nan
    emissions["conditional_climatology_tier"] = pd.NA
    emissions.loc[oos.index, "p_conditional_climatology"] = conditional_probability
    emissions.loc[oos.index, "conditional_climatology_tier"] = conditional_tiers
    bootstrap_sensitivity = {
        f"{block_steps}_steps": circular_block_bootstrap_comparison(
            oos_probability, conditional_probability, oos_label,
            oos["segment_id"].to_numpy(dtype=np.int64), block_steps, config.bootstrap_samples,
            config.bootstrap_seed + block_steps,
        )
        for block_steps in config.bootstrap_block_sensitivity_steps
    }
    primary_bootstrap = bootstrap_sensitivity[f"{config.bootstrap_block_sensitivity_steps[-1]}_steps"]
    shuffled_labels = oos_label[np.random.default_rng(config.bootstrap_seed).permutation(len(oos_label))]
    prediction_pass = bool(
        brier_score(oos_probability, oos_label) < brier_score(conditional_probability, oos_label)
        and primary_bootstrap["brier_improvement"]["lower_95"] > 0.0
    )
    tier_counts = pd.Series(conditional_tiers, dtype="string").value_counts().to_dict()
    summary: dict[str, Any] = {
        "version": VERSION,
        "interpretation": "causal classical residual-level research only; no execution, PnL, capacity, or market-impact claim",
        "pair_scope": list(PAIRS),
        "target_contract": {
            "target": "frozen selected residual level converges after its frozen entry-specific holding horizon",
            "gross_convergence": "-sign(S_t) * (S_{t+h,frozen} - S_t)",
            "residual_level": "S_t = rho_regime * S_(t-1) + e_t / frozen_residual_scale",
            "frozen_at_entry": ["factor_mean", "factor_loadings", "residual_scale", "selected_component", "basket_weights", "level_ar", "horizon", "stop"],
            "target_available_only_after": "all entry-to-horizon arrivals are observed contiguous one-minute bars",
            "outcome_diagnostics": ["MAE", "time_to_zero", "percentage_displacement_removed", "breakdown", "path_volatility", "turnover"],
            "target_excluded_from_feature_state": True,
        },
        "causality": {
            "input_at_t": "synchronous canonical BID log returns through bar t only",
            "gap_policy": "reset all factor/regime/residual-level state on observed non-60-second arrival",
            "gap_resets": gap_resets,
            "max_contiguous_state_steps": max_contiguous_state_steps,
            "post_gap_warmup_steps": config.warmup_steps,
            "identity_control": "EURGBP/EURJPY/GBPJPY transformed to arithmetic-residual channels before factor/graph estimation",
        },
        "components": {
            "regime_switching": "posterior-weighted low/high covariance, loadings, residual variance, AR state, refresh cadence, thresholds, holding horizon, stop, factor-neutrality count, and diagnostic position sizing",
            "sparse_interaction_graph": "shrunken precision partial-correlation graph changes residual selection score, breakdown probability, and diagnostic position scale",
            "neutrality": "raw signal T.T @ s; raw factor directions inverse(T) @ B; D.T @ w=0 currency-incidence and selected-factor constraints",
            "currency_units_caveat": "incidence neutrality is in pair coefficient units only; no dollar-risk sizing without contract notionals and conversion prices",
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
            "model": {"brier_score": brier_score(oos_probability, oos_label), "log_loss": log_loss(oos_probability, oos_label), "calibration_error": calibration_error(oos_probability, oos_label)},
            "frozen_train_prior": {"probability": frozen_train_prior, "brier_score": brier_score(prior_probability, oos_label), "log_loss": log_loss(prior_probability, oos_label), "calibration_error": calibration_error(prior_probability, oos_label)},
            "frozen_conditional_climatology": {"brier_score": brier_score(conditional_probability, oos_label), "log_loss": log_loss(conditional_probability, oos_label), "calibration_error": calibration_error(conditional_probability, oos_label), "tier_counts": {str(key): int(value) for key, value in tier_counts.items()}},
            "time_shuffled_residual_placebo": {"model_brier_score": brier_score(oos_probability, shuffled_labels), "conditional_climatology_brier_score": brier_score(conditional_probability, shuffled_labels), "seed": config.bootstrap_seed},
            "circular_block_bootstrap": {"samples": config.bootstrap_samples, "exact_replicate_length": int(len(oos)), "within_uninterrupted_segments": True, "sensitivity": bootstrap_sensitivity},
            "passes_conditional_climatology_prediction_gate": prediction_pass,
        },
        "promotion_status": "REJECTED_NO_EXECUTION_DATA_AND_NO_NEW_UNTOUCHED_HOLDOUT: v0.2 is frozen pending post-2024 data",
    }
    return ArenaResult(emissions=emissions, graph=pd.DataFrame(graph_records), summary=summary)


def load_common_log_prices(max_rows: int, test_year: int | None) -> tuple[np.ndarray, np.ndarray, int]:
    """Load a canonical window only after an explicit burned-holdout acknowledgement."""
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
        filters = [("timestamp", ">=", reference.iloc[-max_rows - 1]["timestamp"])]
    joined: pd.DataFrame | None = None
    for pair in PAIRS:
        frame = pd.read_parquet(CANONICAL_DIR / f"{pair}.parquet", columns=["timestamp", "close"], filters=filters)
        frame = frame.rename(columns={"close": pair})
        joined = frame if joined is None else joined.merge(frame, on="timestamp", how="inner", validate="one_to_one")
    assert joined is not None
    joined = joined.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if test_year is None and len(joined) > max_rows:
        joined = joined.iloc[-max_rows:].reset_index(drop=True)
    times = epoch_ns(joined["timestamp"])
    prices = joined.loc[:, list(PAIRS)].to_numpy(dtype=np.float64)
    if len(times) < 2_000 or not np.all(np.diff(times) > 0) or not np.isfinite(prices).all() or np.any(prices <= 0.0):
        raise ContractError("joined canonical stat-arb input is invalid")
    test_start_index = int(len(times) * 0.70) if test_start is None else int(np.searchsorted(times, test_start.value, side="left"))
    return times, np.log(prices), test_start_index


def write_result(result: ArenaResult, out_dir: Path, label: str) -> dict[str, str]:
    """Write only versioned v0.2 artifacts; v0.1 archives are never overwritten."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"stat_arb_v0_2_{label}"
    emissions_path = out_dir / f"{prefix}_minute.parquet"
    graph_path = out_dir / f"{prefix}_graph.parquet"
    daily_path = out_dir / f"{prefix}_daily.parquet"
    summary_path = out_dir / f"{prefix}_summary.json"
    result.emissions.to_parquet(emissions_path, index=False, compression="zstd")
    result.graph.to_parquet(graph_path, index=False, compression="zstd")
    daily = (result.emissions.assign(day=result.emissions["timestamp"].dt.floor("D"))
             .groupby("day", as_index=False)
             .agg(emissions=("source_index", "size"), mean_p_convergence=("p_convergence", "mean"),
                  diagnostic_turnover_l1=("turnover_l1_diagnostic", "sum")))
    daily.to_parquet(daily_path, index=False, compression="zstd")
    result.summary["outputs"] = {"minute": str(emissions_path.relative_to(ROOT)).replace("\\", "/"), "graph": str(graph_path.relative_to(ROOT)).replace("\\", "/"), "daily": str(daily_path.relative_to(ROOT)).replace("\\", "/")}
    result.summary["source_hashes"] = {"pipeline/stat_arb.py": sha256_file(Path(__file__).resolve()), **{f"data_canonical/{pair}.parquet": sha256_file(CANONICAL_DIR / f"{pair}.parquet") for pair in PAIRS}}
    summary_path.write_text(json.dumps(result.summary, indent=2) + "\n", encoding="utf-8")
    return {"minute": str(emissions_path), "graph": str(graph_path), "daily": str(daily_path), "summary": str(summary_path)}


def synthetic_input(rows: int = 900) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12345)
    returns = rng.normal(0.0, 2.0e-4, size=(rows, N_PAIRS))
    returns += (0.00015 * np.sin(np.arange(rows) / 19.0))[:, None]
    returns[:, PAIRS.index("EURGBP")] = returns[:, PAIRS.index("EURUSD")] - returns[:, PAIRS.index("GBPUSD")] + rng.normal(0.0, 1e-5, rows)
    returns[:, PAIRS.index("EURJPY")] = returns[:, PAIRS.index("EURUSD")] + returns[:, PAIRS.index("USDJPY")] + rng.normal(0.0, 1e-5, rows)
    returns[:, PAIRS.index("GBPJPY")] = returns[:, PAIRS.index("GBPUSD")] + returns[:, PAIRS.index("USDJPY")] + rng.normal(0.0, 1e-5, rows)
    times = np.arange(rows, dtype=np.int64) * DT_NS
    times[500:] += 2 * DT_NS
    return times, np.cumsum(returns, axis=0)


def _fast_config() -> ArenaConfig:
    return ArenaConfig(
        warmup_steps=64,
        bootstrap_samples=32,
        bootstrap_block_sensitivity_steps=(8, 16, 32),
        low_regime=RegimeParameters(32.0, 32.0, 8, 0.94, 0.50, 8, 2.0, 2),
        high_regime=RegimeParameters(16.0, 16.0, 4, 0.82, 0.75, 6, 1.35, 1),
    )


def self_check() -> dict[str, Any]:
    times, log_prices = synthetic_input()
    result = run_arrays(times, log_prices, test_start_index=500, config=_fast_config())
    altered = log_prices.copy()
    altered[700:] += 0.02
    altered_result = run_arrays(times, altered, test_start_index=500, config=_fast_config())
    comparable = result.emissions[result.emissions["source_index"] < 680].set_index("source_index")
    altered_comparable = altered_result.emissions[altered_result.emissions["source_index"] < 680].set_index("source_index")
    common = comparable.index.intersection(altered_comparable.index)
    prefix_equal = bool(np.array_equal(comparable.loc[common, "p_convergence"], altered_comparable.loc[common, "p_convergence"]))
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
    parser.add_argument("--max-rows", type=int, default=50_000)
    parser.add_argument("--test-year", type=int, choices=OUTER_TEST_YEARS)
    parser.add_argument("--out-dir", type=Path, default=DERIVED_DIR)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--allow-burned-holdout-research", action="store_true", help="explicitly acknowledge that all available canonical data end in the burned 2024 holdout")
    args = parser.parse_args(argv)
    if args.self_check:
        result = self_check()
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    if not args.allow_burned_holdout_research:
        print("[FATAL] ContractError: v0.2 is frozen pending post-2024 data; use only --self-check or explicitly acknowledge a non-promotable burned-holdout research run", file=sys.stderr)
        return 1
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

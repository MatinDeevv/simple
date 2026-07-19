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

from engine.core.contracts import (ContractError as SharedContractError, canonical_pair_order,
                       contiguous_60s, validate_generated_manifest)
from engine.core.run_manifest import build_run_manifest, write_manifest
from engine.core.schema_validate import validate_instance
from engine.evaluation.entry_diagnostics import annotate_entry_diagnostics, summarize_entry_policy
from engine.evaluation.evaluation_protocol import build_evaluation_metadata


ROOT = Path(__file__).resolve().parents[3]
CANONICAL_DIR = ROOT / "data" / "canonical"
DERIVED_DIR = ROOT / "data" / "derived"
PAIRS = canonical_pair_order(ROOT)
N_PAIRS = len(PAIRS)
DT_NS = 60_000_000_000
VERSION = "stat-arb-arena-0.2.0-frozen"
OUTER_TEST_YEARS = (2022, 2023, 2024)
CYCLE_NEUTRAL_MODE = "cycle_neutral"
RELATIVE_VALUE_MODE = "relative_value"


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
    # Units are observed contiguous one-minute bars, not sparse eligible-entry
    # rows.  The bootstrap samples raw time ranges and includes their entries.
    bootstrap_block_sensitivity_minutes: tuple[int, ...] = (30, 240, 1_440)
    bootstrap_seed: int = 20260718
    # Cycle-neutral research is deliberately restricted to closed FX loops.
    # Relative-value mode keeps factor neutrality but caps, rather than erases,
    # currency incidence in pair-coefficient units.
    basket_mode: str = CYCLE_NEUTRAL_MODE
    relative_value_currency_exposure_budget: float = 0.35
    relative_value_max_weight: float = 0.45
    minimum_signal_preservation: float = 0.35
    maximum_basket_concentration: float = 0.75
    basket_neutral_zone_z: float = 0.25
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
class BasketOptimizationResult:
    weights: np.ndarray
    converged: bool
    iterations: int
    objective_value: float
    max_factor_violation: float
    max_currency_violation: float
    max_weight_violation: float
    gross_exposure_violation: float


@dataclass(frozen=True)
class FrozenResidualTarget:
    """t-measurable definition of a single residual-level outcome."""

    source_index: int
    segment_id: int
    selected: int
    entry_level: float
    entry_levels_by_regime: np.ndarray
    regime_probabilities: np.ndarray
    means_by_regime: np.ndarray
    loadings_by_regime: np.ndarray
    residual_scales_by_regime: np.ndarray
    level_ars: np.ndarray
    horizon_steps: int
    stop_multiple: float
    raw_basket_weights: np.ndarray
    basket_one_step_volatility: float
    basket_neutral_zone_z: float


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


def multiclass_log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1.0e-12, 1.0)
    rows = np.arange(len(labels), dtype=np.int64)
    return float(np.mean(-np.log(clipped[rows, np.asarray(labels, dtype=np.int64)])))


def multiclass_brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    one_hot = np.eye(3, dtype=np.float64)[np.asarray(labels, dtype=np.int64)]
    return float(np.mean(np.sum((np.asarray(probabilities, dtype=np.float64) - one_hot) ** 2, axis=1)))


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
                           neutral_factor_count: int = 2,
                           basket_mode: str = CYCLE_NEUTRAL_MODE,
                           relative_value_currency_exposure_budget: float = 0.35,
                           covariance: np.ndarray | None = None,
                           previous_weights: np.ndarray | None = None,
                           maximum_weight: float = 0.45) -> BasketOptimizationResult:
    """Construct either a closed-cycle or currency-budgeted diagnostic basket.

    ``T.T`` maps the residual signal correctly as a dual vector.  In contrast,
    factor loading columns represent primal directions, so their raw mapping is
    ``inverse(T) @ B``.  ``cycle_neutral`` imposes ``D.T @ w = 0`` and is
    intentionally limited to the three-dimensional closed-loop space.  In
    ``relative_value`` mode solves a constrained, L1-normalized projected
    utility problem; its currency budget alters the composition rather than
    merely scaling gross exposure. Both remain pair-coefficient diagnostics,
    never dollar-risk or execution-neutrality claims.
    """
    signal = np.asarray(signal_basis, dtype=np.float64)
    loadings = np.asarray(loadings_basis, dtype=np.float64)
    if signal.shape != (N_PAIRS,) or transform.shape != (N_PAIRS, N_PAIRS):
        raise ContractError("invalid residual signal transform")
    if inverse_transform.shape != (N_PAIRS, N_PAIRS) or loadings.shape[0] != N_PAIRS:
        raise ContractError("invalid factor basis for diagnostic weights")
    if not 0 <= neutral_factor_count <= loadings.shape[1]:
        raise ContractError("neutral_factor_count is outside the factor basis")
    if basket_mode not in (CYCLE_NEUTRAL_MODE, RELATIVE_VALUE_MODE):
        raise ContractError("basket_mode must be cycle_neutral or relative_value")
    if not 0.0 < relative_value_currency_exposure_budget <= 1.0:
        raise ContractError("relative-value currency exposure budget must be in (0, 1]")
    incidence = currency_incidence()[0] if currency_matrix is None else np.asarray(currency_matrix, dtype=np.float64)
    if incidence.shape[0] != N_PAIRS:
        raise ContractError("currency incidence does not match pair weights")
    raw_signal = basis_signal_to_raw_weights(signal, transform)
    raw_factor_directions = inverse_transform @ loadings
    selected_factors = raw_factor_directions[:, :neutral_factor_count]
    covariance = np.eye(N_PAIRS) if covariance is None else np.asarray(covariance, dtype=np.float64)
    previous = np.zeros(N_PAIRS) if previous_weights is None else np.asarray(previous_weights, dtype=np.float64)
    if covariance.shape != (N_PAIRS, N_PAIRS) or previous.shape != (N_PAIRS,):
        raise ContractError("relative-value optimization inputs have invalid shapes")

    def finish(candidate: np.ndarray, iterations: int) -> BasketOptimizationResult:
        candidate = np.asarray(candidate, dtype=np.float64)
        factor_violation = float(np.max(np.abs(selected_factors.T @ candidate))) if selected_factors.size else 0.0
        currency_limit = 0.0 if basket_mode == CYCLE_NEUTRAL_MODE else relative_value_currency_exposure_budget
        currency_violation = max(0.0, float(np.max(np.abs(incidence.T @ candidate))) - currency_limit)
        weight_violation = max(0.0, float(np.max(np.abs(candidate))) - maximum_weight)
        gross_violation = abs(float(np.sum(np.abs(candidate))) - 1.0)
        objective = float(raw_signal @ candidate - 0.35 * candidate @ covariance @ candidate
                          - 0.03 * np.sum(np.abs(candidate - previous)))
        feasible = max(factor_violation, currency_violation, weight_violation, gross_violation) <= 1.0e-8
        return BasketOptimizationResult(candidate, feasible, iterations, objective, factor_violation,
                                        currency_violation, weight_violation, gross_violation)
    constraints = (np.column_stack((incidence, selected_factors))
                   if basket_mode == CYCLE_NEUTRAL_MODE else selected_factors)
    if np.linalg.matrix_rank(constraints, tol=1.0e-12) >= N_PAIRS:
        return finish(np.zeros(N_PAIRS, dtype=np.float64), 0)
    projection = constraints @ (np.linalg.pinv(constraints, rcond=1.0e-12) @ raw_signal)
    weights = raw_signal - projection
    scale = float(np.sum(np.abs(weights)))
    if scale <= 1.0e-12:
        return finish(np.zeros_like(weights), 0)
    weights = weights / scale
    if basket_mode == CYCLE_NEUTRAL_MODE:
        return finish(weights, 1)
    factor_neutral_unit = weights.copy()
    # Projected proximal ascent of s'w - lambda w'Covw - gamma|w-w_prev|.
    # The alternating equality/currency/L1 projections are deterministic and
    # change direction when the currency half-space is binding.
    equality = selected_factors
    for _ in range(80):
        weights += 0.12 * (raw_signal - 0.35 * (covariance @ weights) - 0.03 * np.sign(weights - previous))
        if equality.size:
            weights -= equality @ (np.linalg.pinv(equality, rcond=1.0e-12) @ weights)
        weights = np.clip(weights, -maximum_weight, maximum_weight)
        exposure = incidence.T @ weights
        for column, value in enumerate(exposure):
            if abs(value) > relative_value_currency_exposure_budget:
                direction = incidence[:, column]
                weights -= ((value - math.copysign(relative_value_currency_exposure_budget, value)) /
                            max(float(direction @ direction), 1.0e-12)) * direction
        if equality.size:
            weights -= equality @ (np.linalg.pinv(equality, rcond=1.0e-12) @ weights)
        l1 = float(np.sum(np.abs(weights)))
        if l1 > 1.0e-12:
            weights /= l1
    # If alternating projections cannot satisfy a tight budget together with
    # unit gross, fall back to the feasible closed-incidence face.  This is a
    # composition change, never scalar shrinking of an infeasible basket.
    if float(np.max(np.abs(incidence.T @ weights))) > relative_value_currency_exposure_budget + 1.0e-10:
        raw_unit = factor_neutral_unit
        peak = float(np.max(np.abs(incidence.T @ raw_unit)))
        base = raw_unit * min(1.0, relative_value_currency_exposure_budget / max(peak, 1.0e-12))
        # Add an equality- and currency-neutral loop to restore gross one;
        # unlike scalar shrinking, this necessarily changes composition.
        strict = np.column_stack((incidence, selected_factors))
        seed = np.linspace(-1.0, 1.0, N_PAIRS)
        neutral_loop = seed - strict @ (np.linalg.pinv(strict, rcond=1.0e-12) @ seed)
        neutral_l1 = float(np.sum(np.abs(neutral_loop)))
        if neutral_l1 <= 1.0e-12:
            return finish(np.zeros_like(weights), 140)
        neutral_loop /= neutral_l1
        alpha_lo, alpha_hi = 0.0, 4.0
        for _ in range(60):
            alpha = 0.5 * (alpha_lo + alpha_hi)
            if float(np.sum(np.abs(base + alpha * neutral_loop))) < 1.0:
                alpha_lo = alpha
            else:
                alpha_hi = alpha
        weights = base + alpha_hi * neutral_loop
        weights = weights / max(float(np.sum(np.abs(weights))), 1.0e-12)
    result = finish(weights, 140 if basket_mode == RELATIVE_VALUE_MODE else 80)
    if result.converged:
        return result
    # Feasibility takes priority over signal utility. Search the closed
    # factor/currency-null space for a diversified feasible diagnostic basket.
    strict = np.column_stack((incidence, selected_factors))
    best: BasketOptimizationResult | None = None
    for phase in np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False):
        seed = np.sin(np.arange(N_PAIRS, dtype=np.float64) * 1.618 + phase)
        candidate = seed - strict @ (np.linalg.pinv(strict, rcond=1.0e-12) @ seed)
        candidate /= max(float(np.sum(np.abs(candidate))), 1.0e-12)
        evaluated = finish(candidate, 172)
        if evaluated.converged:
            return evaluated
        if best is None or max(evaluated.max_factor_violation, evaluated.max_currency_violation,
                               evaluated.max_weight_violation, evaluated.gross_exposure_violation) < max(
                                   best.max_factor_violation, best.max_currency_violation,
                                   best.max_weight_violation, best.gross_exposure_violation):
            best = evaluated
    return best if best is not None else result


def basket_projection_diagnostics(signal_basis: np.ndarray, transform: np.ndarray,
                                  unit_weights: np.ndarray) -> tuple[float, float]:
    """Quantify how much constrained basket construction changed the signal."""
    raw_signal = basis_signal_to_raw_weights(signal_basis, transform)
    raw_norm = float(np.linalg.norm(raw_signal))
    basket_norm = float(np.linalg.norm(unit_weights))
    if raw_norm <= 1.0e-12 or basket_norm <= 1.0e-12:
        return float("nan"), 0.0
    raw_unit = raw_signal / raw_norm
    basket_unit = np.asarray(unit_weights, dtype=np.float64) / basket_norm
    # For an orthogonal projection this cosine is the fraction of original
    # signal length retained; it remains interpretable after L1 normalization.
    retained_fraction = float(np.clip(abs(raw_unit @ basket_unit), 0.0, 1.0))
    distortion_l2 = float(np.linalg.norm(raw_unit - basket_unit))
    return distortion_l2, retained_fraction


def observed_minute_block_resample_indices(source_indices: np.ndarray, source_segment_ids: np.ndarray,
                                           timeline_times: np.ndarray, timeline_segment_ids: np.ndarray,
                                           block_minutes: int, rng: np.random.Generator) -> np.ndarray:
    """Resample eligible entries from real contiguous minute-duration blocks.

    Candidate starts are raw observed bars, never positions in the sparse
    labelled-entry frame.  Every selected entry belongs to an actual contiguous
    range no longer than ``block_minutes``.  Blocks near a session boundary are
    shorter rather than circularly wrapping across time.
    """
    source = np.asarray(source_indices, dtype=np.int64)
    source_segments = np.asarray(source_segment_ids, dtype=np.int64)
    timeline = np.asarray(timeline_times, dtype=np.int64)
    timeline_segments = np.asarray(timeline_segment_ids, dtype=np.int64)
    if block_minutes < 1:
        raise ContractError("bootstrap block_minutes must be positive")
    if len(source) == 0 or len(source) != len(source_segments):
        return np.empty(0, dtype=np.int64)
    if len(timeline) != len(timeline_segments) or np.any(source < 0) or np.any(source >= len(timeline)):
        raise ContractError("bootstrap source indices do not match the observed timeline")
    if not np.array_equal(timeline_segments[source], source_segments):
        raise ContractError("bootstrap source segment IDs do not match the observed timeline")
    segment_ranges: list[tuple[int, int, int]] = []
    start = 0
    for end in range(1, len(timeline) + 1):
        if end == len(timeline) or timeline_segments[end] != timeline_segments[start]:
            if np.any(source_segments == timeline_segments[start]):
                segment_ranges.append((int(timeline_segments[start]), start, end))
            start = end
    if not segment_ranges:
        return np.empty(0, dtype=np.int64)
    lengths = np.asarray([end - start for _segment, start, end in segment_ranges], dtype=np.float64)
    probabilities = lengths / lengths.sum()
    drawn: list[np.ndarray] = []
    remaining = len(source)
    attempts = 0
    max_attempts = max(10_000, len(source) * 200)
    while remaining > 0:
        attempts += 1
        if attempts > max_attempts:
            raise ContractError("time-block bootstrap could not draw enough eligible entries")
        segment, start, end = segment_ranges[int(rng.choice(len(segment_ranges), p=probabilities))]
        block_start = int(rng.integers(start, end))
        block_end = min(block_start + block_minutes, end)
        # The segment contract means this is a genuine elapsed observed-minute
        # interval, not an arbitrary group of entry observations.
        if block_end - block_start > 1 and not np.all(np.diff(timeline[block_start:block_end]) == DT_NS):
            raise ContractError("time-block bootstrap crossed a non-contiguous observed interval")
        selected = np.flatnonzero((source_segments == segment) & (source >= block_start) & (source < block_end))
        if len(selected):
            drawn.append(selected)
            remaining -= len(selected)
    return np.concatenate(drawn)[:len(source)]


def observed_minute_block_bootstrap_comparison(model_probability: np.ndarray, baseline_probability: np.ndarray,
                                               labels: np.ndarray, source_indices: np.ndarray,
                                               source_segment_ids: np.ndarray, timeline_times: np.ndarray,
                                               timeline_segment_ids: np.ndarray, block_minutes: int,
                                               samples: int, seed: int) -> dict[str, dict[str, float]]:
    """Exact-entry-count CIs whose dependence blocks are real minute durations."""
    model = np.asarray(model_probability, dtype=np.float64)
    baseline = np.asarray(baseline_probability, dtype=np.float64)
    outcome = np.asarray(labels, dtype=np.float64)
    if (len(model) < 2 or len(model) != len(baseline) or len(model) != len(outcome)
            or len(model) != len(source_indices) or len(model) != len(source_segment_ids)):
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
        indices = observed_minute_block_resample_indices(
            source_indices, source_segment_ids, timeline_times, timeline_segment_ids, block_minutes, rng)
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


def observed_minute_block_multiclass_bootstrap_comparison(model_probabilities: np.ndarray,
                                                          baseline_probabilities: np.ndarray,
                                                          labels: np.ndarray, source_indices: np.ndarray,
                                                          source_segment_ids: np.ndarray, timeline_times: np.ndarray,
                                                          timeline_segment_ids: np.ndarray, block_minutes: int,
                                                          samples: int, seed: int) -> dict[str, dict[str, float]]:
    """Observed-minute block CIs for the primary three-class target."""
    model = np.asarray(model_probabilities, dtype=np.float64)
    baseline = np.asarray(baseline_probabilities, dtype=np.float64)
    outcome = np.asarray(labels, dtype=np.int8)
    if model.ndim != 2 or model.shape[1] != 3 or model.shape != baseline.shape or len(model) < 2 or len(model) != len(outcome):
        return {name: {"point": float("nan"), "lower_95": float("nan"), "upper_95": float("nan")}
                for name in ("brier_improvement", "log_loss_improvement")}
    point = np.array([multiclass_brier_score(baseline, outcome) - multiclass_brier_score(model, outcome),
                      multiclass_log_loss(baseline, outcome) - multiclass_log_loss(model, outcome)])
    estimates = np.empty((samples, 2), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for draw in range(samples):
        indices = observed_minute_block_resample_indices(source_indices, source_segment_ids, timeline_times,
                                                          timeline_segment_ids, block_minutes, rng)
        estimates[draw] = (multiclass_brier_score(baseline[indices], outcome[indices]) - multiclass_brier_score(model[indices], outcome[indices]),
                           multiclass_log_loss(baseline[indices], outcome[indices]) - multiclass_log_loss(model[indices], outcome[indices]))
    return {name: {"point": float(point[column]), "lower_95": float(np.quantile(estimates[:, column], 0.05)),
                   "upper_95": float(np.quantile(estimates[:, column], 0.95))}
            for column, name in enumerate(("brier_improvement", "log_loss_improvement"))}


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
        # Each level is defined only in its own mean/loading/scale coordinate
        # system.  A blended level is emitted separately; no branch ever
        # receives innovations from the other regime definition.
        self.residual_levels = [np.zeros(N_PAIRS, dtype=np.float64) for _ in range(2)]
        self.previous_regime_probabilities = np.asarray([0.5, 0.5], dtype=np.float64)
        self.log_vol_mean = 0.0
        self.log_vol_variance = 1.0
        self.regime_high_probability = 0.5
        # The optimizer acts on unit-gross composition; turnover diagnostics
        # additionally track scaled candidates and accepted entries.  Never
        # mix those three states.
        self.previous_unit_weights = np.zeros(N_PAIRS, dtype=np.float64)
        self.previous_scaled_weights = np.zeros(N_PAIRS, dtype=np.float64)
        self.previous_accepted_weights = np.zeros(N_PAIRS, dtype=np.float64)

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
        prior_active_regime = int(self.regime_high_probability >= 0.5)
        volatility_z = self._update_regime_probability(transformed_return)
        posterior = (1.0 - self.regime_high_probability, self.regime_high_probability)
        self.steps += 1
        residuals: list[np.ndarray] = []
        factors_by_regime: list[np.ndarray] = []
        residual_scales: list[np.ndarray] = []
        residual_innovations: list[np.ndarray] = []
        prior_levels_by_regime = np.stack(self.residual_levels).copy()
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
            scale = np.sqrt(np.maximum(self.residual_variances[regime], 1.0e-16))
            innovation = residual / scale
            # Conditional-state mixture: each regime gets its full innovation
            # in its own coordinate system.  Posterior weights are applied once
            # only when the regime states are blended below.
            self.residual_levels[regime] = (parameters.level_ar * self.residual_levels[regime]
                                            + innovation)
            residual_scales.append(scale)
            residual_innovations.append(innovation)
        active_regime = int(self.regime_high_probability >= 0.5)
        parameters = self.parameters[active_regime]
        residual = residuals[active_regime]
        residual_scale = residual_scales[active_regime]
        residual_innovation = residual_innovations[active_regime]
        regime_probabilities = np.asarray(posterior, dtype=np.float64)
        levels_by_regime = np.stack(self.residual_levels)
        blended_residual_level = regime_probabilities @ levels_by_regime
        decay_levels = np.stack([
            self.parameters[regime].level_ar * prior_levels_by_regime[regime]
            for regime in range(2)
        ])
        residual_level_decay_effect = regime_probabilities @ (decay_levels - prior_levels_by_regime)
        residual_level_innovation_effect = regime_probabilities @ np.stack(residual_innovations)
        residual_level_posterior_reweighting_effect = (
            regime_probabilities - self.previous_regime_probabilities) @ prior_levels_by_regime
        prior_blended_residual_level = self.previous_regime_probabilities @ prior_levels_by_regime
        residual_level_change = blended_residual_level - prior_blended_residual_level
        self.previous_regime_probabilities = regime_probabilities.copy()
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
            "residual_level": blended_residual_level.copy(),
            "residual_levels_by_regime": levels_by_regime.copy(),
            "regime_probabilities": regime_probabilities.copy(),
            "residual_level_change": residual_level_change.copy(),
            "residual_level_decay_effect": residual_level_decay_effect.copy(),
            "residual_level_innovation_effect": residual_level_innovation_effect.copy(),
            "residual_level_posterior_reweighting_effect": residual_level_posterior_reweighting_effect.copy(),
            "inactive_regime_residual_level_variance": float(np.var(levels_by_regime[1 - active_regime])),
            "inactive_regime_innovation_magnitude": float(np.linalg.norm(residual_innovations[1 - active_regime])),
            "regime_state_disagreement_l2": float(np.linalg.norm(levels_by_regime[0] - levels_by_regime[1])),
            "regime_switch": bool(active_regime != prior_active_regime),
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
            "means_by_regime": np.stack(self.means).copy(),
            "covariances_by_regime": np.stack(self.covariances).copy(),
            "loadings": self.loadings_by_regime[active_regime].copy(),
            "loadings_by_regime": np.stack(self.loadings_by_regime).copy(),
            "residual_scales_by_regime": np.stack(residual_scales).copy(),
            "graph_pressure": self.graph_pressures[active_regime].copy(),
            "graph_cluster_size": self.graph_cluster_sizes[active_regime].copy(),
        }, edges


def selection_from_features(features: dict[str, Any]) -> dict[str, Any]:
    """Select a residual signal before, but do not score before, basket construction."""
    level = np.asarray(features["residual_level"], dtype=np.float64)
    pressure = np.asarray(features["graph_pressure"], dtype=np.float64)
    cluster_size = np.asarray(features["graph_cluster_size"], dtype=np.float64)
    score = np.abs(level) / (1.0 + 0.75 * pressure + 0.10 * np.maximum(cluster_size - 1.0, 0.0))
    selected = int(np.argmax(score))
    parameters: RegimeParameters = features["parameters"]
    selected_level = float(level[selected])
    selected_pressure = float(pressure[selected])
    selected_cluster = int(cluster_size[selected])
    residual_entry_eligible = bool(abs(selected_level) >= parameters.entry_abs_level)
    beta = np.asarray(features["beta"], dtype=np.float64)
    unstable = float(np.mean((beta <= 0.0) | (beta >= 0.995)))
    baseline_breakdown_logit = (-2.10 + 1.20 * float(features["regime_high_probability"])
                                + 0.30 * min(abs(selected_level), 8.0) + 1.00 * selected_pressure
                                + 0.15 * max(selected_cluster - 1, 0) + 1.50 * unstable)
    half_life = float(np.asarray(features["half_life"])[selected])
    speed = math.log1p(parameters.holding_horizon_steps / half_life) if math.isfinite(half_life) and half_life > 0 else 0.0
    signal_basis = np.zeros(N_PAIRS, dtype=np.float64)
    signal_basis[selected] = -float(np.sign(selected_level) or 1.0)
    scale = max(0.0, min(1.0, (abs(selected_level) - parameters.entry_abs_level) /
                         max(parameters.stop_multiple * parameters.entry_abs_level, 1.0e-12)))
    return {
        "selected": selected,
        "signal_basis": signal_basis,
        "residual_entry_eligible": residual_entry_eligible,
        "base_position_scale": scale,
        "baseline_breakdown_logit": baseline_breakdown_logit,
        "speed": speed,
        "selected_level": selected_level,
        "selected_graph_pressure": selected_pressure,
        "selected_cluster_size": selected_cluster,
    }


def basket_concentration(weights: np.ndarray) -> float:
    """L1-share Herfindahl concentration; one means a single-pair basket."""
    gross = float(np.sum(np.abs(weights)))
    if gross <= 1.0e-12:
        return 1.0
    shares = np.abs(weights) / gross
    return float(np.sum(shares * shares))


def prediction_from_constructed_basket(features: dict[str, Any], selection: dict[str, Any],
                                       unit_weights: np.ndarray, projection_distortion_l2: float,
                                       signal_preserved_fraction: float,
                                       config: ArenaConfig) -> dict[str, Any]:
    """Score and gate the actual constrained basket, not the pre-projection signal."""
    gross_exposure = float(np.sum(np.abs(unit_weights)))
    concentration = basket_concentration(unit_weights)
    geometric_signal_alignment = signal_preserved_fraction
    breakdown = sigmoid(float(selection["baseline_breakdown_logit"])
                        + 1.10 * projection_distortion_l2
                        + 0.75 * max(concentration - 0.10, 0.0)
                        + 1.25 * max(config.minimum_signal_preservation - signal_preserved_fraction, 0.0))
    probability = sigmoid(-0.40 + 0.28 * min(abs(float(selection["selected_level"])), 8.0)
                          + 0.25 * float(selection["speed"]) - 2.25 * breakdown
                          - 0.65 * float(features["regime_high_probability"])
                          + 1.10 * signal_preserved_fraction
                          - 0.65 * projection_distortion_l2
                          - 0.45 * max(concentration - 0.10, 0.0))
    active_probability = sigmoid(-0.55 + 0.22 * min(abs(float(selection["selected_level"])), 8.0)
                                 + 0.15 * float(selection["speed"]) - 0.80 * breakdown
                                 - 0.35 * projection_distortion_l2)
    entry_eligible = bool(
        selection["residual_entry_eligible"]
        and signal_preserved_fraction >= config.minimum_signal_preservation
        and concentration <= config.maximum_basket_concentration
        and gross_exposure > 1.0e-12
        and float(selection["base_position_scale"]) > 1.0e-8
    )
    return {
        **selection,
        "probability": probability,
        "active_probability": active_probability,
        "breakdown": breakdown,
        "entry_eligible": entry_eligible,
        "position_scale": (float(selection["base_position_scale"]) * (1.0 - breakdown)
                           if entry_eligible else 0.0),
        "basket_concentration": concentration,
        "geometric_signal_alignment": geometric_signal_alignment,
        "basket_gross_exposure_l1_unit": gross_exposure,
    }


def evaluate_frozen_residual_target(times: np.ndarray, log_prices: np.ndarray, transform: np.ndarray,
                                    entry: FrozenResidualTarget) -> dict[str, Any] | None:
    """Observe frozen residual and basket paths without changing regime definitions."""
    target_index = entry.source_index + entry.horizon_steps
    if target_index >= len(times):
        return None
    for index in range(entry.source_index + 1, target_index + 1):
        if not contiguous_60s(int(times[index - 1]), int(times[index]), DT_NS):
            return None
    levels_by_regime = entry.entry_levels_by_regime.copy()
    levels: list[float] = []
    residual_innovations: list[float] = []
    selected_residual_returns: list[float] = []
    basket_returns: list[float] = []
    for index in range(entry.source_index + 1, target_index + 1):
        raw_return = log_prices[index] - log_prices[index - 1]
        basket_returns.append(float(entry.raw_basket_weights @ raw_return))
        transformed = transform @ raw_return
        residual_values = np.empty(2, dtype=np.float64)
        innovations = np.empty(2, dtype=np.float64)
        for regime in range(2):
            centered = transformed - entry.means_by_regime[regime]
            residual = centered - entry.loadings_by_regime[regime] @ (
                entry.loadings_by_regime[regime].T @ centered)
            residual_values[regime] = residual[entry.selected]
            innovations[regime] = residual[entry.selected] / max(
                entry.residual_scales_by_regime[regime, entry.selected], 1.0e-12)
            levels_by_regime[regime] = (entry.level_ars[regime] * levels_by_regime[regime]
                                        + innovations[regime])
        levels.append(float((entry.regime_probabilities @ levels_by_regime)[entry.selected]))
        residual_innovations.append(float(entry.regime_probabilities @ innovations))
        selected_residual_returns.append(float(entry.regime_probabilities @ residual_values))
    path = np.asarray(levels, dtype=np.float64)
    entry_sign = float(np.sign(entry.entry_level))
    gross_convergence = -entry_sign * (float(path[-1]) - entry.entry_level)
    path_gross = -entry_sign * (path - entry.entry_level)
    time_to_zero = next((offset + 1 for offset, value in enumerate(path) if value * entry_sign <= 0.0), None)
    removed = 100.0 * (1.0 - abs(float(path[-1])) / max(abs(entry.entry_level), 1.0e-12))
    breakdown = bool(np.max(np.abs(path)) > entry.stop_multiple * abs(entry.entry_level))
    basket_path = np.cumsum(np.asarray(basket_returns, dtype=np.float64))
    basket_gross = float(basket_path[-1])
    basket_standardized_return = basket_gross / max(
        entry.basket_one_step_volatility * math.sqrt(entry.horizon_steps), 1.0e-12)
    if basket_standardized_return > entry.basket_neutral_zone_z:
        basket_outcome_class = 1
        basket_binary_label: float = 1.0
    elif basket_standardized_return < -entry.basket_neutral_zone_z:
        basket_outcome_class = -1
        basket_binary_label = 0.0
    else:
        basket_outcome_class = 0
        basket_binary_label = float("nan")
    residual_label = int(gross_convergence > 0.0 and abs(float(path[-1])) < abs(entry.entry_level))
    if len(basket_returns) < 2 or np.std(basket_returns) <= 1.0e-16 or np.std(selected_residual_returns) <= 1.0e-16:
        tracking_correlation = float("nan")
    else:
        tracking_correlation = float(np.corrcoef(basket_returns, selected_residual_returns)[0, 1])
    return {
        "target_time": pd.Timestamp(int(times[target_index]), unit="ns", tz="UTC"),
        # The residual label remains explanatory.  The primary predictive
        # label is the fixed basket's directional gross log return below.
        "convergence_label": residual_label,
        "gross_convergence": gross_convergence,
        "mae_level": float(max(0.0, -np.min(path_gross))),
        "time_to_zero_steps": float(time_to_zero) if time_to_zero is not None else float("nan"),
        "percentage_displacement_removed": removed,
        "breakdown_label": int(breakdown),
        "target_path_volatility": float(np.std(residual_innovations, ddof=0)),
        "frozen_target_level": float(path[-1]),
        "basket_standardized_return": basket_standardized_return,
        "basket_neutral_zone_z": entry.basket_neutral_zone_z,
        "basket_outcome_class": basket_outcome_class,
        "basket_binary_label": basket_binary_label,
        "basket_directional_label": basket_outcome_class,
        "basket_tracking_correlation": tracking_correlation,
        "basket_max_adverse_excursion": float(max(0.0, -np.min(basket_path))),
        "basket_residual_convergence_disagreement": (
            int(int(basket_binary_label) != residual_label) if math.isfinite(basket_binary_label) else float("nan")),
        "basket_cumulative_gross_log_return": basket_gross,
        "frozen_basket_cumulative_log_return": basket_gross,
        "frozen_basket_path_volatility": float(np.std(basket_returns, ddof=0)),
    }


def _level_bin(values: pd.Series) -> pd.Series:
    return pd.Series(np.digitize(np.abs(values.to_numpy(dtype=np.float64)), [0.75, 1.50, 2.50, 4.00]),
                     index=values.index, dtype=np.int8)


def frozen_conditional_climatology(train: pd.DataFrame, target: pd.DataFrame, label_column: str,
                                   minimum_cell_count: int = 20) -> tuple[np.ndarray, list[str]]:
    """Fit only on train labels; predict a conditional primary-label climatology."""
    if train.empty or target.empty:
        return np.full(len(target), float("nan")), []
    source = train.copy()
    source["_level_bin"] = _level_bin(source["entry_residual_level"])
    destination = target.copy()
    destination["_level_bin"] = _level_bin(destination["entry_residual_level"])

    def table(columns: list[str]) -> dict[tuple[Any, ...], tuple[int, int]]:
        grouped = source.groupby(columns, dropna=False)[label_column].agg(["sum", "count"])
        return {key if isinstance(key, tuple) else (key,): (int(row["sum"]), int(row["count"]))
                for key, row in grouped.iterrows()}

    detailed_columns = ["_level_bin", "selected_component_index", "session_bucket", "active_regime"]
    component_columns = ["_level_bin", "selected_component_index", "active_regime"]
    regime_columns = ["_level_bin", "active_regime"]
    detailed, component, regime = (table(detailed_columns), table(component_columns), table(regime_columns))
    global_probability = float(source[label_column].mean())
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
    if config.basket_mode not in (CYCLE_NEUTRAL_MODE, RELATIVE_VALUE_MODE):
        raise ContractError("basket_mode must be cycle_neutral or relative_value")
    if not 0.0 < config.minimum_signal_preservation <= 1.0:
        raise ContractError("minimum_signal_preservation must be in (0, 1]")
    if not 0.10 <= config.maximum_basket_concentration <= 1.0:
        raise ContractError("maximum_basket_concentration must be in [0.10, 1]")
    if not 0.0 < config.relative_value_max_weight <= 1.0:
        raise ContractError("relative_value_max_weight must be in (0, 1]")
    if not 0.0 < config.basket_neutral_zone_z < 5.0:
        raise ContractError("basket_neutral_zone_z must be in (0, 5)")
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
    optimizer_rejections = 0
    segment_id = 0
    gap_resets = 0
    max_contiguous_state_steps = 0
    timeline_segment_ids = np.zeros(len(times), dtype=np.int64)
    for index in range(1, len(times)):
        if not contiguous_60s(int(times[index - 1]), int(times[index]), DT_NS):
            state.reset()
            segment_id += 1
            gap_resets += 1
            timeline_segment_ids[index] = segment_id
            continue
        timeline_segment_ids[index] = segment_id
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
        selection = selection_from_features(features)
        parameters: RegimeParameters = features["parameters"]
        raw_covariance = np.zeros((N_PAIRS, N_PAIRS), dtype=np.float64)
        for probability, covariance_basis in zip(features["regime_probabilities"], features["covariances_by_regime"], strict=True):
            raw_covariance += float(probability) * (inverse_transform @ covariance_basis @ inverse_transform.T)
        optimization = factor_neutral_weights(
            selection["signal_basis"], transform, inverse_transform, features["loadings"], incidence,
            parameters.neutral_factor_count, config.basket_mode,
            config.relative_value_currency_exposure_budget, raw_covariance, state.previous_unit_weights,
            config.relative_value_max_weight,
        )
        if not optimization.converged:
            # A pathological minute is ineligible; it must not abort a full
            # historical replay or contaminate any of the weight states.
            optimizer_rejections += 1
            continue
        unit_weights = optimization.weights
        projection_distortion_l2, signal_preserved_fraction = basket_projection_diagnostics(
            selection["signal_basis"], transform, unit_weights)
        decision = prediction_from_constructed_basket(
            features, selection, unit_weights, projection_distortion_l2, signal_preserved_fraction, config)
        weights = unit_weights * float(decision["position_scale"])
        candidate_turnover = float(np.sum(np.abs(weights - state.previous_scaled_weights)))
        accepted_turnover = float(np.sum(np.abs(weights - state.previous_accepted_weights)))
        state.previous_unit_weights = unit_weights.copy()
        state.previous_scaled_weights = weights.copy()
        raw_factor_directions = inverse_transform @ np.asarray(features["loadings"])
        factor_exposure = raw_factor_directions[:, :parameters.neutral_factor_count].T @ weights
        currency_exposure = incidence.T @ weights
        selected = int(decision["selected"])
        entry_eligible = bool(decision["entry_eligible"])
        if entry_eligible:
            state.previous_accepted_weights = weights.copy()
        basket_one_step_volatility = float(np.sqrt(max(weights @ raw_covariance @ weights, 1.0e-16)))
        record: dict[str, Any] = {
            "source_index": index,
            "timestamp": pd.Timestamp(int(times[index]), unit="ns", tz="UTC"),
            "segment_id": segment_id,
            "selected_component_index": selected,
            "selected_component": component_names[selected],
            "active_regime": features["active_regime"],
            "basket_mode": config.basket_mode,
            "relative_value_currency_exposure_budget": config.relative_value_currency_exposure_budget,
            "high_volatility_regime_probability": float(features["regime_high_probability"]),
            "session_bucket": int(pd.Timestamp(int(times[index]), unit="ns", tz="UTC").hour // 6),
            "p_basket_directional": float(decision["probability"]),
            "p_basket_active": float(decision["active_probability"]),
            "p_basket_neutral": float(1.0 - decision["active_probability"]),
            "p_basket_positive": float(decision["active_probability"] * decision["probability"]),
            "p_basket_negative": float(decision["active_probability"] * (1.0 - decision["probability"])),
            "breakdown_probability": float(decision["breakdown"]),
            "residual_entry_eligible": bool(selection["residual_entry_eligible"]),
            "entry_eligible": entry_eligible,
            "entry_residual_level": float(decision["selected_level"]),
            "entry_abs_level_threshold": parameters.entry_abs_level,
            "holding_horizon_steps": parameters.holding_horizon_steps,
            "stop_multiple": parameters.stop_multiple,
            "neutral_factor_count": parameters.neutral_factor_count,
            "diagnostic_position_scale": float(decision["position_scale"] if entry_eligible else 0.0),
            "diagnostic_gross_exposure_l1": float(np.sum(np.abs(weights))),
            "optimizer_converged": bool(optimization.converged),
            "optimizer_iterations": int(optimization.iterations),
            "optimizer_objective_value": float(optimization.objective_value),
            "optimizer_max_factor_violation": float(optimization.max_factor_violation),
            "optimizer_max_currency_violation": float(optimization.max_currency_violation),
            "optimizer_max_weight_violation": float(optimization.max_weight_violation),
            "optimizer_gross_exposure_violation": float(optimization.gross_exposure_violation),
            "projection_distortion_l2": projection_distortion_l2,
            "signal_preserved_fraction": signal_preserved_fraction,
            "geometric_signal_alignment": float(decision["geometric_signal_alignment"]),
            "basket_concentration": float(decision["basket_concentration"]),
            "basket_one_step_volatility": basket_one_step_volatility,
            "basket_neutral_zone_z": config.basket_neutral_zone_z,
            "turnover_l1_diagnostic": candidate_turnover,
            "accepted_entry_turnover_l1_diagnostic": accepted_turnover if entry_eligible else np.nan,
            "selected_graph_pressure": float(decision["selected_graph_pressure"]),
            "selected_graph_cluster_size": int(decision["selected_cluster_size"]),
            "factor_volatility": float(features["factor_volatility"]),
            "residual_volatility": float(features["residual_volatility"]),
            "volatility_z": float(features["volatility_z"]),
            "selected_level_change": float(features["residual_level_change"][selected]),
            "selected_level_decay_effect": float(features["residual_level_decay_effect"][selected]),
            "selected_level_innovation_effect": float(features["residual_level_innovation_effect"][selected]),
            "selected_level_posterior_reweighting_effect": float(features["residual_level_posterior_reweighting_effect"][selected]),
            "inactive_regime_residual_level_variance": float(features["inactive_regime_residual_level_variance"]),
            "inactive_regime_innovation_magnitude": float(features["inactive_regime_innovation_magnitude"]),
            "regime_state_disagreement_l2": float(features["regime_state_disagreement_l2"]),
            "regime_switch": bool(features["regime_switch"]),
            "target_time": pd.NaT,
            "convergence_label": np.nan,
            "basket_standardized_return": np.nan,
            "basket_outcome_class": np.nan,
            "basket_binary_label": np.nan,
            "basket_directional_label": np.nan,
            "basket_tracking_correlation": np.nan,
            "basket_max_adverse_excursion": np.nan,
            "basket_residual_convergence_disagreement": np.nan,
            "basket_cumulative_gross_log_return": np.nan,
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
            entry_levels_by_regime=np.asarray(features["residual_levels_by_regime"]).copy(),
            regime_probabilities=np.asarray(features["regime_probabilities"]).copy(),
            means_by_regime=np.asarray(features["means_by_regime"]).copy(),
            loadings_by_regime=np.asarray(features["loadings_by_regime"]).copy(),
            residual_scales_by_regime=np.asarray(features["residual_scales_by_regime"]).copy(),
            level_ars=np.asarray([config.low_regime.level_ar, config.high_regime.level_ar], dtype=np.float64),
            horizon_steps=parameters.holding_horizon_steps,
            stop_multiple=parameters.stop_multiple,
            raw_basket_weights=weights.copy(),
            basket_one_step_volatility=basket_one_step_volatility,
            basket_neutral_zone_z=config.basket_neutral_zone_z,
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
    emissions = annotate_entry_diagnostics(emissions)
    labelled = emissions[emissions["basket_outcome_class"].notna()].copy()
    labelled["basket_class"] = labelled["basket_outcome_class"].map({0: 0, 1: 1, -1: 2}).astype(np.int8)
    frozen_training = (labelled["source_index"] + labelled["holding_horizon_steps"] < test_start_index)
    evaluation_metadata, purge_summary = build_evaluation_metadata(
        labelled["source_index"].to_numpy(dtype=np.int64), config.max_horizon_steps,
        labelled["segment_id"].to_numpy(dtype=np.int64), frozen_training.to_numpy(dtype=bool),
        embargo_steps=config.max_horizon_steps,
    )
    for column in evaluation_metadata:
        emissions[column] = pd.NA
        emissions.loc[labelled.index, column] = evaluation_metadata[column].to_numpy()
    train = labelled[labelled["source_index"] + labelled["holding_horizon_steps"] < test_start_index].copy()
    oos = labelled[labelled["source_index"] >= test_start_index].copy()
    if len(train) < 100 or len(oos) < 100:
        raise ContractError("train/OOS target partitions require at least 100 valid frozen entries each")
    class_prior = train["basket_class"].value_counts(normalize=True).reindex([0, 1, 2], fill_value=0.0).to_numpy(dtype=np.float64)
    model_probabilities = oos[["p_basket_neutral", "p_basket_positive", "p_basket_negative"]].to_numpy(dtype=np.float64)
    oos_class = oos["basket_class"].to_numpy(dtype=np.int8)
    prior_probabilities = np.tile(class_prior, (len(oos), 1))
    purged_train = labelled.loc[frozen_training & ~evaluation_metadata["embargoed"].to_numpy(dtype=bool)].copy()
    if purged_train.empty:
        raise ContractError("purged/embargoed sensitivity left no training entries")
    purged_class_prior = (purged_train["basket_class"].value_counts(normalize=True)
                          .reindex([0, 1, 2], fill_value=0.0).to_numpy(dtype=np.float64))
    purged_prior_probabilities = np.tile(purged_class_prior, (len(oos), 1))
    primary_bootstrap_sensitivity = {
        f"{block_minutes}_observed_minutes": observed_minute_block_multiclass_bootstrap_comparison(
            model_probabilities, prior_probabilities, oos_class, oos["source_index"].to_numpy(dtype=np.int64),
            oos["segment_id"].to_numpy(dtype=np.int64), times, timeline_segment_ids, block_minutes,
            config.bootstrap_samples, config.bootstrap_seed + 10_000 + block_minutes)
        for block_minutes in config.bootstrap_block_sensitivity_minutes
    }
    directional_train = train[train["basket_binary_label"].notna()].copy()
    directional_oos = oos[oos["basket_binary_label"].notna()].copy()
    if len(directional_train) < 100 or len(directional_oos) < 100:
        raise ContractError("directional secondary partitions require at least 100 non-neutral frozen entries each")
    directional_train["basket_binary_label"] = directional_train["basket_binary_label"].astype(np.int8)
    directional_oos["basket_binary_label"] = directional_oos["basket_binary_label"].astype(np.int8)
    frozen_train_prior = float(directional_train["basket_binary_label"].mean())
    conditional_probability, conditional_tiers = frozen_conditional_climatology(
        directional_train, directional_oos, "basket_binary_label")
    oos_probability = directional_oos["p_basket_directional"].to_numpy(dtype=np.float64)
    oos_label = directional_oos["basket_binary_label"].to_numpy(dtype=np.float64)
    prior_probability = np.full(len(directional_oos), frozen_train_prior, dtype=np.float64)
    emissions["p_conditional_climatology"] = np.nan
    emissions["conditional_climatology_tier"] = pd.NA
    emissions.loc[directional_oos.index, "p_conditional_climatology"] = conditional_probability
    emissions.loc[directional_oos.index, "conditional_climatology_tier"] = conditional_tiers
    bootstrap_sensitivity = {
        f"{block_minutes}_observed_minutes": observed_minute_block_bootstrap_comparison(
            oos_probability, conditional_probability, oos_label,
            directional_oos["source_index"].to_numpy(dtype=np.int64), directional_oos["segment_id"].to_numpy(dtype=np.int64),
            times, timeline_segment_ids, block_minutes, config.bootstrap_samples,
            config.bootstrap_seed + block_minutes,
        )
        for block_minutes in config.bootstrap_block_sensitivity_minutes
    }
    primary_bootstrap = primary_bootstrap_sensitivity[
        f"{config.bootstrap_block_sensitivity_minutes[-1]}_observed_minutes"]
    shuffled_labels = oos_label[np.random.default_rng(config.bootstrap_seed).permutation(len(oos_label))]
    primary_prediction_pass = bool(
        primary_bootstrap["brier_improvement"]["lower_95"] > 0.0
        and primary_bootstrap["log_loss_improvement"]["lower_95"] > 0.0
    )
    directional_prediction_pass = bool(
        brier_score(oos_probability, oos_label) < brier_score(conditional_probability, oos_label)
        and bootstrap_sensitivity[f"{config.bootstrap_block_sensitivity_minutes[-1]}_observed_minutes"]["brier_improvement"]["lower_95"] > 0.0
    )
    tier_counts = pd.Series(conditional_tiers, dtype="string").value_counts().to_dict()
    summary: dict[str, Any] = {
        "version": VERSION,
        "interpretation": "causal classical residual-level research only; no execution, PnL, capacity, or market-impact claim",
        "pair_scope": list(PAIRS),
        "target_contract": {
            "primary_target": "three-class frozen basket standardized gross return: neutral, positive beyond the frozen neutral zone, or negative beyond it; neutral outcomes remain in every primary score",
            "basket_standardized_return": "sum(w_entry * raw_log_return) / (frozen_one_step_basket_volatility * sqrt(horizon))",
            "basket_neutral_zone_z": config.basket_neutral_zone_z,
            "residual_explanatory_target": "frozen selected residual level converges after its frozen entry-specific holding horizon",
            "gross_convergence": "-sign(S_t) * (S_{t+h,frozen} - S_t)",
            "residual_level": "S_t = p_low S_t_low + p_high S_t_high; each branch updates only in its own frozen coordinate system",
            "frozen_at_entry": ["both_regime_factor_means", "both_regime_factor_loadings", "both_regime_residual_scales", "regime_probabilities", "selected_component", "basket_weights", "both_level_ARs", "horizon", "stop"],
            "target_available_only_after": "all entry-to-horizon arrivals are observed contiguous one-minute bars",
            "outcome_diagnostics": ["residual_MAE", "time_to_zero", "percentage_displacement_removed", "residual_breakdown", "residual_path_volatility", "basket_tracking_correlation", "projection_distortion", "signal_preserved_fraction", "basket_disagreement", "basket_MAE", "basket_gross_log_return", "turnover"],
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
            "basket_construction": ("cycle_neutral: D.T @ w=0 closed-loop research" if config.basket_mode == CYCLE_NEUTRAL_MODE
                                    else "relative_value: selected factor neutrality plus bounded max(abs(D.T @ w)) in pair-coefficient units"),
            "post_construction_gate": "probability and entry use signal preservation, projection distortion, geometric signal alignment, concentration, and nonzero gross exposure; gross exposure is an eligibility check only",
            "currency_units_caveat": "incidence neutrality/budgets are pair coefficient units only; no dollar-risk sizing without contract notionals and conversion prices",
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
            "model_three_class": {"brier_score": multiclass_brier_score(model_probabilities, oos_class), "log_loss": multiclass_log_loss(model_probabilities, oos_class), "class_order": ["neutral", "positive", "negative"]},
            "basket_standardized_return": {"mean": float(labelled["basket_standardized_return"].mean()), "neutral_outcomes_included": int((labelled["basket_outcome_class"] == 0).sum())},
            "three_class_train_prior": {"probabilities": class_prior.tolist(), "brier_score": multiclass_brier_score(prior_probabilities, oos_class), "log_loss": multiclass_log_loss(prior_probabilities, oos_class)},
            "primary_three_class_observed_minute_block_bootstrap": {"samples": config.bootstrap_samples, "exact_replicate_entry_count": int(len(oos)), "block_unit": "actual contiguous observed one-minute bars", "sensitivity": primary_bootstrap_sensitivity},
            "primary_three_class_passes_prediction_gate": primary_prediction_pass,
            "frozen_train_prior": {"probability": frozen_train_prior, "brier_score": brier_score(prior_probability, oos_label), "log_loss": log_loss(prior_probability, oos_label), "calibration_error": calibration_error(prior_probability, oos_label)},
            "frozen_conditional_climatology": {"brier_score": brier_score(conditional_probability, oos_label), "log_loss": log_loss(conditional_probability, oos_label), "calibration_error": calibration_error(conditional_probability, oos_label), "tier_counts": {str(key): int(value) for key, value in tier_counts.items()}},
            "time_shuffled_residual_placebo": {"model_brier_score": brier_score(oos_probability, shuffled_labels), "conditional_climatology_brier_score": brier_score(conditional_probability, shuffled_labels), "seed": config.bootstrap_seed},
            "observed_minute_block_bootstrap": {"samples": config.bootstrap_samples, "exact_replicate_entry_count": int(len(oos)), "block_unit": "actual contiguous observed one-minute bars", "entries_selected_from_time_blocks": True, "sensitivity": bootstrap_sensitivity},
            "secondary_directional_passes_conditional_climatology_gate": directional_prediction_pass,
            "purged_embargoed_sensitivity": {
                "horizon_steps": config.max_horizon_steps,
                "train_rows_total": purge_summary.train_rows_total,
                "purged_rows": purge_summary.purged_rows,
                "embargoed_extra_purged_rows": purge_summary.embargoed_extra_purged_rows,
                "remaining_train_rows": purge_summary.remaining_train_rows,
                "three_class_train_prior": {"probabilities": purged_class_prior.tolist(),
                                               "brier_score": multiclass_brier_score(purged_prior_probabilities, oos_class),
                                               "log_loss": multiclass_log_loss(purged_prior_probabilities, oos_class)},
            },
        },
        "entry_policy": summarize_entry_policy(emissions).__dict__,
        "optimizer": {"infeasible_entry_rejections": optimizer_rejections,
                      "infeasible_entries_are_ineligible": True},
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
    manifest_path = out_dir / f"{prefix}_manifest.json"
    result.emissions.to_parquet(emissions_path, index=False, compression="zstd")
    result.graph.to_parquet(graph_path, index=False, compression="zstd")
    daily = (result.emissions.assign(day=result.emissions["timestamp"].dt.floor("D"))
             .groupby("day", as_index=False)
             .agg(emissions=("source_index", "size"),
                  mean_p_basket_directional=("p_basket_directional", "mean"),
                  diagnostic_turnover_l1=("turnover_l1_diagnostic", "sum")))
    daily.to_parquet(daily_path, index=False, compression="zstd")
    result.summary["outputs"] = {"minute": str(emissions_path.relative_to(ROOT)).replace("\\", "/"), "graph": str(graph_path.relative_to(ROOT)).replace("\\", "/"), "daily": str(daily_path.relative_to(ROOT)).replace("\\", "/")}
    result.summary["source_hashes"] = {"models/statistical/stat_arb.py": sha256_file(Path(__file__).resolve()), **{f"data/canonical/{pair}.parquet": sha256_file(CANONICAL_DIR / f"{pair}.parquet") for pair in PAIRS}}
    summary_schema = json.loads((ROOT / "engine" / "config" / "schemas" / "stat-arb-summary.schema.json").read_text(encoding="utf-8"))
    schema_errors = validate_instance(summary_schema, result.summary)
    if schema_errors:
        raise ContractError(f"refusing to write schema-invalid stat-arb summary: {schema_errors}")
    summary_path.write_text(json.dumps(result.summary, indent=2) + "\n", encoding="utf-8")
    manifest = build_run_manifest(
        root=ROOT, frozen_contract_version=VERSION, required_tests_passed=False,
        holdout_status="burned_acknowledged", source_files=[Path(__file__)],
        input_artifacts=[CANONICAL_DIR / f"{pair}.parquet" for pair in PAIRS],
        output_artifacts=[emissions_path, graph_path, daily_path, summary_path],
        configuration={"label": label, "version": VERSION},
    )
    write_manifest(manifest, manifest_path)
    return {"minute": str(emissions_path), "graph": str(graph_path), "daily": str(daily_path),
            "summary": str(summary_path), "manifest": str(manifest_path)}


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
        bootstrap_block_sensitivity_minutes=(8, 16, 32),
        basket_mode=RELATIVE_VALUE_MODE,
        relative_value_currency_exposure_budget=0.35,
        basket_neutral_zone_z=0.05,
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
    prefix_equal = bool(np.array_equal(comparable.loc[common, "p_basket_directional"], altered_comparable.loc[common, "p_basket_directional"]))
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
    parser.add_argument("--basket-mode", choices=(CYCLE_NEUTRAL_MODE, RELATIVE_VALUE_MODE),
                        default=CYCLE_NEUTRAL_MODE,
                        help="diagnostic basket contract; cycle_neutral is the frozen default")
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
        result = run_arrays(times, prices, test_start, ArenaConfig(basket_mode=args.basket_mode))
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

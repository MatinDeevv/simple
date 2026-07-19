"""Independent metric and concentration recomputation from evaluation rows."""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any

from engine.experiments.errors import EvidenceValidationError


def _losses(rows: list[dict[str, Any]], field: str, class_order: list[str]) -> tuple[float, float]:
    brier = 0.0
    log_loss = 0.0
    for row in rows:
        target = class_order.index(row["class_label"])
        probs = [float(value) for value in row[field]]
        brier += sum((prob - (1.0 if index == target else 0.0)) ** 2
                     for index, prob in enumerate(probs)) / len(class_order)
        log_loss -= math.log(max(probs[target], 1e-15))
    return brier / len(rows), log_loss / len(rows)


def recompute(rows: list[dict[str, Any]], *, class_order: list[str]) -> dict[str, Any]:
    model_brier, model_log = _losses(rows, "model_probabilities", class_order)
    comparator_brier, comparator_log = _losses(rows, "comparator_probabilities", class_order)
    counts = Counter(row["class_label"] for row in rows)

    def concentration(field: str) -> float:
        grouped = Counter(row[field] for row in rows)
        return max(grouped.values()) / len(rows)

    return {"accepted_rows": len(rows), "class_counts": dict(sorted(counts.items())),
            "model_brier": model_brier, "comparator_brier": comparator_brier,
            "brier_improvement": comparator_brier - model_brier,
            "model_log_loss": model_log, "comparator_log_loss": comparator_log,
            "log_loss_improvement": comparator_log - model_log,
            "largest_signal_episode_fraction": concentration("signal_episode_id"),
            "largest_target_cluster_fraction": concentration("target_cluster_id"),
            "largest_causal_segment_fraction": concentration("segment_id"),
            "largest_day_fraction": (max(Counter(
                row["timestamp_utc"][:10] for row in rows).values()) / len(rows)),
            "largest_component_fraction": concentration("component"),
            "largest_regime_fraction": concentration("regime"),
            "largest_fallback_tier_fraction": concentration("fallback_tier")}


def assert_summary_matches(recomputed: dict[str, Any], submitted: dict[str, Any],
                           *, tolerance: float = 1e-12) -> None:
    for key, expected in recomputed.items():
        if key not in submitted:
            raise EvidenceValidationError(f"missing independently recomputable metric {key}")
        observed = submitted[key]
        if isinstance(expected, float):
            if not isinstance(observed, (int, float)) or abs(float(observed) - expected) > tolerance:
                raise EvidenceValidationError(f"submitted {key} differs from recomputation")
        elif observed != expected:
            raise EvidenceValidationError(f"submitted {key} differs from recomputation")


def deterministic_block_bootstrap(rows: list[dict[str, Any]], *, class_order: list[str],
                                  block_length: int, replicate_count: int, seed: int,
                                  confidence_level: float = 0.95,
                                  two_sided_confidence_level: float | None = None
                                  ) -> dict[str, float]:
    """Independent segment-preserving moving-block bootstrap.

    Blocks never cross segment boundaries and no replicate is silently
    discarded. Insufficient segment support fails before sampling.
    """
    two_sided_confidence_level = two_sided_confidence_level or confidence_level
    if (block_length < 1 or replicate_count < 1 or not 0 < confidence_level < 1
            or not 0 < two_sided_confidence_level < 1):
        raise EvidenceValidationError("invalid bootstrap configuration")
    segments: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        segments.setdefault(str(row["segment_id"]), []).append(row)
    candidates = [(segment, start) for segment, items in sorted(segments.items())
                  for start in range(0, len(items) - block_length + 1)]
    if not candidates:
        raise EvidenceValidationError("insufficient segment-preserving bootstrap support")
    rng = random.Random(seed)
    brier_values: list[float] = []
    log_values: list[float] = []
    target = len(rows)
    for _ in range(replicate_count):
        sampled: list[dict[str, Any]] = []
        while len(sampled) < target:
            segment, start = candidates[rng.randrange(len(candidates))]
            sampled.extend(segments[segment][start:start + block_length])
        result = recompute(sampled[:target], class_order=class_order)
        brier_values.append(result["brier_improvement"])
        log_values.append(result["log_loss_improvement"])
    def quantile(values: list[float], q: float) -> float:
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
        return ordered[index]
    alpha_one = 1 - confidence_level
    alpha_two = 1 - two_sided_confidence_level
    return {"brier_lower_one_sided": quantile(brier_values, alpha_one),
            "brier_lower_two_sided": quantile(brier_values, alpha_two / 2),
            "brier_upper_two_sided": quantile(brier_values, 1 - alpha_two / 2),
            "log_loss_lower_one_sided": quantile(log_values, alpha_one),
            "log_loss_lower_two_sided": quantile(log_values, alpha_two / 2),
            "log_loss_upper_two_sided": quantile(log_values, 1 - alpha_two / 2)}

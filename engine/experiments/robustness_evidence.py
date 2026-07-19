"""Independent row-level recomputation for every preregistered robustness cell."""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from engine.experiments.canonical import load_strict_json_text
from engine.experiments.errors import EvidenceValidationError
from engine.experiments.robustness import planned_cell_registry

VERSION = "edge-tribunal-robustness-rows-v1"


def recompute_robustness_rows(path: Path, *, contract: dict[str, Any],
                              class_order: list[str]) -> list[dict[str, Any]]:
    planned = planned_cell_registry(contract)
    headers: dict[str, dict[str, Any]] = {}
    accum: dict[str, dict[str, Any]] = {}
    row_ids: set[tuple[str, str]] = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        item = load_strict_json_text(line)
        if not isinstance(item, dict) or item.get("version") != VERSION:
            raise EvidenceValidationError(f"robustness line {number} has invalid version")
        cell_id = item.get("cell_id")
        if cell_id not in planned:
            raise EvidenceValidationError(f"robustness line {number} has unregistered cell")
        if item.get("kind") == "cell":
            if cell_id in headers or item.get("dimensions") != planned[cell_id]:
                raise EvidenceValidationError(f"robustness line {number} has duplicate/mismatched header")
            status = item.get("status")
            if status not in ("scored", "insufficient_train_population",
                              "insufficient_oos_population", "insufficient_class_support",
                              "insufficient_target_clusters", "insufficient_bootstrap_support",
                              "missing"):
                raise EvidenceValidationError(f"robustness line {number} has invalid status")
            headers[cell_id] = item
            accum[cell_id] = {"count": 0, "clusters": Counter(), "model_brier": 0.0,
                              "comparator_brier": 0.0, "model_log": 0.0,
                              "comparator_log": 0.0}
            continue
        if item.get("kind") != "row" or cell_id not in headers:
            raise EvidenceValidationError(f"robustness line {number} row precedes its header")
        row_id = item.get("row_id")
        identity = (cell_id, row_id)
        if not isinstance(row_id, str) or not row_id or identity in row_ids:
            raise EvidenceValidationError(f"robustness line {number} has duplicate/invalid row_id")
        row_ids.add(identity)
        label = item.get("class_label")
        if label not in class_order:
            raise EvidenceValidationError(f"robustness line {number} has unknown class")
        target = class_order.index(label)
        values: list[list[float]] = []
        for key in ("model_probabilities", "comparator_probabilities"):
            probs = item.get(key)
            if (not isinstance(probs, list) or len(probs) != len(class_order)
                    or any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(float(value)) or not 0 <= float(value) <= 1
                           for value in probs)
                    or abs(sum(map(float, probs)) - 1.0) > 1e-12):
                raise EvidenceValidationError(f"robustness line {number} has invalid probabilities")
            values.append([float(value) for value in probs])
        state = accum[cell_id]
        state["count"] += 1
        state["clusters"][str(item.get("target_cluster_id"))] += 1
        for prefix, probs in zip(("model", "comparator"), values):
            state[f"{prefix}_brier"] += sum(
                (prob - (1.0 if index == target else 0.0)) ** 2
                for index, prob in enumerate(probs)) / len(class_order)
            state[f"{prefix}_log"] -= math.log(max(probs[target], 1e-15))
    if set(headers) != set(planned):
        raise EvidenceValidationError("robustness artifact does not cover every planned cell")
    results: list[dict[str, Any]] = []
    for cell_id in sorted(planned):
        header, state = headers[cell_id], accum[cell_id]
        status = header["status"]
        count = state["count"]
        if status == "scored" and count < contract["minimum_samples_per_cell"]:
            raise EvidenceValidationError(f"scored robustness cell {cell_id} lacks samples")
        if status != "scored" and count:
            raise EvidenceValidationError(f"insufficient/missing cell {cell_id} contains rows")
        result = {"cell_id": cell_id, "dimensions": planned[cell_id], "status": status,
                  "sample_count": count, "target_clusters": len(state["clusters"])}
        if status == "scored":
            result["brier_improvement"] = ((state["comparator_brier"]
                                             - state["model_brier"]) / count)
            result["log_loss_improvement"] = ((state["comparator_log"]
                                                - state["model_log"]) / count)
        results.append(result)
    return results

"""Strict, inspectable JSONL contract for primary evaluation observations."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from engine.experiments.canonical import load_strict_json_text, normalize_utc_timestamp
from engine.experiments.errors import EvidenceValidationError

PROBABILITY_TOLERANCE = 1e-12
REQUIRED = ("row_id", "source_index", "timestamp_utc", "segment_id",
            "target_cluster_id", "signal_episode_id", "policy_population",
            "split_boundary", "class_label", "model_probabilities",
            "comparator_probabilities", "regime", "component", "fallback_tier")


def load_evaluation_rows(path: Path, *, class_order: list[str],
                         allowed_policies: list[str] | None = None,
                         allowed_boundaries: list[str] | None = None,
                         duplicate_policy: str = "reject") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_indices: set[int] = set()
    last_timestamp: dict[str, str] = {}
    closed_segments: set[str] = set()
    active_segment: str | None = None
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise EvidenceValidationError(f"evaluation rows line {number} is blank")
        row = load_strict_json_text(line)
        if not isinstance(row, dict) or any(key not in row for key in REQUIRED):
            raise EvidenceValidationError(f"evaluation rows line {number} lacks required fields")
        row_id = row["row_id"]
        if not isinstance(row_id, str) or not row_id or row_id in seen:
            raise EvidenceValidationError(f"evaluation rows line {number} has duplicate/invalid row_id")
        seen.add(row_id)
        source_index = row["source_index"]
        if (not isinstance(source_index, int) or isinstance(source_index, bool)
                or source_index < 0):
            raise EvidenceValidationError(f"evaluation rows line {number} has invalid source_index")
        if source_index in source_indices:
            raise EvidenceValidationError(f"evaluation rows line {number} has duplicate source_index")
        source_indices.add(source_index)
        if row["class_label"] not in class_order:
            raise EvidenceValidationError(f"evaluation rows line {number} has unknown class")
        row["timestamp_utc"] = normalize_utc_timestamp(row["timestamp_utc"])
        segment = str(row["segment_id"])
        if active_segment is not None and segment != active_segment:
            closed_segments.add(active_segment)
        if segment in closed_segments:
            raise EvidenceValidationError(f"evaluation rows line {number} reopens a closed segment")
        if segment in last_timestamp and row["timestamp_utc"] < last_timestamp[segment]:
            raise EvidenceValidationError(
                f"evaluation rows line {number} moves backward within segment")
        last_timestamp[segment] = row["timestamp_utc"]
        active_segment = segment
        if allowed_policies is not None and row["policy_population"] not in allowed_policies:
            raise EvidenceValidationError(f"evaluation rows line {number} has unregistered policy")
        if allowed_boundaries is not None and row["split_boundary"] not in allowed_boundaries:
            raise EvidenceValidationError(f"evaluation rows line {number} has unregistered boundary")
        for field in ("model_probabilities", "comparator_probabilities"):
            values = row[field]
            if (not isinstance(values, list) or len(values) != len(class_order)
                    or any(isinstance(x, bool) or not isinstance(x, (int, float))
                           or not 0 <= float(x) <= 1 for x in values)
                    or abs(sum(map(float, values)) - 1.0) > PROBABILITY_TOLERANCE):
                raise EvidenceValidationError(f"evaluation rows line {number} has invalid {field}")
        rows.append(row)
    if not rows:
        raise EvidenceValidationError("evaluation row artifact is empty")
    if duplicate_policy != "reject":
        raise EvidenceValidationError(
            "evaluation-row v1 supports only the sealed duplicate policy 'reject'")
    return rows

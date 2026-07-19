"""Dataset binding: hash-locking the exact evaluation data after sealing.

Binding happens only after a plan is sealed, so the plan can never be written
around observed outcomes. The binding records dataset identity (file hashes,
row counts, interval, pair scope), data-quality counters, execution-data
availability, the holdout claim, and the *reduced* promotion ceiling implied
by what the dataset lacks: BID-only data without ask/spread/fill/commission/
latency/impact/notional/conversion evidence can never support more than
FORWARD_TEST_ELIGIBLE.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from engine.experiments.canonical import (
    is_sha256_hex,
    normalize_logical_path,
    normalize_utc_timestamp,
    normalize_uuid,
    sha256_payload,
)
from engine.experiments.errors import DatasetBindingError, PathSecurityError

BINDING_VERSION = "edge-tribunal-dataset-binding-v1"

EXECUTION_DATA_FIELDS: tuple[str, ...] = (
    "ask_available", "spread_available", "fill_available", "commission_available",
    "latency_available", "impact_available", "contract_notional_available",
    "conversion_price_available",
)

_MANIFEST_REQUIRED_FIELDS: tuple[str, ...] = (
    "dataset_name", "provenance", "provenance_claims", "files", "minimum_timestamp_utc",
    "maximum_timestamp_utc", "pair_scope", "pair_order", "frequency", "timestamp_timezone",
    "columns", "missing_row_count", "duplicate_row_count", "segment_count",
    "gap_count", "price_side", "prior_inspection_cutoff_utc", "execution_data",
    "untouched_claim", "untouched_evidence",
)


def dataset_content_fingerprint(manifest: dict[str, Any]) -> str:
    """Return the holdout identity, excluding descriptive/layout metadata.

    Logical names, provenance prose, dataset names, and file ordering cannot
    make identical bytes untouched again.  ``size_bytes`` and ``instrument``
    are included when supplied by a physically verified manifest.
    """
    # Byte hashes are the only per-file values that identify content. Caller
    # row counts, sizes, instrument labels, and paths are separately committed
    # by the manifest/layout hashes and cannot mint a fresh holdout identity.
    files = sorted(item["sha256"] for item in manifest["files"])
    invariant = {
        "files": files,
        "pair_scope": sorted(manifest["pair_scope"]),
        "pair_order": list(manifest["pair_order"]),
        "frequency": manifest["frequency"],
        "price_side": manifest["price_side"],
        "timestamp_timezone": manifest["timestamp_timezone"],
        "minimum_timestamp_utc": normalize_utc_timestamp(manifest["minimum_timestamp_utc"]),
        "maximum_timestamp_utc": normalize_utc_timestamp(manifest["maximum_timestamp_utc"]),
        "columns": sorted(manifest["columns"]),
        "schema_version": manifest.get("schema_version"),
    }
    return sha256_payload(invariant)


def dataset_layout_sha256(manifest: dict[str, Any]) -> str:
    entries = [
        {"logical_path": normalize_logical_path(item["logical_path"]),
         "sha256": item["sha256"]}
        for item in manifest["files"]
    ]
    return sha256_payload(sorted(entries, key=lambda item: item["logical_path"]))


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_dataset_manifest(manifest: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return ["dataset manifest must be a JSON object"]
    for field in _MANIFEST_REQUIRED_FIELDS:
        if field not in manifest:
            errors.append(f"dataset manifest missing required field {field!r}")
    if errors:
        return errors
    if not isinstance(manifest["dataset_name"], str) or not manifest["dataset_name"]:
        errors.append("dataset_name must be a non-empty string")
    provenance = manifest["provenance"]
    if not isinstance(provenance, str) or not provenance.strip():
        errors.append("provenance must be a non-empty string describing dataset origin")
    if not isinstance(manifest["provenance_claims"], list) or not all(
            isinstance(item, str) and item for item in manifest["provenance_claims"]):
        errors.append("provenance_claims must be a list of non-empty strings")
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        errors.append("files must be a non-empty list")
    else:
        seen_paths: set[str] = set()
        for index, entry in enumerate(files):
            prefix = f"files[{index}]"
            if not isinstance(entry, dict):
                errors.append(f"{prefix} must be an object")
                continue
            try:
                logical = normalize_logical_path(entry.get("logical_path", ""))
            except PathSecurityError as exc:
                errors.append(f"{prefix}.logical_path: {exc}")
                logical = None
            if logical is not None:
                if logical in seen_paths:
                    errors.append(f"{prefix}.logical_path duplicates {logical!r}")
                seen_paths.add(logical)
            if not is_sha256_hex(entry.get("sha256")):
                errors.append(f"{prefix}.sha256 must be a 64-char lowercase hex SHA-256")
            if not _nonnegative_int(entry.get("row_count")):
                errors.append(f"{prefix}.row_count must be a non-negative integer")
    for key in ("missing_row_count", "duplicate_row_count", "segment_count", "gap_count"):
        if not _nonnegative_int(manifest[key]):
            errors.append(f"{key} must be a non-negative integer")
    if manifest["timestamp_timezone"] != "UTC":
        errors.append("timestamp_timezone must be exactly 'UTC'")
    if not isinstance(manifest["pair_scope"], list) or \
            not all(isinstance(item, str) and item for item in manifest["pair_scope"]):
        errors.append("pair_scope must be a list of non-empty strings")
    if not isinstance(manifest["pair_order"], list) or \
            not all(isinstance(item, str) and item for item in manifest["pair_order"]):
        errors.append("pair_order must be a list of non-empty strings")
    if not isinstance(manifest["columns"], list) or \
            not all(isinstance(item, str) and item for item in manifest["columns"]):
        errors.append("columns must be a list of non-empty strings")
    execution = manifest["execution_data"]
    if not isinstance(execution, dict):
        errors.append("execution_data must be an object")
    else:
        for field in EXECUTION_DATA_FIELDS:
            if not isinstance(execution.get(field), bool):
                errors.append(f"execution_data.{field} must be a boolean")
    if not isinstance(manifest["untouched_claim"], bool):
        errors.append("untouched_claim must be a boolean")
    elif manifest["untouched_claim"] is not True:
        errors.append("untouched_claim must be exactly true for a clean dataset binding")
    if not isinstance(manifest["untouched_evidence"], str) or not manifest["untouched_evidence"].strip():
        errors.append("untouched_evidence must be a non-empty string")
    return errors


def contract_mismatches(plan: dict[str, Any], dataset_manifest: dict[str, Any]) -> list[str]:
    """Cross-check a structurally valid manifest against the sealed data
    contract. Returned mismatches make the manifest unbindable; callers check
    this *before* claiming a holdout so a bad manifest never burns one."""
    contract = plan["data_contract"]
    mismatches: list[str] = []
    if dataset_manifest["frequency"] != contract["frequency"]:
        mismatches.append(
            f"frequency mismatch: manifest {dataset_manifest['frequency']!r} "
            f"vs plan {contract['frequency']!r}")
    if dataset_manifest["price_side"] != contract["price_side"]:
        mismatches.append(
            f"price_side mismatch: manifest {dataset_manifest['price_side']!r} "
            f"vs plan {contract['price_side']!r}")
    if sorted(dataset_manifest["pair_scope"]) != sorted(contract["universe"]):
        mismatches.append("pair_scope does not match the plan universe")
    if dataset_manifest["pair_order"] != contract["pair_order"]:
        mismatches.append("pair_order does not exactly match the plan pair_order")
    missing_columns = sorted(set(contract["required_columns"]) - set(dataset_manifest["columns"]))
    if missing_columns:
        mismatches.append(f"dataset lacks required columns: {missing_columns}")
    missing_provenance = sorted(set(contract["dataset_provenance_requirements"])
                                - set(dataset_manifest["provenance_claims"]))
    if missing_provenance:
        mismatches.append(f"dataset lacks required provenance claims: {missing_provenance}")

    manifest_min = normalize_utc_timestamp(dataset_manifest["minimum_timestamp_utc"])
    manifest_max = normalize_utc_timestamp(dataset_manifest["maximum_timestamp_utc"])
    holdout_start = normalize_utc_timestamp(contract["untouched_evaluation_start_utc"])
    holdout_end = normalize_utc_timestamp(contract["untouched_evaluation_end_utc"])
    cutoff = normalize_utc_timestamp(dataset_manifest["prior_inspection_cutoff_utc"])
    as_dt = datetime.fromisoformat
    if as_dt(manifest_min) > as_dt(holdout_start) or as_dt(manifest_max) < as_dt(holdout_end):
        mismatches.append(
            f"dataset interval [{manifest_min} .. {manifest_max}] does not cover the "
            f"holdout interval [{holdout_start} .. {holdout_end}]")
    if as_dt(holdout_start) < as_dt(cutoff):
        mismatches.append(
            f"untouched evaluation window begins at {holdout_start}, before the declared "
            f"prior-inspection cutoff {cutoff}: this interval cannot be claimed untouched")
    return mismatches


def build_dataset_binding(
    *,
    plan: dict[str, Any],
    seal: dict[str, Any],
    dataset_manifest: dict[str, Any],
    holdout_claim: dict[str, Any],
    binding_id: str,
    bound_at_utc: str,
) -> dict[str, Any]:
    """Build a binding artifact, cross-checking the manifest against the sealed
    plan's data contract. Raises ``DatasetBindingError`` on any contradiction."""
    errors = validate_dataset_manifest(dataset_manifest)
    if errors:
        raise DatasetBindingError("invalid dataset manifest:\n- " + "\n- ".join(errors))
    mismatches = contract_mismatches(plan, dataset_manifest)
    if mismatches:
        raise DatasetBindingError(
            "dataset manifest contradicts the sealed data contract:\n- " + "\n- ".join(mismatches))

    contract = plan["data_contract"]
    manifest_min = normalize_utc_timestamp(dataset_manifest["minimum_timestamp_utc"])
    manifest_max = normalize_utc_timestamp(dataset_manifest["maximum_timestamp_utc"])
    holdout_start = normalize_utc_timestamp(contract["untouched_evaluation_start_utc"])
    holdout_end = normalize_utc_timestamp(contract["untouched_evaluation_end_utc"])
    cutoff = normalize_utc_timestamp(dataset_manifest["prior_inspection_cutoff_utc"])
    execution = {field: dataset_manifest["execution_data"][field]
                 for field in EXECUTION_DATA_FIELDS}
    execution_complete = all(execution.values())
    plan_ceiling = plan["promotion_ceiling"]["maximum_verdict"]
    # Missing execution realism caps historical evidence at forward-test
    # eligibility regardless of what the plan hoped for.
    if execution_complete:
        maximum_verdict = plan_ceiling
        evidence_level = "LEVEL_2_QUOTE_AND_COST_AWARE_HISTORICAL_RESEARCH"
    else:
        maximum_verdict = ("FORWARD_TEST_ELIGIBLE"
                           if plan_ceiling != "RESEARCH_ONLY" else "RESEARCH_ONLY")
        evidence_level = "LEVEL_1_HISTORICAL_BID_ONLY_RESEARCH"
    if holdout_claim.get("promotion_blocked") or holdout_claim.get("status") == "FORENSIC_REUSE":
        maximum_verdict = "RESEARCH_ONLY"

    total_rows = sum(entry["row_count"] for entry in dataset_manifest["files"])
    binding: dict[str, Any] = {
        "binding_version": BINDING_VERSION,
        "binding_id": normalize_uuid(binding_id),
        "experiment_id": normalize_uuid(plan["identity"]["experiment_id"]),
        "plan_sha256": seal["plan_sha256"],
        "seal_sha256": seal["seal_sha256"],
        "dataset_manifest_sha256": sha256_payload(dataset_manifest),
        "dataset_content_fingerprint": dataset_content_fingerprint(dataset_manifest),
        "dataset_layout_sha256": dataset_layout_sha256(dataset_manifest),
        "dataset_bytes_verified": bool(dataset_manifest.get("dataset_bytes_verified", False)),
        "dataset_name": dataset_manifest["dataset_name"],
        "provenance": dataset_manifest["provenance"],
        "files": [
            {
                "logical_path": normalize_logical_path(entry["logical_path"]),
                "sha256": entry["sha256"],
                "row_count": entry["row_count"],
                **({"size_bytes": entry["size_bytes"]} if "size_bytes" in entry else {}),
            }
            for entry in dataset_manifest["files"]
        ],
        "row_count_total": total_rows,
        "minimum_timestamp_utc": manifest_min,
        "maximum_timestamp_utc": manifest_max,
        "pair_scope": sorted(dataset_manifest["pair_scope"]),
        "frequency": dataset_manifest["frequency"],
        "timestamp_timezone": "UTC",
        "required_columns": list(contract["required_columns"]),
        "missing_row_count": dataset_manifest["missing_row_count"],
        "duplicate_row_count": dataset_manifest["duplicate_row_count"],
        "segment_count": dataset_manifest["segment_count"],
        "gap_count": dataset_manifest["gap_count"],
        "price_side": dataset_manifest["price_side"],
        "execution_data": execution,
        "execution_data_complete": execution_complete,
        "holdout_interval": {"start_utc": holdout_start, "end_utc": holdout_end},
        "prior_inspection_cutoff_utc": cutoff,
        "untouched_claim": dataset_manifest["untouched_claim"],
        "untouched_evidence": dataset_manifest["untouched_evidence"],
        "holdout_id": holdout_claim["holdout_id"],
        "holdout_claim_token": holdout_claim.get("claim_token"),
        "holdout_registry_revision": holdout_claim.get("registry_revision"),
        "holdout_claim_status": holdout_claim["status"],
        "holdout_reuse_kind": holdout_claim["reuse_kind"],
        "promotion_blocked_by_holdout": bool(holdout_claim.get("promotion_blocked")),
        "maximum_verdict_after_binding": maximum_verdict,
        "maximum_evidence_level": evidence_level,
        "bound_at_utc": normalize_utc_timestamp(bound_at_utc),
    }
    binding["binding_sha256"] = sha256_payload(binding)
    return binding


def verify_binding_integrity(binding: dict[str, Any]) -> bool:
    if "binding_sha256" not in binding:
        return False
    body = {key: value for key, value in binding.items() if key != "binding_sha256"}
    return sha256_payload(body) == binding["binding_sha256"]

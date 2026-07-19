"""Fail-closed reconciliation between experiment snapshots and holdout registry."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from engine.experiments import holdout_registry
from engine.experiments.canonical import sha256_payload
from engine.experiments.errors import TribunalError


def reconcile_binding(binding: dict[str, Any], registry_root: Path) -> dict[str, Any]:
    document = holdout_registry.load_registry(registry_root)
    matches = [entry for entry in document["entries"]
               if entry["holdout_id"] == binding.get("holdout_id")]
    if len(matches) != 1:
        raise TribunalError("registry/binding disagreement: holdout claim missing or duplicated")
    entry = matches[0]
    expected = {
        "experiment_id": binding["experiment_id"],
        "plan_sha256": binding["plan_sha256"],
        "dataset_fingerprint": binding["dataset_content_fingerprint"],
        "interval_start_utc": binding["holdout_interval"]["start_utc"],
        "interval_end_utc": binding["holdout_interval"]["end_utc"],
    }
    mismatches = [key for key, value in expected.items() if entry.get(key) != value]
    if mismatches:
        raise TribunalError(f"registry/binding disagreement in {mismatches}")
    token_payload = {
        "holdout_id": entry["holdout_id"], "experiment_id": entry["experiment_id"],
        "plan_sha256": entry["plan_sha256"], "claim_revision": entry["claim_revision"],
        "dataset_fingerprint": entry["dataset_fingerprint"],
        "claimed_at_utc": entry["claimed_at_utc"],
        "previous_registry_sha256": entry["previous_registry_sha256"],
    }
    if sha256_payload(token_payload) != entry.get("claim_token"):
        raise TribunalError("registry claim token is invalid")
    if binding.get("holdout_claim_token") != entry["claim_token"] or \
            binding.get("holdout_registry_revision") != entry["claim_revision"]:
        raise TribunalError("registry claim token/revision differs from dataset binding")
    receipt = {"holdout_id": entry["holdout_id"], "status": entry["status"],
               "registry_revision": document.get("revision", 0),
               "registry_sha256": document["registry_sha256"]}
    receipt["receipt_sha256"] = sha256_payload(receipt)
    return receipt


def reconcile_verdict(binding: dict[str, Any], registry_root: Path) -> dict[str, Any]:
    reconcile_binding(binding, registry_root)
    return holdout_registry.consume_holdout_idempotent(registry_root, binding["holdout_id"])

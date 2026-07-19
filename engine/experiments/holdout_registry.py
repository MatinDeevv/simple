"""Persistent holdout registry.

A holdout interval on a given dataset fingerprint can be presented as
"untouched" exactly once. The registry records every claim, detects exact and
partial interval overlap, survives renamed experiments and new model commits
(neither makes old data untouched again), and permits reuse only as an
explicitly declared forensic reuse that automatically blocks promotion.

Updates are transactional (temp file + atomic replace) under an exclusive
lock file, so two concurrent claims of the same holdout produce exactly one
clean claimant. The registry file carries an integrity hash over its entries
so out-of-band tampering is detectable.
"""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from engine.experiments.canonical import (
    is_sha256_hex,
    load_strict_json_text,
    normalize_utc_timestamp,
    normalize_uuid,
    sha256_payload,
    strict_json_text,
)
from engine.experiments.errors import (
    CanonicalJsonError,
    HoldoutReuseError,
    LockError,
    TribunalError,
)

UNUSED = "UNUSED"
SEALED_FOR_EXPERIMENT = "SEALED_FOR_EXPERIMENT"
CONSUMED = "CONSUMED"
FORENSIC_REUSE = "FORENSIC_REUSE"
INVALIDATED = "INVALIDATED"

STATUSES: tuple[str, ...] = (UNUSED, SEALED_FOR_EXPERIMENT, CONSUMED, FORENSIC_REUSE, INVALIDATED)

REGISTRY_FILENAME = "holdout-registry.json"
LOCK_FILENAME = "holdout-registry.lock"
REGISTRY_VERSION = "edge-tribunal-holdout-registry-v1"

_DEFAULT_STALE_LOCK_SECONDS = 600.0


class RegistryLock:
    """Exclusive advisory lock via O_CREAT|O_EXCL. Stale locks (older than
    ``stale_after_seconds``) are broken once, loudly, on the next acquire."""

    def __init__(self, root: Path, *, timeout_seconds: float = 5.0,
                 stale_after_seconds: float = _DEFAULT_STALE_LOCK_SECONDS) -> None:
        self.path = Path(root) / LOCK_FILENAME
        self.timeout_seconds = timeout_seconds
        self.stale_after_seconds = stale_after_seconds
        self._acquired = False

    def __enter__(self) -> "RegistryLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(handle, f"pid={os.getpid()}\n".encode("utf-8"))
                os.close(handle)
                self._acquired = True
                return self
            except (FileExistsError, PermissionError):
                # Windows may report a sharing violation as PermissionError while
                # another thread is creating or unlinking the lock file. Treat it
                # as contention and retry through the same bounded lock path.
                if self._is_stale():
                    try:
                        self.path.unlink()
                    except OSError:
                        pass
                    continue
                if time.monotonic() >= deadline:
                    raise LockError(f"could not acquire holdout registry lock at {self.path}")
                time.sleep(0.05)

    def _is_stale(self) -> bool:
        try:
            age = time.time() - self.path.stat().st_mtime
        except OSError:
            return False
        return age > self.stale_after_seconds

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._acquired:
            try:
                self.path.unlink()
            except OSError:
                pass
            self._acquired = False


def _entries_sha256(entries: list[dict[str, Any]]) -> str:
    return sha256_payload(entries)


def _empty_registry() -> dict[str, Any]:
    document: dict[str, Any] = {"registry_version": REGISTRY_VERSION, "entries": []}
    document["registry_sha256"] = _entries_sha256(document["entries"])
    return document


def load_registry(root: Path) -> dict[str, Any]:
    """Load and integrity-check the registry; a missing file is an empty registry."""
    path = Path(root) / REGISTRY_FILENAME
    if not path.is_file():
        return _empty_registry()
    try:
        document = load_strict_json_text(path.read_text(encoding="utf-8"))
    except CanonicalJsonError as exc:
        raise TribunalError(f"holdout registry is unreadable: {exc}") from exc
    if not isinstance(document, dict) or document.get("registry_version") != REGISTRY_VERSION:
        raise TribunalError("holdout registry has an unknown format")
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise TribunalError("holdout registry entries must be a list")
    if document.get("registry_sha256") != _entries_sha256(entries):
        raise TribunalError("holdout registry integrity hash mismatch: registry was tampered "
                            "with or corrupted")
    return document


def _save_registry(root: Path, document: dict[str, Any]) -> None:
    document["registry_sha256"] = _entries_sha256(document["entries"])
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / REGISTRY_FILENAME
    staging = root / f".{REGISTRY_FILENAME}.staging-{uuid.uuid4().hex}"
    staging.write_text(strict_json_text(document), encoding="utf-8")
    os.replace(staging, final)


def _parse_interval(start_utc: str, end_utc: str) -> tuple[datetime, datetime]:
    start = datetime.fromisoformat(normalize_utc_timestamp(start_utc))
    end = datetime.fromisoformat(normalize_utc_timestamp(end_utc))
    if start >= end:
        raise TribunalError(f"holdout interval start must precede end ({start_utc} .. {end_utc})")
    return start, end


def find_overlaps(
    document: dict[str, Any],
    *,
    dataset_fingerprint: str,
    interval_start_utc: str,
    interval_end_utc: str,
) -> list[dict[str, Any]]:
    """Overlapping prior entries for a fingerprint, each annotated exact/partial.

    Overlap is keyed on the dataset fingerprint and time interval only:
    renaming an experiment, changing its title, bumping the model commit, or
    switching the hypothesis family never makes previously seen data
    untouched again, so none of those fields participate in overlap detection.
    """
    start, end = _parse_interval(interval_start_utc, interval_end_utc)
    overlaps: list[dict[str, Any]] = []
    for entry in document["entries"]:
        if entry["dataset_fingerprint"] != dataset_fingerprint:
            continue
        entry_start, entry_end = _parse_interval(
            entry["interval_start_utc"], entry["interval_end_utc"])
        if start < entry_end and entry_start < end:
            exact = (entry_start == start and entry_end == end)
            overlaps.append({
                "holdout_id": entry["holdout_id"],
                "status": entry["status"],
                "experiment_id": entry["experiment_id"],
                "hypothesis_family": entry["hypothesis_family"],
                "overlap_kind": "exact" if exact else "partial",
            })
    return overlaps


def claim_holdout(
    root: Path,
    *,
    dataset_fingerprint: str,
    universe: list[str],
    interval_start_utc: str,
    interval_end_utc: str,
    target_contract_sha256: str,
    model_family: str,
    hypothesis_family: str,
    experiment_id: str,
    plan_sha256: str,
    claimed_at_utc: str,
    declare_forensic_reuse: bool = False,
    holdout_id: str | None = None,
    lock_timeout_seconds: float = 5.0,
    stale_lock_seconds: float = _DEFAULT_STALE_LOCK_SECONDS,
) -> dict[str, Any]:
    """Claim a holdout interval for one experiment.

    A clean (untouched) claim succeeds only when no prior entry overlaps the
    interval on the same dataset fingerprint. Overlap with any prior claim
    raises ``HoldoutReuseError`` unless the caller explicitly declares
    forensic reuse, which records the claim with ``FORENSIC_REUSE`` status and
    ``promotion_blocked: true``.
    """
    if not is_sha256_hex(dataset_fingerprint):
        raise TribunalError("dataset_fingerprint must be a SHA-256 hex digest")
    if not is_sha256_hex(target_contract_sha256):
        raise TribunalError("target_contract_sha256 must be a SHA-256 hex digest")
    if not is_sha256_hex(plan_sha256):
        raise TribunalError("plan_sha256 must be a SHA-256 hex digest")
    experiment_id = normalize_uuid(experiment_id)
    claimed_at_utc = normalize_utc_timestamp(claimed_at_utc)
    _parse_interval(interval_start_utc, interval_end_utc)

    with RegistryLock(root, timeout_seconds=lock_timeout_seconds,
                      stale_after_seconds=stale_lock_seconds):
        document = load_registry(root)
        overlaps = find_overlaps(
            document,
            dataset_fingerprint=dataset_fingerprint,
            interval_start_utc=interval_start_utc,
            interval_end_utc=interval_end_utc,
        )
        blocking = [item for item in overlaps if item["status"] != UNUSED]
        unused_exact = [item for item in overlaps
                        if item["status"] == UNUSED and item["overlap_kind"] == "exact"]
        if blocking and not declare_forensic_reuse:
            details = "; ".join(
                f"{item['holdout_id']} ({item['status']}, {item['overlap_kind']} overlap)"
                for item in blocking)
            raise HoldoutReuseError(
                f"holdout interval {interval_start_utc}..{interval_end_utc} on this dataset "
                f"fingerprint is not untouched: {details}. Reuse is only allowed as an "
                f"explicitly declared forensic reuse, which blocks promotion.")
        status = FORENSIC_REUSE if (blocking and declare_forensic_reuse) else SEALED_FOR_EXPERIMENT
        if unused_exact and status == SEALED_FOR_EXPERIMENT:
            holdout_id = unused_exact[0]["holdout_id"]
            for entry in document["entries"]:
                if entry["holdout_id"] == holdout_id:
                    entry["status"] = SEALED_FOR_EXPERIMENT
                    entry["experiment_id"] = experiment_id
                    entry["plan_sha256"] = plan_sha256
                    entry["claimed_at_utc"] = claimed_at_utc
                    break
        else:
            # Deterministic identity: the same claim by the same experiment on
            # the same interval always mints the same holdout ID.
            holdout_id = holdout_id or str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"edge-tribunal-holdout:{dataset_fingerprint}:"
                f"{interval_start_utc}:{interval_end_utc}:{experiment_id}"))
            document["entries"].append({
                "holdout_id": holdout_id,
                "dataset_fingerprint": dataset_fingerprint,
                "universe": sorted(universe),
                "interval_start_utc": normalize_utc_timestamp(interval_start_utc),
                "interval_end_utc": normalize_utc_timestamp(interval_end_utc),
                "target_contract_sha256": target_contract_sha256,
                "model_family": model_family,
                "hypothesis_family": hypothesis_family,
                "status": status,
                "experiment_id": experiment_id,
                "plan_sha256": plan_sha256,
                "claimed_at_utc": claimed_at_utc,
            })
        _save_registry(root, document)
    return {
        "holdout_id": holdout_id,
        "status": status,
        "reuse_kind": ("exact" if any(item["overlap_kind"] == "exact" for item in blocking)
                       else "partial") if blocking else "none",
        "overlapping_holdout_ids": [item["holdout_id"] for item in overlaps],
        "promotion_blocked": status == FORENSIC_REUSE,
    }


def _update_status(root: Path, holdout_id: str, *, expected: tuple[str, ...],
                   new_status: str, lock_timeout_seconds: float = 5.0) -> None:
    with RegistryLock(root, timeout_seconds=lock_timeout_seconds):
        document = load_registry(root)
        for entry in document["entries"]:
            if entry["holdout_id"] == holdout_id:
                if entry["status"] not in expected:
                    raise TribunalError(
                        f"holdout {holdout_id} is {entry['status']}, expected one of {expected}")
                entry["status"] = new_status
                _save_registry(root, document)
                return
        raise TribunalError(f"holdout {holdout_id} is not registered")


def consume_holdout(root: Path, holdout_id: str) -> None:
    """Mark a claimed holdout CONSUMED when its experiment's verdict issues."""
    _update_status(root, holdout_id, expected=(SEALED_FOR_EXPERIMENT, FORENSIC_REUSE),
                   new_status=CONSUMED)


def invalidate_holdout(root: Path, holdout_id: str) -> None:
    _update_status(root, holdout_id,
                   expected=(UNUSED, SEALED_FOR_EXPERIMENT, CONSUMED, FORENSIC_REUSE),
                   new_status=INVALIDATED)


def holdout_status(root: Path, holdout_id: str) -> str:
    document = load_registry(root)
    for entry in document["entries"]:
        if entry["holdout_id"] == holdout_id:
            return entry["status"]
    raise TribunalError(f"holdout {holdout_id} is not registered")

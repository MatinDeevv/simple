"""Append-only, hash-chained JSONL audit log.

Every state transition appends exactly one event. Each event commits to the
previous event's hash, its own payload hash, and the state transition it
records, so deletion, reordering, modification, insertion, duplicate IDs,
backward timestamps, and mid-chain experiment-ID switches are all detectable.

Honesty note (see docs/edge-tribunal.md): the chain is *tamper-evident* only
when a trusted copy of the latest event hash is retained somewhere the log's
owner cannot rewrite. It is not a blockchain, not a digital signature, and
not proof of any human identity.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from engine.experiments import state_machine
from engine.experiments.canonical import (
    canonical_json_bytes,
    is_sha256_hex,
    load_strict_json_text,
    normalize_utc_timestamp,
    sha256_bytes,
    sha256_payload,
)
from engine.experiments.errors import (
    AuditIntegrityError,
    CanonicalJsonError,
    InvalidStateTransitionError,
)

GENESIS_HASH = "0" * 64

EVENT_TYPES: frozenset[str] = frozenset({
    "PLAN_CREATED",
    "PLAN_AMENDED",
    "PLAN_SEALED",
    "DATASET_BOUND",
    "HOLDOUT_CLAIMED",
    "EVIDENCE_RECORDED",
    "GATES_EVALUATED",
    "VERDICT_ISSUED",
    "EXPERIMENT_ARCHIVED",
    "VERIFICATION_FAILED",
    "FORENSIC_REUSE_DECLARED",
    "HOLDOUT_CONSUMED",
    "REGISTRY_RECONCILED",
})

_EVENT_FIELDS = (
    "event_id", "experiment_id", "event_type", "timestamp_utc", "actor",
    "state_before", "state_after", "payload_sha256", "previous_event_sha256",
    "event_sha256",
)

# Event types that record an actual state transition; the rest are
# informational and must keep state_before == state_after.
_TRANSITION_EVENTS = frozenset({
    "PLAN_SEALED", "DATASET_BOUND", "EVIDENCE_RECORDED",
    "VERDICT_ISSUED", "EXPERIMENT_ARCHIVED",
})


def compute_event_hash(event: dict[str, Any]) -> str:
    body = {key: value for key, value in event.items() if key != "event_sha256"}
    return sha256_bytes(canonical_json_bytes(body))


def build_event(
    *,
    event_id: str,
    experiment_id: str,
    event_type: str,
    timestamp_utc: str,
    actor: str,
    state_before: str,
    state_after: str,
    payload: Any,
    previous_event_sha256: str,
) -> dict[str, Any]:
    if event_type not in EVENT_TYPES:
        raise AuditIntegrityError(f"unknown audit event type: {event_type!r}")
    if not is_sha256_hex(previous_event_sha256):
        raise AuditIntegrityError("previous_event_sha256 must be 64 lowercase hex chars")
    event = {
        "event_id": event_id,
        "experiment_id": experiment_id,
        "event_type": event_type,
        "timestamp_utc": normalize_utc_timestamp(timestamp_utc),
        "actor": actor,
        "state_before": state_before,
        "state_after": state_after,
        "payload_sha256": sha256_payload(payload),
        "previous_event_sha256": previous_event_sha256,
    }
    event["event_sha256"] = compute_event_hash(event)
    return event


def serialize_events(events: list[dict[str, Any]]) -> str:
    lines = [canonical_json_bytes(event).decode("utf-8") for event in events]
    return "\n".join(lines) + ("\n" if lines else "")


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AuditIntegrityError(f"audit log does not exist: {path}")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise AuditIntegrityError(f"audit log has a blank line at line {line_number}")
        try:
            event = load_strict_json_text(line)
        except CanonicalJsonError as exc:
            raise AuditIntegrityError(f"audit log line {line_number} is invalid JSON: {exc}") from exc
        if not isinstance(event, dict):
            raise AuditIntegrityError(f"audit log line {line_number} is not a JSON object")
        events.append(event)
    return events


def verify_chain(events: list[dict[str, Any]]) -> None:
    """Raise ``AuditIntegrityError`` on any structural or chain defect."""
    if not events:
        raise AuditIntegrityError("audit chain is empty")
    seen_event_ids: set[str] = set()
    experiment_id = None
    previous_hash = GENESIS_HASH
    previous_timestamp: datetime | None = None
    current_state: str | None = None
    for index, event in enumerate(events):
        where = f"audit event {index}"
        missing = [field for field in _EVENT_FIELDS if field not in event]
        if missing:
            raise AuditIntegrityError(f"{where}: missing fields {missing}")
        extra = sorted(set(event) - set(_EVENT_FIELDS))
        if extra:
            raise AuditIntegrityError(f"{where}: unexpected fields {extra}")
        if event["event_type"] not in EVENT_TYPES:
            raise AuditIntegrityError(f"{where}: unknown event type {event['event_type']!r}")
        if event["event_id"] in seen_event_ids:
            raise AuditIntegrityError(f"{where}: duplicate event_id {event['event_id']!r}")
        seen_event_ids.add(event["event_id"])
        if experiment_id is None:
            experiment_id = event["experiment_id"]
        elif event["experiment_id"] != experiment_id:
            raise AuditIntegrityError(
                f"{where}: experiment_id changed mid-chain "
                f"({experiment_id!r} -> {event['experiment_id']!r})")
        if event["previous_event_sha256"] != previous_hash:
            raise AuditIntegrityError(f"{where}: previous_event_sha256 does not match the chain")
        recomputed = compute_event_hash(event)
        if event["event_sha256"] != recomputed:
            raise AuditIntegrityError(f"{where}: event_sha256 does not match event contents")
        timestamp = event["timestamp_utc"]
        try:
            normalized_timestamp = normalize_utc_timestamp(timestamp)
        except CanonicalJsonError as exc:
            raise AuditIntegrityError(f"{where}: invalid timestamp: {exc}") from exc
        if normalized_timestamp != timestamp:
            raise AuditIntegrityError(f"{where}: timestamp is not in canonical UTC form")
        parsed_timestamp = datetime.fromisoformat(timestamp)
        if previous_timestamp is not None and parsed_timestamp < previous_timestamp:
            raise AuditIntegrityError(f"{where}: timestamp moves backward")
        previous_timestamp = parsed_timestamp
        state_before = event["state_before"]
        state_after = event["state_after"]
        if event["event_type"] in _TRANSITION_EVENTS:
            try:
                state_machine.validate_transition(state_before, state_after)
            except InvalidStateTransitionError as exc:
                raise AuditIntegrityError(f"{where}: {exc}") from exc
        elif state_before != state_after:
            raise AuditIntegrityError(
                f"{where}: informational event {event['event_type']!r} must not change state")
        if current_state is not None and state_before != current_state:
            raise AuditIntegrityError(
                f"{where}: state_before {state_before!r} disagrees with chain state {current_state!r}")
        current_state = state_after
        previous_hash = event["event_sha256"]


def chain_head_hash(events: list[dict[str, Any]]) -> str:
    if not events:
        return GENESIS_HASH
    return events[-1]["event_sha256"]


def chain_state(events: list[dict[str, Any]]) -> str:
    verify_chain(events)
    return events[-1]["state_after"]

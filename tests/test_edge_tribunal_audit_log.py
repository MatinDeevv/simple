from __future__ import annotations

import copy
from pathlib import Path

import pytest

from engine.experiments import audit_log
from engine.experiments.errors import AuditIntegrityError

EXPERIMENT_ID = "00000000-0000-4000-8000-000000000001"


def build_chain() -> list[dict]:
    """A valid five-event chain: created, sealed, bound, evidence, verdict."""
    stages = [
        ("PLAN_CREATED", "DRAFT", "DRAFT"),
        ("PLAN_SEALED", "DRAFT", "SEALED"),
        ("DATASET_BOUND", "SEALED", "DATA_BOUND"),
        ("EVIDENCE_RECORDED", "DATA_BOUND", "EVIDENCE_RECORDED"),
        ("VERDICT_ISSUED", "EVIDENCE_RECORDED", "VERDICT_ISSUED"),
    ]
    events: list[dict] = []
    previous = audit_log.GENESIS_HASH
    for index, (event_type, before, after) in enumerate(stages):
        event = audit_log.build_event(
            event_id=f"event-{index}", experiment_id=EXPERIMENT_ID,
            event_type=event_type,
            timestamp_utc=f"2026-01-02T00:0{index}:00+00:00", actor="tester",
            state_before=before, state_after=after,
            payload={"index": index}, previous_event_sha256=previous)
        events.append(event)
        previous = event["event_sha256"]
    return events


def test_valid_chain_verifies() -> None:
    events = build_chain()
    audit_log.verify_chain(events)
    assert audit_log.chain_state(events) == "VERDICT_ISSUED"
    assert audit_log.chain_head_hash(events) == events[-1]["event_sha256"]


def test_empty_chain_fails() -> None:
    with pytest.raises(AuditIntegrityError, match="empty"):
        audit_log.verify_chain([])


def test_deleted_event_fails() -> None:
    events = build_chain()
    del events[2]
    with pytest.raises(AuditIntegrityError):
        audit_log.verify_chain(events)


def test_deleted_first_event_fails() -> None:
    events = build_chain()
    del events[0]
    with pytest.raises(AuditIntegrityError):
        audit_log.verify_chain(events)


def test_modified_event_fails() -> None:
    events = build_chain()
    events[1]["actor"] = "attacker"
    with pytest.raises(AuditIntegrityError, match="event_sha256"):
        audit_log.verify_chain(events)


def test_reordered_events_fail() -> None:
    events = build_chain()
    events[1], events[2] = events[2], events[1]
    with pytest.raises(AuditIntegrityError):
        audit_log.verify_chain(events)


def test_inserted_event_fails() -> None:
    events = build_chain()
    forged = copy.deepcopy(events[2])
    forged["event_id"] = "forged"
    events.insert(3, forged)
    with pytest.raises(AuditIntegrityError):
        audit_log.verify_chain(events)


def test_duplicate_event_id_fails() -> None:
    events = build_chain()
    events[3]["event_id"] = events[2]["event_id"]
    # Recompute downstream hashes so only the duplicate ID is wrong.
    previous = events[2]["event_sha256"]
    for event in events[3:]:
        event["previous_event_sha256"] = previous
        event["event_sha256"] = audit_log.compute_event_hash(event)
        previous = event["event_sha256"]
    with pytest.raises(AuditIntegrityError, match="duplicate event_id"):
        audit_log.verify_chain(events)


def test_wrong_previous_hash_fails() -> None:
    events = build_chain()
    events[2]["previous_event_sha256"] = "f" * 64
    events[2]["event_sha256"] = audit_log.compute_event_hash(events[2])
    with pytest.raises(AuditIntegrityError, match="previous_event_sha256"):
        audit_log.verify_chain(events)


def _rehash_from(events: list[dict], start: int) -> None:
    previous = events[start - 1]["event_sha256"] if start else audit_log.GENESIS_HASH
    for event in events[start:]:
        event["previous_event_sha256"] = previous
        event["event_sha256"] = audit_log.compute_event_hash(event)
        previous = event["event_sha256"]


def test_state_inconsistency_fails() -> None:
    events = build_chain()
    events[2]["state_before"] = "DRAFT"  # chain is in SEALED here
    _rehash_from(events, 2)
    with pytest.raises(AuditIntegrityError):
        audit_log.verify_chain(events)


def test_invalid_state_transition_fails() -> None:
    events = build_chain()
    events[2]["state_after"] = "VERDICT_ISSUED"  # skips two states
    _rehash_from(events, 2)
    with pytest.raises(AuditIntegrityError, match="skips"):
        audit_log.verify_chain(events)


def test_backward_timestamp_fails() -> None:
    events = build_chain()
    events[3]["timestamp_utc"] = "2026-01-01T00:00:00+00:00"
    _rehash_from(events, 3)
    with pytest.raises(AuditIntegrityError, match="backward"):
        audit_log.verify_chain(events)


def test_experiment_id_switch_fails() -> None:
    events = build_chain()
    events[3]["experiment_id"] = "00000000-0000-4000-8000-000000000099"
    _rehash_from(events, 3)
    with pytest.raises(AuditIntegrityError, match="mid-chain"):
        audit_log.verify_chain(events)


def test_informational_event_must_not_change_state() -> None:
    events = build_chain()[:2]
    bad = audit_log.build_event(
        event_id="info-1", experiment_id=EXPERIMENT_ID,
        event_type="HOLDOUT_CLAIMED", timestamp_utc="2026-01-02T00:03:00+00:00",
        actor="tester", state_before="SEALED", state_after="DATA_BOUND",
        payload={}, previous_event_sha256=events[-1]["event_sha256"])
    events.append(bad)
    with pytest.raises(AuditIntegrityError, match="must not change state"):
        audit_log.verify_chain(events)


def test_unknown_event_type_rejected_at_build() -> None:
    with pytest.raises(AuditIntegrityError, match="unknown audit event type"):
        audit_log.build_event(
            event_id="x", experiment_id=EXPERIMENT_ID, event_type="TRADE_EXECUTED",
            timestamp_utc="2026-01-02T00:00:00+00:00", actor="tester",
            state_before="DRAFT", state_after="DRAFT", payload={},
            previous_event_sha256=audit_log.GENESIS_HASH)


def test_naive_timestamp_rejected_at_build() -> None:
    from engine.experiments.errors import CanonicalJsonError
    with pytest.raises(CanonicalJsonError, match="timezone"):
        audit_log.build_event(
            event_id="x", experiment_id=EXPERIMENT_ID, event_type="PLAN_CREATED",
            timestamp_utc="2026-01-02T00:00:00", actor="tester",
            state_before="DRAFT", state_after="DRAFT", payload={},
            previous_event_sha256=audit_log.GENESIS_HASH)


def test_serialization_roundtrip(tmp_path: Path) -> None:
    events = build_chain()
    path = tmp_path / "audit.jsonl"
    path.write_text(audit_log.serialize_events(events), encoding="utf-8")
    loaded = audit_log.read_events(path)
    audit_log.verify_chain(loaded)
    assert loaded == events


def test_manually_edited_audit_file_detected(tmp_path: Path) -> None:
    events = build_chain()
    path = tmp_path / "audit.jsonl"
    text = audit_log.serialize_events(events)
    path.write_text(text.replace("tester", "someone-else"), encoding="utf-8")
    with pytest.raises(AuditIntegrityError):
        audit_log.verify_chain(audit_log.read_events(path))


def test_thousand_event_chain_verifies_quickly() -> None:
    events: list[dict] = []
    previous = audit_log.GENESIS_HASH
    for index in range(1000):
        event = audit_log.build_event(
            event_id=f"event-{index}", experiment_id=EXPERIMENT_ID,
            event_type="VERIFICATION_FAILED",
            timestamp_utc=f"2026-01-02T{index // 3600:02d}:"
                          f"{(index // 60) % 60:02d}:{index % 60:02d}+00:00",
            actor="tester", state_before="DRAFT", state_after="DRAFT",
            payload={"index": index}, previous_event_sha256=previous)
        events.append(event)
        previous = event["event_sha256"]
    audit_log.verify_chain(events)
    assert audit_log.chain_head_hash(events) == events[-1]["event_sha256"]

"""Irreversible experiment state machine.

DRAFT -> SEALED -> DATA_BOUND -> EVIDENCE_RECORDED -> VERDICT_ISSUED -> ARCHIVED

No backward transitions, no skipped states, no repeated transitions. A sealed
experiment can never return to DRAFT; the only way to change a sealed plan is
to create a *new* experiment that records its parent and an amendment reason.
"""

from __future__ import annotations

from engine.experiments.errors import InvalidStateTransitionError

DRAFT = "DRAFT"
SEALED = "SEALED"
DATA_BOUND = "DATA_BOUND"
EVIDENCE_RECORDED = "EVIDENCE_RECORDED"
VERDICT_ISSUED = "VERDICT_ISSUED"
ARCHIVED = "ARCHIVED"

STATES: tuple[str, ...] = (DRAFT, SEALED, DATA_BOUND, EVIDENCE_RECORDED, VERDICT_ISSUED, ARCHIVED)

ALLOWED_TRANSITIONS: dict[str, str] = {
    DRAFT: SEALED,
    SEALED: DATA_BOUND,
    DATA_BOUND: EVIDENCE_RECORDED,
    EVIDENCE_RECORDED: VERDICT_ISSUED,
    VERDICT_ISSUED: ARCHIVED,
}

# The state each artifact-producing operation requires and produces.
TRANSITION_FOR_OPERATION: dict[str, tuple[str, str]] = {
    "seal": (DRAFT, SEALED),
    "bind-data": (SEALED, DATA_BOUND),
    "record-evidence": (DATA_BOUND, EVIDENCE_RECORDED),
    "evaluate": (EVIDENCE_RECORDED, VERDICT_ISSUED),
    "archive": (VERDICT_ISSUED, ARCHIVED),
}


def state_index(state: str) -> int:
    if state not in STATES:
        raise InvalidStateTransitionError(f"unknown experiment state: {state!r}")
    return STATES.index(state)


def validate_transition(state_before: str, state_after: str) -> None:
    """Raise unless ``state_before -> state_after`` is the single allowed step."""
    before_index = state_index(state_before)
    after_index = state_index(state_after)
    if state_after == state_before:
        raise InvalidStateTransitionError(
            f"repeated transition: experiment is already in state {state_before}")
    if after_index < before_index:
        raise InvalidStateTransitionError(
            f"backward transition {state_before} -> {state_after} is forbidden")
    if ALLOWED_TRANSITIONS.get(state_before) != state_after:
        raise InvalidStateTransitionError(
            f"transition {state_before} -> {state_after} skips required states")


def require_state(current: str, expected: str, *, operation: str) -> None:
    if current != expected:
        raise InvalidStateTransitionError(
            f"operation {operation!r} requires state {expected}, experiment is in {current}")

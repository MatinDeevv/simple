"""Exception hierarchy for the Edge Tribunal.

Every failure mode the Tribunal can detect maps to a dedicated exception so
callers (and the CLI) can distinguish "the researcher made an invalid request"
from "an artifact chain has been tampered with". All exceptions derive from
``TribunalError`` so a single ``except TribunalError`` catches every ordinary,
expected failure without swallowing programming defects.
"""

from __future__ import annotations


class TribunalError(RuntimeError):
    """Base class for every expected Edge Tribunal failure."""


class CanonicalJsonError(TribunalError):
    """Payload cannot be canonically serialized (NaN, infinity, unsupported type)."""


class PathSecurityError(TribunalError):
    """A logical path is absolute, escapes its root, or is otherwise unsafe."""


class PlanValidationError(TribunalError):
    """A preregistration plan is structurally or semantically invalid."""


class PlanMutationError(TribunalError):
    """A sealed plan's bytes no longer match the sealed hash commitment."""


class InvalidStateTransitionError(TribunalError):
    """A state-machine transition is skipped, reversed, or repeated."""


class DatasetBindingError(TribunalError):
    """A dataset binding contradicts the sealed plan's data contract."""


class EvidenceValidationError(TribunalError):
    """An evidence bundle is incomplete, inconsistent, or mismatched."""


class HoldoutReuseError(TribunalError):
    """A holdout interval was presented as untouched but is already claimed."""


class GateEvaluationError(TribunalError):
    """Gate evaluation cannot proceed (not a gate merely failing)."""


class VerdictIntegrityError(TribunalError):
    """A verdict artifact is missing, duplicated, or hash-inconsistent."""


class AuditIntegrityError(TribunalError):
    """The append-only audit chain fails verification."""


class TransactionError(TribunalError):
    """A transactional state transition could not be completed safely."""


class LockError(TribunalError):
    """A required file lock could not be acquired."""

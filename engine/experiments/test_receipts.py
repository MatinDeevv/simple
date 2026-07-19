"""Physical verification of immutable test receipts and their logs."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engine.experiments.artifact_binding import sha256_file
from engine.experiments.canonical import (is_sha256_hex, load_strict_json_text,
                                          normalize_utc_timestamp, sha256_payload)
from engine.experiments.errors import EvidenceValidationError

RECEIPT_VERSION = "test-receipt-v2"
REQUIRED_FIELDS = frozenset({
    "receipt_version", "command_argv", "exit_code", "commit_sha",
    "started_at_utc", "completed_at_utc", "python_version",
    "dependency_snapshot_sha256", "logical_log_path", "log_sha256",
    "runner_version", "working_tree_status", "command_environment_policy",
    "receipt_sha256",
})


def verify_test_receipt(receipt_path: Path, log_path: Path, *, expected_commit: str,
                        required_command: list[str], now_utc: str | None = None) -> dict[str, Any]:
    receipt = load_strict_json_text(Path(receipt_path).read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("receipt_version") != RECEIPT_VERSION:
        raise EvidenceValidationError("unsupported test receipt version")
    if set(receipt) != REQUIRED_FIELDS:
        raise EvidenceValidationError("test receipt fields do not match strict v2 contract")
    claimed = receipt.get("receipt_sha256")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if not is_sha256_hex(claimed) or sha256_payload(body) != claimed:
        raise EvidenceValidationError("test receipt self-hash mismatch")
    if receipt.get("commit_sha") != expected_commit:
        raise EvidenceValidationError("test receipt commit mismatch")
    if receipt.get("command_argv") != required_command or not required_command:
        raise EvidenceValidationError("test receipt command mismatch")
    exit_code = receipt.get("exit_code")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool) or exit_code != 0:
        raise EvidenceValidationError("test receipt exit code is not integer zero")
    if not is_sha256_hex(receipt.get("dependency_snapshot_sha256")):
        raise EvidenceValidationError("test receipt dependency snapshot is invalid")
    for key in ("logical_log_path", "runner_version", "working_tree_status",
                "command_environment_policy", "python_version"):
        if not isinstance(receipt.get(key), str) or not receipt[key]:
            raise EvidenceValidationError(f"test receipt {key} is invalid")
    started = datetime.fromisoformat(normalize_utc_timestamp(receipt.get("started_at_utc")))
    completed = datetime.fromisoformat(normalize_utc_timestamp(receipt.get("completed_at_utc")))
    if completed < started:
        raise EvidenceValidationError("test receipt completion precedes start")
    now = datetime.fromisoformat(normalize_utc_timestamp(now_utc)) if now_utc else datetime.now(timezone.utc)
    if started > now or completed > now:
        raise EvidenceValidationError("test receipt timestamp is in the future")
    if not Path(log_path).is_file() or Path(log_path).stat().st_size == 0:
        raise EvidenceValidationError("test receipt log is missing or empty")
    digest = sha256_file(Path(log_path))
    if receipt.get("log_sha256") != digest:
        raise EvidenceValidationError("test receipt log hash mismatch")
    return {"receipt_sha256": claimed, "log_sha256": digest,
            "command_argv": required_command, "verified": True}

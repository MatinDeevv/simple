from pathlib import Path
import pytest
from engine.experiments.artifact_binding import bind_physical_files
from engine.experiments.errors import EvidenceValidationError
from engine.experiments.canonical import sha256_payload
from engine.experiments.test_receipts import verify_test_receipt
import json, hashlib


def test_physical_file_is_hashed_and_assertions_checked(tmp_path: Path):
    path = tmp_path / "rows.jsonl"; path.write_bytes(b"{}\n")
    result = bind_physical_files([{"physical_path": str(path), "logical_path": "rows.jsonl",
                                   "expected_size_bytes": 3}], allowed_root=tmp_path)
    assert result[0]["size_bytes"] == 3 and len(result[0]["sha256"]) == 64
    with pytest.raises(EvidenceValidationError, match="hash mismatch"):
        bind_physical_files([{"physical_path": str(path), "logical_path": "rows.jsonl",
                              "expected_sha256": "0" * 64}], allowed_root=tmp_path)


def test_symlink_and_escape_fail(tmp_path: Path):
    outside = tmp_path.parent / "outside-artifact"; outside.write_text("x")
    with pytest.raises(EvidenceValidationError):
        bind_physical_files([{"physical_path": str(outside), "logical_path": "x"}],
                            allowed_root=tmp_path)


def test_physical_test_receipt_and_log_are_verified(tmp_path: Path):
    log = tmp_path / "pytest.log"; log.write_text("1 passed\n", encoding="utf-8")
    body = {"receipt_version": "test-receipt-v2",
            "command_argv": ["python", "-m", "pytest", "-q"], "exit_code": 0,
            "commit_sha": "a" * 40, "started_at_utc": "2026-01-01T00:00:00+00:00",
            "completed_at_utc": "2026-01-01T00:01:00+00:00",
            "python_version": "3.11", "dependency_snapshot_sha256": "b" * 64,
            "logical_log_path": "logs/pytest.log", "runner_version": "test-runner-v2",
            "working_tree_status": "clean", "command_environment_policy": "sealed-v1",
            "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()}
    receipt = dict(body, receipt_sha256=sha256_payload(body))
    path = tmp_path / "receipt.json"; path.write_text(json.dumps(receipt), encoding="utf-8")
    assert verify_test_receipt(path, log, expected_commit="a" * 40,
                               required_command=body["command_argv"],
                               now_utc="2026-01-02T00:00:00+00:00")["verified"]
    log.write_text("tampered", encoding="utf-8")
    with pytest.raises(EvidenceValidationError, match="log hash"):
        verify_test_receipt(path, log, expected_commit="a" * 40,
                            required_command=body["command_argv"],
                            now_utc="2026-01-02T00:00:00+00:00")

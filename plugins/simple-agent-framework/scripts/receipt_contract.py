"""Strict canonical contract shared by receipt producers, CLI verification and MCP."""
from __future__ import annotations
import hashlib, json, re
from datetime import datetime, timezone, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

VERSION = "simple-test-receipt-v2"
SHA = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
FIELDS = {
    "receipt_version", "command_argv", "exit_code", "started_at_utc", "completed_at_utc",
    "commit_sha", "python_version", "dependency_snapshot_sha256", "logical_log_path",
    "log_path", "log_sha256", "runner_version", "working_tree_status",
    "environment_policy", "receipt_sha256",
}

class ReceiptError(ValueError): pass

def canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()

def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result: raise ReceiptError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

def load_strict(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object,
                           parse_constant=lambda value: (_ for _ in ()).throw(ReceiptError(f"non-finite value: {value}")))
    except (json.JSONDecodeError, UnicodeError) as exc: raise ReceiptError(f"invalid receipt JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != FIELDS: raise ReceiptError("receipt fields do not exactly match v2 contract")
    return value

def verify(path: Path, *, expected_commit: str | None = None) -> dict[str, Any]:
    payload = load_strict(path)
    if payload["receipt_version"] != VERSION: raise ReceiptError("unsupported receipt version")
    claimed = payload["receipt_sha256"]; unsigned = {k:v for k,v in payload.items() if k != "receipt_sha256"}
    if not SHA.fullmatch(str(claimed)) or hashlib.sha256(canonical(unsigned)).hexdigest() != claimed: raise ReceiptError("receipt self-hash mismatch")
    if not COMMIT.fullmatch(str(payload["commit_sha"])): raise ReceiptError("invalid commit SHA")
    if expected_commit is not None and payload["commit_sha"] != expected_commit: raise ReceiptError("receipt commit mismatch")
    if type(payload["exit_code"]) is not int: raise ReceiptError("exit code must be an integer")
    argv=payload["command_argv"]
    if not isinstance(argv,list) or not argv or not all(isinstance(x,str) and x for x in argv): raise ReceiptError("command argv is not canonical")
    if not SHA.fullmatch(str(payload["dependency_snapshot_sha256"])) or not SHA.fullmatch(str(payload["log_sha256"])): raise ReceiptError("invalid SHA-256")
    start=datetime.fromisoformat(str(payload["started_at_utc"]).replace("Z","+00:00")); end=datetime.fromisoformat(str(payload["completed_at_utc"]).replace("Z","+00:00"))
    if start.utcoffset()!=timedelta(0) or end.utcoffset()!=timedelta(0) or end < start or end > datetime.now(timezone.utc)+timedelta(minutes=5): raise ReceiptError("invalid receipt timestamps")
    raw=str(payload["log_path"]); logical=PurePosixPath(raw.replace("\\","/"))
    if logical.is_absolute() or len(logical.parts)!=1 or ".." in logical.parts: raise ReceiptError("unsafe receipt log path")
    log=(path.parent/logical.as_posix()).resolve()
    if log.parent != path.parent.resolve() or not log.is_file() or log.is_symlink(): raise ReceiptError("receipt log is missing or unsafe")
    content=log.read_bytes()
    if not content: raise ReceiptError("receipt log must not be empty")
    if hashlib.sha256(content).hexdigest()!=payload["log_sha256"]: raise ReceiptError("receipt log hash mismatch")
    return payload

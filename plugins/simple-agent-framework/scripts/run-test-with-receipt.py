"""Execute argv and atomically emit a strict, self-verifying test receipt."""
from __future__ import annotations

import hashlib, json, os, subprocess, sys, tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from receipt_contract import VERSION, canonical
RUNNER_VERSION = "2"
ENV_ALLOWLIST = {"PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "USERPROFILE", "PYTHONHASHSEED", "LANG", "LC_ALL"}

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def atomic_write(path: Path, data: bytes) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".staging", dir=path.parent)
    staging = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)

def dependency_hash() -> str:
    rows = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in metadata.distributions())
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()

def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit("usage: run-test-with-receipt.py command [args ...]")
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    status = "dirty" if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip() else "clean"
    started = datetime.now(timezone.utc).isoformat()
    safe_env = {key: value for key, value in os.environ.items() if key.upper() in ENV_ALLOWLIST}
    completed = subprocess.run(argv, cwd=root, env=safe_env, capture_output=True, text=True, check=False)
    finished = datetime.now(timezone.utc).isoformat()
    receipts = root / ".agents" / "receipts"; receipts.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log = receipts / f"test-{stamp}.log"
    transcript=(completed.stdout + completed.stderr).encode("utf-8") or b"<no output>\n"
    atomic_write(log, transcript)
    payload = {
        "receipt_version": VERSION, "command_argv": argv, "exit_code": completed.returncode,
        "started_at_utc": started, "completed_at_utc": finished, "commit_sha": commit,
        "python_version": sys.version.split()[0], "dependency_snapshot_sha256": dependency_hash(),
        "logical_log_path": f".agents/receipts/{log.name}", "log_path": log.name,
        "log_sha256": digest(log), "runner_version": RUNNER_VERSION,
        "working_tree_status": status, "environment_policy": "minimal-test-v1",
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    atomic_write(receipts / f"test-{stamp}.json", json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
    return completed.returncode

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

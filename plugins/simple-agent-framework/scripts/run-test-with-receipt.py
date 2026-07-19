"""Execute an argv command and emit a self-verifying, secret-free receipt."""
from __future__ import annotations

import hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit("usage: run-test-with-receipt.py command [args ...]")
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    started = datetime.now(timezone.utc).isoformat()
    safe_env = {key: value for key, value in os.environ.items() if not any(token in key.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    completed = subprocess.run(argv, cwd=root, env=safe_env, capture_output=True, text=True)
    finished = datetime.now(timezone.utc).isoformat()
    receipts = root / ".agents" / "receipts"; receipts.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log = receipts / f"test-{stamp}.log"; log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    payload = {"command_argv": argv, "exit_code": completed.returncode, "started_at_utc": started, "completed_at_utc": finished, "commit_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), "python_version": sys.version.split()[0], "log_path": log.name, "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()}
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    (receipts / f"test-{stamp}.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""Research-run manifest: a signed-by-hash record of what produced an artifact.

A manifest never asserts an artifact is valid merely because a file exists on
disk; ``promotion_eligible`` is derived only from explicit, caller-supplied
facts (git cleanliness, recorded source hashes, a recorded test-pass flag,
and holdout status). Every manifest is self-verifying: ``manifest_sha256``
covers the canonical JSON serialization of every other field, so any later
tamper with a saved manifest file is detectable by recomputing the hash.

This module does not run tests or self-checks itself and does not decide
what "required tests" means for a given research module -- the caller
records that fact explicitly (e.g. by actually running pytest/self-checks
and passing the boolean result in). This is deliberate: a manifest must not
infer success from an artifact's existence.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunManifestError(RuntimeError):
    """Raised when manifest inputs are unusable (not when a run is merely non-promotable)."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hashed_paths(root: Path, paths: list[Path] | None) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths or []:
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file():
            raise RunManifestError(f"path recorded in a manifest must exist: {resolved}")
        try:
            relative = str(resolved.resolve().relative_to(root.resolve())).replace("\\", "/")
        except ValueError:
            relative = str(resolved.resolve()).replace("\\", "/")
        hashes[relative] = sha256_file(resolved)
    return hashes


def git_info(root: Path) -> tuple[str | None, str]:
    """Return ``(commit_sha_or_None, status)`` where status is clean/dirty/unavailable.

    Never raises: a synthetic CI self-check environment may have no git
    binary or no ``.git`` directory at all, and that must degrade to an
    explicit ``"unavailable"`` status rather than crash the caller.
    """
    try:
        commit_result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                       text=True, timeout=10)
        if commit_result.returncode != 0:
            return None, "unavailable"
        commit = commit_result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None, "unavailable"
    try:
        status_result = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                                       text=True, timeout=10)
        if status_result.returncode != 0:
            return commit, "unavailable"
    except (OSError, subprocess.SubprocessError):
        return commit, "unavailable"
    return commit, ("dirty" if status_result.stdout.strip() else "clean")


def _evaluate_promotion(*, git_status: str, holdout_status: str, source_file_sha256: dict[str, str],
                        required_tests_passed: bool) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    if git_status != "clean":
        blockers.append(f"git worktree is not clean (status={git_status!r})")
    if holdout_status == "burned_acknowledged":
        blockers.append("run used an explicit burned-holdout acknowledgement")
    if not source_file_sha256:
        blockers.append("no source file hashes were recorded")
    if not required_tests_passed:
        blockers.append("required tests were not recorded as passed")
    return (len(blockers) == 0), blockers


def build_run_manifest(
    *,
    root: Path,
    frozen_contract_version: str,
    required_tests_passed: bool,
    holdout_status: str = "not_used",
    source_files: list[Path] | None = None,
    input_artifacts: list[Path] | None = None,
    output_artifacts: list[Path] | None = None,
    configuration: dict[str, Any] | None = None,
    dependency_versions: dict[str, str] | None = None,
    random_seeds: dict[str, int] | None = None,
    run_id: str | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build a fully-populated, self-verifying run manifest as a plain dict.

    ``holdout_status`` must be one of ``"not_used"``, ``"clean_holdout"``, or
    ``"burned_acknowledged"``. Any value other than ``"not_used"`` or
    ``"clean_holdout"`` is treated conservatively: only an explicit,
    known-safe status permits promotion consideration.
    """
    if holdout_status not in {"not_used", "clean_holdout", "burned_acknowledged"}:
        raise RunManifestError(f"unsupported holdout_status: {holdout_status!r}")
    commit, git_status = git_info(root)
    source_hashes = _hashed_paths(root, source_files)
    input_hashes = _hashed_paths(root, input_artifacts)
    output_hashes = _hashed_paths(root, output_artifacts)
    configuration_sha256 = hashlib.sha256(canonical_json(configuration or {}).encode("utf-8")).hexdigest()
    promotion_eligible, blockers = _evaluate_promotion(
        git_status=git_status, holdout_status=holdout_status,
        source_file_sha256=source_hashes, required_tests_passed=required_tests_passed,
    )

    payload: dict[str, Any] = {
        "run_id": run_id or str(uuid.uuid4()),
        "created_at_utc": created_at_utc or datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_status": git_status,
        "dirty_worktree": (git_status == "dirty") if git_status != "unavailable" else None,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dependency_versions": dict(dependency_versions or {}),
        "configuration_sha256": configuration_sha256,
        "source_file_sha256": source_hashes,
        "input_artifact_sha256": input_hashes,
        "output_artifact_sha256": output_hashes,
        "random_seeds": dict(random_seeds or {}),
        "frozen_contract_version": frozen_contract_version,
        "holdout_status": holdout_status,
        "required_tests_passed": bool(required_tests_passed),
        "promotion_eligible": promotion_eligible,
        "promotion_blockers": blockers,
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return payload


def verify_manifest_integrity(payload: dict[str, Any]) -> bool:
    """Recompute the manifest hash over every field except manifest_sha256 itself."""
    if "manifest_sha256" not in payload:
        return False
    claimed = payload["manifest_sha256"]
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest() == claimed


def write_manifest(payload: dict[str, Any], path: Path) -> Path:
    if not verify_manifest_integrity(payload):
        raise RunManifestError("refusing to write a manifest whose hash does not match its own contents")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RunManifestError(f"manifest at {path} is not a JSON object")
    return payload

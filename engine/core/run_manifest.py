"""Versioned, self-verifying provenance manifests (not a promotion oracle)."""
from __future__ import annotations

import hashlib, json, platform, re, subprocess, sys, uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from engine.core.schema_validate import validate_instance

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
V2 = "fxsim-research-run-manifest-v2"

class RunManifestError(RuntimeError): pass

@dataclass(frozen=True)
class ArtifactBinding:
    physical_path: Path
    logical_path: str

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as h:
        for block in iter(lambda: h.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()

def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)

def _logical_path(path: str) -> str:
    value = PurePosixPath(path.replace("\\", "/"))
    if value.is_absolute() or ".." in value.parts or str(value) in {"", "."}:
        raise RunManifestError(f"logical artifact path must be relative and traversal-free: {path!r}")
    return value.as_posix()

def _hashed_paths(root: Path, paths: list[Path | ArtifactBinding] | None) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for item in paths or []:
        binding = item if isinstance(item, ArtifactBinding) else ArtifactBinding(Path(item), str(item))
        physical = binding.physical_path if binding.physical_path.is_absolute() else root / binding.physical_path
        if not physical.is_file(): raise RunManifestError(f"path recorded in a manifest must exist: {physical}")
        logical = _logical_path(binding.logical_path)
        if logical in hashes: raise RunManifestError(f"duplicate logical artifact path: {logical}")
        hashes[logical] = sha256_file(physical)
    return hashes

def git_info(root: Path) -> tuple[str | None, str]:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        if result.returncode: return None, "unavailable"
        commit = result.stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10)
        return commit, ("dirty" if status.stdout.strip() else "clean") if not status.returncode else "unavailable"
    except (OSError, subprocess.SubprocessError): return None, "unavailable"

def _utc(value: str) -> bool:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.tzinfo is not None and dt.utcoffset() == timezone.utc.utcoffset(dt)
    except ValueError: return False

def _evidence_ok(evidence: list[dict[str, Any]], commit: str | None) -> bool:
    return bool(commit and any(e.get("exit_code") == 0 and e.get("commit_sha") == commit for e in evidence))

def _promotion(*, status: str, holdout: str, sources: dict[str,str], evidence: list[dict[str,Any]], commit: str | None) -> tuple[bool,list[str]]:
    blockers=[]
    if status != "clean": blockers.append(f"git worktree is not clean (status={status!r})")
    if holdout == "burned_acknowledged": blockers.append("run used an explicit burned-holdout acknowledgement")
    if not sources: blockers.append("no source file hashes were recorded")
    if not _evidence_ok(evidence, commit): blockers.append("no successful test evidence bound to this commit")
    return not blockers, blockers

def build_run_manifest(*, root: Path, frozen_contract_version: str, required_tests_passed: bool = False,
                       test_evidence: list[dict[str, Any]] | None = None, holdout_status: str = "not_used",
                       source_files: list[Path | ArtifactBinding] | None = None, input_artifacts: list[Path | ArtifactBinding] | None = None,
                       output_artifacts: list[Path | ArtifactBinding] | None = None, configuration: dict[str, Any] | None = None,
                       dependency_versions: dict[str,str] | None = None, random_seeds: dict[str,int] | None = None,
                       run_id: str | None = None, created_at_utc: str | None = None) -> dict[str, Any]:
    if holdout_status not in {"not_used", "clean_holdout", "burned_acknowledged"}: raise RunManifestError(f"unsupported holdout_status: {holdout_status!r}")
    commit,status = git_info(root); evidence=list(test_evidence or [])
    sources=_hashed_paths(root, source_files); inputs=_hashed_paths(root,input_artifacts); outputs=_hashed_paths(root,output_artifacts)
    eligible, blockers = _promotion(status=status, holdout=holdout_status, sources=sources, evidence=evidence, commit=commit)
    payload: dict[str,Any] = {"manifest_schema_version":V2,"run_id":run_id or str(uuid.uuid4()),"created_at_utc":created_at_utc or datetime.now(timezone.utc).isoformat(),"git_commit":commit,"git_status":status,"dirty_worktree": status == "dirty" if status != "unavailable" else None,"python_version":sys.version.split()[0],"platform":platform.platform(),"dependency_versions":dict(dependency_versions or {}),"configuration_sha256":hashlib.sha256(canonical_json(configuration or {}).encode()).hexdigest(),"source_file_sha256":sources,"input_artifact_sha256":inputs,"output_artifact_sha256":outputs,"random_seeds":dict(random_seeds or {}),"frozen_contract_version":frozen_contract_version,"holdout_status":holdout_status,"required_tests_passed":bool(required_tests_passed),"test_evidence":evidence,"promotion_eligible":eligible,"promotion_blockers":blockers}
    payload["manifest_sha256"] = "0" * 64
    _validate_payload(payload, verify_hash=False)
    payload["manifest_sha256"] = hashlib.sha256(canonical_json({k:v for k,v in payload.items() if k != "manifest_sha256"}).encode()).hexdigest()
    return payload

def _schema() -> dict[str,Any]:
    return json.loads((Path(__file__).parents[1] / "config" / "schemas" / "research-run-manifest.schema.json").read_text(encoding="utf-8"))

def _validate_payload(payload: dict[str,Any], *, verify_hash: bool = True) -> None:
    if payload.get("manifest_schema_version") != V2: return  # legacy v1 is integrity-only and never promotable.
    errors=validate_instance(_schema(), payload)
    if errors: raise RunManifestError("manifest schema validation failed: " + "; ".join(errors))
    if not _SHA256.fullmatch(payload["configuration_sha256"]): raise RunManifestError("configuration_sha256 must be SHA-256")
    try: uuid.UUID(payload["run_id"])
    except (ValueError, AttributeError): raise RunManifestError("run_id must be UUID") from None
    if not _utc(payload["created_at_utc"]): raise RunManifestError("created_at_utc must be UTC timezone-aware")
    if payload["git_commit"] is not None and not _COMMIT.fullmatch(payload["git_commit"]): raise RunManifestError("git_commit must be 40 lowercase hex")
    if payload["dirty_worktree"] != (payload["git_status"] == "dirty" if payload["git_status"] != "unavailable" else None): raise RunManifestError("dirty_worktree inconsistent with git_status")
    for hashes in (payload["source_file_sha256"],payload["input_artifact_sha256"],payload["output_artifact_sha256"]):
        if not all(_SHA256.fullmatch(v) for v in hashes.values()): raise RunManifestError("artifact hashes must be SHA-256")
    for evidence in payload["test_evidence"]:
        required={"command","exit_code","commit_sha","started_at_utc","completed_at_utc","artifact_sha256","python_version","dependency_snapshot_sha256"}
        if set(evidence) != required or not isinstance(evidence["command"],str) or not isinstance(evidence["exit_code"],int): raise RunManifestError("test evidence has invalid fields")
        if not _COMMIT.fullmatch(evidence["commit_sha"]) or not _SHA256.fullmatch(evidence["artifact_sha256"]) or not _SHA256.fullmatch(evidence["dependency_snapshot_sha256"]) or not _utc(evidence["started_at_utc"]) or not _utc(evidence["completed_at_utc"]): raise RunManifestError("test evidence has invalid provenance")
    derived,blockers=_promotion(status=payload["git_status"],holdout=payload["holdout_status"],sources=payload["source_file_sha256"],evidence=payload["test_evidence"],commit=payload["git_commit"])
    if payload["promotion_eligible"] != derived or payload["promotion_blockers"] != blockers: raise RunManifestError("promotion fields are inconsistent with manifest evidence")
    if verify_hash and not verify_manifest_integrity(payload): raise RunManifestError("manifest integrity verification failed")

def verify_manifest_integrity(payload: dict[str,Any]) -> bool:
    claimed=payload.get("manifest_sha256")
    return isinstance(claimed,str) and hashlib.sha256(canonical_json({k:v for k,v in payload.items() if k != "manifest_sha256"}).encode()).hexdigest() == claimed

def write_manifest(payload: dict[str,Any], path: Path) -> Path:
    _validate_payload(payload); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8"); return path

def read_manifest(path: Path, *, verify: bool = True) -> dict[str,Any]:
    payload=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload,dict): raise RunManifestError(f"manifest at {path} is not a JSON object")
    if verify:
        if payload.get("manifest_schema_version") == V2: _validate_payload(payload)
        elif not verify_manifest_integrity(payload): raise RunManifestError("legacy manifest integrity verification failed")
    return payload

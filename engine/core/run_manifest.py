"""Versioned, self-verifying provenance manifests (not a promotion oracle)."""
from __future__ import annotations

import hashlib, json, os, platform, re, subprocess, sys, tempfile, uuid
from importlib import metadata
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from engine.core.schema_validate import validate_instance

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
V2 = "fxsim-research-run-manifest-v2"
V1 = "fxsim-research-run-manifest-v1"
CURRENT_WRITABLE_VERSION = V2
READABLE_LEGACY_VERSIONS = frozenset({V1})
REJECTED_FUTURE_VERSION_PREFIX = "fxsim-research-run-manifest-v"
_RECEIPT_FIELDS={"receipt_version","command_argv","exit_code","started_at_utc","completed_at_utc","commit_sha","python_version","dependency_snapshot_sha256","logical_log_path","log_path","log_sha256","runner_version","working_tree_status","environment_policy","receipt_sha256"}

class RunManifestError(RuntimeError): pass

@dataclass(frozen=True)
class ArtifactBinding:
    physical_path: Path
    logical_path: str

@dataclass(frozen=True)
class ReceiptBinding:
    receipt_path: Path
    logical_receipt_path: str

def _strict_json_file(path: Path) -> dict[str, Any]:
    value=_strict_load(path.read_text(encoding="utf-8"),source=path)
    if not isinstance(value,dict): raise RunManifestError("test receipt must be a JSON object")
    return value

def _physical_receipt(binding: ReceiptBinding, *, root: Path, commit: str | None, dependency_hash: str) -> dict[str,Any]:
    physical=binding.receipt_path if binding.receipt_path.is_absolute() else root/binding.receipt_path
    if physical.is_symlink() or not physical.is_file(): raise RunManifestError("test receipt must be a regular file")
    logical=_logical_path(binding.logical_receipt_path); payload=_strict_json_file(physical)
    if set(payload)!=_RECEIPT_FIELDS or payload.get("receipt_version")!="simple-test-receipt-v2": raise RunManifestError("test receipt does not match the strict v2 contract")
    claimed=payload.get("receipt_sha256"); unsigned={k:v for k,v in payload.items() if k!="receipt_sha256"}
    if not _SHA256.fullmatch(str(claimed)) or hashlib.sha256(canonical_json(unsigned).encode()).hexdigest()!=claimed: raise RunManifestError("test receipt self-hash mismatch")
    argv=payload.get("command_argv")
    if not isinstance(argv,list) or not argv or not all(isinstance(x,str) and x for x in argv): raise RunManifestError("test receipt command_argv is invalid")
    if type(payload.get("exit_code")) is not int: raise RunManifestError("test receipt exit_code must be an integer")
    try:
        started=datetime.fromisoformat(str(payload["started_at_utc"]).replace("Z","+00:00")); completed=datetime.fromisoformat(str(payload["completed_at_utc"]).replace("Z","+00:00"))
    except ValueError as exc: raise RunManifestError("test receipt timestamps are invalid") from exc
    if started.utcoffset()!=timedelta(0) or completed.utcoffset()!=timedelta(0) or completed<started or completed>datetime.now(timezone.utc)+timedelta(minutes=5): raise RunManifestError("test receipt timestamps are invalid")
    if not isinstance(payload.get("python_version"),str) or not payload["python_version"].strip(): raise RunManifestError("test receipt Python version is invalid")
    raw_log=str(payload.get("log_path","")); log_rel=PurePosixPath(raw_log.replace("\\","/"))
    if log_rel.is_absolute() or len(log_rel.parts)!=1 or ".." in log_rel.parts: raise RunManifestError("test receipt log path is unsafe")
    log=(physical.parent/log_rel.as_posix()).resolve()
    if log.parent!=physical.parent.resolve() or log.is_symlink() or not log.is_file() or not log.read_bytes(): raise RunManifestError("test receipt log is missing, empty, or unsafe")
    log_hash=sha256_file(log)
    if log_hash!=payload.get("log_sha256"): raise RunManifestError("test receipt log hash mismatch")
    if payload.get("commit_sha")!=commit or payload.get("dependency_snapshot_sha256")!=dependency_hash: raise RunManifestError("test receipt provenance mismatch")
    return {"command":canonical_json({"argv":argv}),"command_argv":argv,"exit_code":payload.get("exit_code"),"commit_sha":payload.get("commit_sha"),"started_at_utc":payload.get("started_at_utc"),"completed_at_utc":payload.get("completed_at_utc"),"artifact_sha256":claimed,"receipt_sha256":claimed,"receipt_logical_path":logical,"log_logical_path":_logical_path(str(PurePosixPath(logical).parent/log_rel)),"log_sha256":log_hash,"python_version":payload.get("python_version"),"dependency_snapshot_sha256":payload.get("dependency_snapshot_sha256"),"physical_files_verified":True}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as h:
        for block in iter(lambda: h.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()

def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)

def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RunManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

def _strict_load(text: str, *, source: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise RunManifestError(f"non-finite JSON number in {source}: {value}")
    try:
        return json.loads(text, object_pairs_hook=_strict_object, parse_constant=reject_constant)
    except RunManifestError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise RunManifestError(f"invalid manifest JSON at {source}: {exc}") from exc

def _logical_path(path: str) -> str:
    if re.match(r"^[A-Za-z]:[\\/]", path) or path.startswith(("\\\\", "//")):
        raise RunManifestError(f"logical artifact path must not be drive-qualified: {path!r}")
    value = PurePosixPath(path.replace("\\", "/"))
    if value.is_absolute() or ".." in value.parts or str(value) in {"", "."}:
        raise RunManifestError(f"logical artifact path must be relative and traversal-free: {path!r}")
    return value.as_posix()

def _hashed_paths(root: Path, paths: list[Path | ArtifactBinding] | None) -> dict[str, str]:
    hashes: dict[str, str] = {}
    root = root.resolve()
    for item in paths or []:
        if isinstance(item, ArtifactBinding):
            binding = item
            physical = binding.physical_path if binding.physical_path.is_absolute() else root / binding.physical_path
            logical = _logical_path(binding.logical_path)
        else:
            physical = Path(item) if Path(item).is_absolute() else root / Path(item)
            physical = physical.resolve()
            try:
                logical = physical.relative_to(root).as_posix()
            except ValueError:
                raise RunManifestError("external physical paths require an explicit ArtifactBinding") from None
        if physical.is_symlink(): raise RunManifestError(f"symlink artifacts are not permitted: {physical}")
        physical = physical.resolve()
        if not physical.is_file(): raise RunManifestError(f"path recorded in a manifest must exist: {physical}")
        logical = _logical_path(logical)
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

def _evidence_ok(evidence: list[dict[str, Any]], commit: str | None, required_commands: list[list[str]]) -> bool:
    if not commit or not evidence:
        return False
    seen: set[tuple[str,...]] = set()
    for item in evidence:
        if (not isinstance(item.get("command"), str) or not item["command"].strip()
                or type(item.get("exit_code")) is not int or item["exit_code"] != 0
                or item.get("commit_sha") != commit or item.get("physical_files_verified") is not True
                or not isinstance(item.get("command_argv"),list)):
            return False
        seen.add(tuple(item["command_argv"]))
    return {tuple(command) for command in required_commands}.issubset(seen)

def _promotion(*, status: str, holdout: str, sources: dict[str,str], evidence: list[dict[str,Any]], commit: str | None, required_tests_passed: bool, required_commands: list[list[str]]) -> tuple[bool,list[str]]:
    blockers=[]
    if status != "clean": blockers.append(f"git worktree is not clean (status={status!r})")
    if holdout != "clean_holdout": blockers.append("promotion requires holdout_status='clean_holdout'")
    if not sources: blockers.append("no source file hashes were recorded")
    if required_tests_passed is not True: blockers.append("required_tests_passed is not true")
    if not required_commands: blockers.append("promotion requires at least one required test command")
    if not _evidence_ok(evidence, commit, required_commands): blockers.append("test evidence is incomplete or not bound to this commit")
    return not blockers, blockers

def build_run_manifest(*, root: Path, frozen_contract_version: str, required_tests_passed: bool = False,
                       test_evidence: list[dict[str, Any]] | None = None, holdout_status: str = "not_used",
                       source_files: list[Path | ArtifactBinding] | None = None, input_artifacts: list[Path | ArtifactBinding] | None = None,
                       output_artifacts: list[Path | ArtifactBinding] | None = None, configuration: dict[str, Any] | None = None,
                       dependency_versions: dict[str,str] | None = None, random_seeds: dict[str,int] | None = None,
                       run_id: str | None = None, created_at_utc: str | None = None,
                       required_test_commands: list[list[str]] | None = None,
                       test_receipts: list[ReceiptBinding] | None = None) -> dict[str, Any]:
    if holdout_status not in {"not_used", "clean_holdout", "burned_acknowledged"}: raise RunManifestError(f"unsupported holdout_status: {holdout_status!r}")
    commit,status = git_info(root); evidence=list(test_evidence or []); required_commands=list(required_test_commands or [])
    normalized=[]
    for command in required_commands:
        if not isinstance(command,list) or not command or not all(isinstance(x,str) and x for x in command): raise RunManifestError("required_test_commands must contain non-empty argv arrays")
        normalized.append(tuple(command))
    if len(set(normalized))!=len(normalized): raise RunManifestError("required_test_commands must be unique")
    sources=_hashed_paths(root, source_files); inputs=_hashed_paths(root,input_artifacts); outputs=_hashed_paths(root,output_artifacts)
    deps=dict(dependency_versions or _default_dependencies())
    dep_hash=hashlib.sha256(canonical_json(deps).encode()).hexdigest()
    if test_receipts:
        evidence=[_physical_receipt(item,root=root,commit=commit,dependency_hash=dep_hash) for item in test_receipts]
    eligible, blockers = _promotion(status=status, holdout=holdout_status, sources=sources, evidence=evidence, commit=commit, required_tests_passed=required_tests_passed, required_commands=required_commands)
    payload: dict[str,Any] = {"manifest_schema_version":V2,"run_id":run_id or str(uuid.uuid4()),"created_at_utc":created_at_utc or datetime.now(timezone.utc).isoformat(),"git_commit":commit,"git_status":status,"dirty_worktree": status == "dirty" if status != "unavailable" else None,"python_version":sys.version.split()[0],"platform":platform.platform(),"dependency_versions":deps,"dependency_snapshot_sha256":dep_hash,"configuration_sha256":hashlib.sha256(canonical_json(configuration or {}).encode()).hexdigest(),"source_file_sha256":sources,"input_artifact_sha256":inputs,"output_artifact_sha256":outputs,"random_seeds":dict(random_seeds or {}),"frozen_contract_version":frozen_contract_version,"holdout_status":holdout_status,"required_tests_passed":required_tests_passed is True,"required_test_commands":required_commands,"test_evidence":evidence,"promotion_eligible":eligible,"promotion_blockers":blockers}
    payload["manifest_sha256"] = "0" * 64
    _validate_payload(payload, verify_hash=False)
    payload["manifest_sha256"] = hashlib.sha256(canonical_json({k:v for k,v in payload.items() if k != "manifest_sha256"}).encode()).hexdigest()
    return payload

def _schema() -> dict[str,Any]:
    return json.loads((Path(__file__).parents[1] / "config" / "schemas" / "research-run-manifest.schema.json").read_text(encoding="utf-8"))

def _default_dependencies() -> dict[str, str]:
    def version(name: str) -> str:
        try: return metadata.version(name)
        except metadata.PackageNotFoundError: return "unavailable"
    return {"engine": version("engine"), "numpy": version("numpy"), "pandas": version("pandas"), "pyarrow": version("pyarrow")}

def _validate_payload(payload: dict[str,Any], *, verify_hash: bool = True) -> None:
    if payload.get("manifest_schema_version") != V2:
        raise RunManifestError("only the current V2 manifest schema is writable/validatable")
    errors=validate_instance(_schema(), payload)
    if errors: raise RunManifestError("manifest schema validation failed: " + "; ".join(errors))
    if not _SHA256.fullmatch(payload["configuration_sha256"]): raise RunManifestError("configuration_sha256 must be SHA-256")
    if not _SHA256.fullmatch(payload["dependency_snapshot_sha256"]): raise RunManifestError("dependency_snapshot_sha256 must be SHA-256")
    if hashlib.sha256(canonical_json(payload["dependency_versions"]).encode()).hexdigest() != payload["dependency_snapshot_sha256"]: raise RunManifestError("dependency snapshot hash is inconsistent")
    try: uuid.UUID(payload["run_id"])
    except (ValueError, AttributeError): raise RunManifestError("run_id must be UUID") from None
    if not _utc(payload["created_at_utc"]): raise RunManifestError("created_at_utc must be UTC timezone-aware")
    if payload["git_commit"] is not None and not _COMMIT.fullmatch(payload["git_commit"]): raise RunManifestError("git_commit must be 40 lowercase hex")
    if payload["dirty_worktree"] != (payload["git_status"] == "dirty" if payload["git_status"] != "unavailable" else None): raise RunManifestError("dirty_worktree inconsistent with git_status")
    for hashes in (payload["source_file_sha256"],payload["input_artifact_sha256"],payload["output_artifact_sha256"]):
        if not all(_SHA256.fullmatch(v) for v in hashes.values()): raise RunManifestError("artifact hashes must be SHA-256")
    for evidence in payload["test_evidence"]:
        required={"command","exit_code","commit_sha","started_at_utc","completed_at_utc","artifact_sha256","python_version","dependency_snapshot_sha256"}
        physical={"command_argv","receipt_sha256","receipt_logical_path","log_logical_path","log_sha256","physical_files_verified"}
        keys=frozenset(evidence)
        if keys not in {frozenset(required),frozenset(required|physical)} or not isinstance(evidence["command"],str) or not evidence["command"].strip() or type(evidence["exit_code"]) is not int: raise RunManifestError("test evidence has invalid fields")
        if physical <= set(evidence):
            if evidence["physical_files_verified"] is not True or not isinstance(evidence["command_argv"],list) or not evidence["command_argv"] or not all(isinstance(x,str) and x for x in evidence["command_argv"]): raise RunManifestError("physical test evidence is invalid")
            for key in ("receipt_sha256","log_sha256"):
                if not _SHA256.fullmatch(evidence[key]): raise RunManifestError("physical evidence hashes must be SHA-256")
            for key in ("receipt_logical_path","log_logical_path"): _logical_path(evidence[key])
        now = datetime.now(timezone.utc)
        if (not evidence["python_version"].strip() or not _COMMIT.fullmatch(evidence["commit_sha"]) or not _SHA256.fullmatch(evidence["artifact_sha256"]) or not _SHA256.fullmatch(evidence["dependency_snapshot_sha256"]) or not _utc(evidence["started_at_utc"]) or not _utc(evidence["completed_at_utc"]) or datetime.fromisoformat(evidence["completed_at_utc"].replace("Z","+00:00")) < datetime.fromisoformat(evidence["started_at_utc"].replace("Z","+00:00")) or datetime.fromisoformat(evidence["started_at_utc"].replace("Z","+00:00")) > now): raise RunManifestError("test evidence has invalid provenance")
        if evidence["dependency_snapshot_sha256"] != payload["dependency_snapshot_sha256"]: raise RunManifestError("test evidence dependency snapshot does not match manifest")
    derived,blockers=_promotion(status=payload["git_status"],holdout=payload["holdout_status"],sources=payload["source_file_sha256"],evidence=payload["test_evidence"],commit=payload["git_commit"],required_tests_passed=payload["required_tests_passed"],required_commands=payload["required_test_commands"])
    if payload["promotion_eligible"] != derived or payload["promotion_blockers"] != blockers: raise RunManifestError("promotion fields are inconsistent with manifest evidence")
    if verify_hash and not verify_manifest_integrity(payload): raise RunManifestError("manifest integrity verification failed")

def verify_manifest_integrity(payload: dict[str,Any]) -> bool:
    claimed=payload.get("manifest_sha256")
    return isinstance(claimed,str) and hashlib.sha256(canonical_json({k:v for k,v in payload.items() if k != "manifest_sha256"}).encode()).hexdigest() == claimed

def write_manifest(payload: dict[str,Any], path: Path, *, replace: bool = False) -> Path:
    """Durably stage, verify, and atomically publish a current manifest."""
    _validate_payload(payload)
    if payload.get("manifest_schema_version") != CURRENT_WRITABLE_VERSION:
        raise RunManifestError("legacy manifests are read-only")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise RunManifestError(f"refusing to overwrite existing manifest: {path}")
    staging: Path | None = None
    try:
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".staging", dir=path.parent)
        staging = Path(name)
        with os.fdopen(fd, "wb") as handle:
            handle.write((json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
            handle.flush(); os.fsync(handle.fileno())
        if read_manifest(staging) != payload:
            raise RunManifestError("manifest staging readback mismatch")
        if replace:
            os.replace(staging, path); staging = None
        else:
            try:
                os.link(staging, path)
            except FileExistsError:
                raise RunManifestError(f"refusing to overwrite existing manifest: {path}") from None
            staging.unlink(); staging = None
        if hasattr(os, "O_DIRECTORY"):
            fd_dir = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                try: os.fsync(fd_dir)
                except OSError: pass  # Publication already succeeded; durability is best-effort.
            finally: os.close(fd_dir)
        return path
    except BaseException:
        if staging is not None:
            try: staging.unlink(missing_ok=True)
            except OSError: pass
        raise

def read_manifest(path: Path, *, verify: bool = True) -> dict[str,Any]:
    payload=_strict_load(path.read_text(encoding="utf-8"), source=path)
    if not isinstance(payload,dict): raise RunManifestError(f"manifest at {path} is not a JSON object")
    version = payload.get("manifest_schema_version")
    if version != CURRENT_WRITABLE_VERSION:
        if version not in READABLE_LEGACY_VERSIONS:
            raise RunManifestError(f"unsupported manifest schema version: {version!r}")
        if verify and not verify_manifest_integrity(payload): raise RunManifestError("legacy manifest integrity verification failed")
    elif verify: _validate_payload(payload)
    return payload

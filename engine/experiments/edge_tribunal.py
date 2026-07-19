"""Edge Tribunal orchestrator and command-line interface.

An experiment lives in one directory of immutable versioned snapshots:

    experiment_dir/
      CURRENT                  # name of the authoritative version directory
      versions/v000001/        # complete snapshot: artifacts + state + audit
      versions/v000002/
      ...

Every state transition is transactional: the next snapshot is built in a
sibling staging directory, fully validated (including its audit chain and
schema contracts), atomically renamed into place, and only then does the
CURRENT pointer move (atomic replace). A failure at any point leaves the
prior snapshot authoritative; prior versions are never modified, so no
artifact is ever silently replaced.

CLI (the future ``auractl experiments`` integration point; the shared CLI in
``engine/cli`` is owned elsewhere and deliberately not touched):

    python -m engine.experiments.edge_tribunal init|seal|bind-data|
        record-evidence|evaluate|verify|report|show-state|archive
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from engine.experiments import (
    audit_log,
    dataset_binding,
    evidence as evidence_module,
    gates,
    holdout_registry,
    multiplicity,
    preregistration,
    reporting,
    robustness,
    state_machine,
    verdict as verdict_module,
)
from engine.experiments.canonical import (
    load_strict_json_text,
    normalize_logical_path,
    normalize_utc_timestamp,
    normalize_uuid,
    sha256_bytes,
    sha256_payload,
    strict_json_text,
    validate_against_schema,
)
from engine.experiments.errors import (
    AuditIntegrityError,
    CanonicalJsonError,
    EvidenceValidationError,
    InvalidStateTransitionError,
    LockError,
    PathSecurityError,
    PlanMutationError,
    TransactionError,
    TribunalError,
    VerdictIntegrityError,
)

CURRENT_POINTER = "CURRENT"
VERSIONS_DIR = "versions"
LOCK_FILENAME = "experiment.lock"

PLAN_FILE = "plan.json"
SEAL_FILE = "seal.json"
BINDING_FILE = "dataset-binding.json"
EVIDENCE_FILE = "evidence.json"
VERDICT_FILE = "verdict.json"
SCORECARD_FILE = "scorecard.json"
VERIFICATION_FILE = "verification.json"
REGISTRY_RECONCILIATION_FILE = "registry-reconciliation.json"
SNAPSHOT_META_FILE = "snapshot-meta.json"
REPORT_FILE = "report.md"
STATE_FILE = "state.json"
AUDIT_FILE = "audit.jsonl"

SEAL_VERSION = "edge-tribunal-seal-v1"

_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "config" / "schemas"
_SCHEMA_FOR_FILE = {
    PLAN_FILE: "edge-tribunal-plan.schema.json",
    SEAL_FILE: "edge-tribunal-seal.schema.json",
    BINDING_FILE: "edge-tribunal-dataset-binding.schema.json",
    EVIDENCE_FILE: "edge-tribunal-evidence.schema.json",
    VERDICT_FILE: "edge-tribunal-verdict.schema.json",
}
_ALLOWED_SNAPSHOT_FILES = frozenset({
    PLAN_FILE, SEAL_FILE, BINDING_FILE, EVIDENCE_FILE, VERDICT_FILE,
    SCORECARD_FILE, VERIFICATION_FILE, REPORT_FILE, STATE_FILE, AUDIT_FILE,
    REGISTRY_RECONCILIATION_FILE,
    SNAPSHOT_META_FILE,
})


def _test_failpoint(stage: str) -> None:
    """No-op hook; tests monkeypatch this to inject BaseException interruptions
    at exact transaction stages (pre-write, after-artifact-write,
    before-manifest-write, before-rename, after-rename)."""


def _deterministic_id(kind: str, experiment_id: str, discriminator: str = "") -> str:
    """Stable UUIDv5 so identical inputs (including caller-fixed timestamps)
    produce byte-identical artifacts and reports."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL,
                          f"edge-tribunal:{kind}:{experiment_id}:{discriminator}"))


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_file_schema(filename: str, payload: Any) -> None:
    schema_name = _SCHEMA_FOR_FILE.get(filename)
    if schema_name is None:
        return
    schema_path = _SCHEMA_DIR / schema_name
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = validate_against_schema(schema, payload)
    if errors:
        raise TransactionError(
            f"{filename} violates {schema_name}:\n- " + "\n- ".join(errors))


class _ExperimentLock(holdout_registry.RegistryLock):
    def __init__(self, experiment_dir: Path, *, timeout_seconds: float = 5.0,
                 stale_after_seconds: float = 600.0) -> None:
        super().__init__(experiment_dir, timeout_seconds=timeout_seconds,
                         stale_after_seconds=stale_after_seconds)
        self.path = Path(experiment_dir) / LOCK_FILENAME


def current_version_name(experiment_dir: Path) -> str | None:
    pointer = Path(experiment_dir) / CURRENT_POINTER
    if not pointer.is_file():
        return None
    name = pointer.read_text(encoding="utf-8").strip()
    if not name.startswith("v") or "/" in name or "\\" in name or ".." in name:
        raise TransactionError(f"corrupt CURRENT pointer: {name!r}")
    return name


def _current_dir(experiment_dir: Path) -> Path:
    name = current_version_name(experiment_dir)
    if name is None:
        raise TribunalError(f"no experiment is initialized at {experiment_dir}")
    version_dir = Path(experiment_dir) / VERSIONS_DIR / name
    if not version_dir.is_dir():
        raise TransactionError(f"CURRENT points to a missing version: {name}")
    return version_dir


def _read_snapshot(version_dir: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for entry in sorted(version_dir.iterdir()):
        if entry.is_symlink():
            raise PathSecurityError(f"snapshot contains a symlink: {entry.name}")
        if entry.name not in _ALLOWED_SNAPSHOT_FILES:
            raise PathSecurityError(f"snapshot contains an unexpected artifact: {entry.name}")
        if entry.is_file():
            files[entry.name] = entry.read_bytes()
        else:
            raise PathSecurityError(f"snapshot contains a non-file entry: {entry.name}")
    return files


def load_artifact(experiment_dir: Path, filename: str) -> dict[str, Any]:
    version_dir = _current_dir(experiment_dir)
    path = version_dir / filename
    if not path.is_file():
        raise TribunalError(f"artifact {filename} does not exist yet "
                            f"(experiment state: {current_state(experiment_dir)})")
    payload = load_strict_json_text(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TribunalError(f"artifact {filename} is not a JSON object")
    return payload


def load_events(experiment_dir: Path) -> list[dict[str, Any]]:
    return audit_log.read_events(_current_dir(experiment_dir) / AUDIT_FILE)


def current_state(experiment_dir: Path) -> str:
    state_payload = load_artifact(experiment_dir, STATE_FILE)
    state = state_payload.get("state")
    if state not in state_machine.STATES:
        raise TransactionError(f"state file records unknown state {state!r}")
    return state


def _next_version_name(experiment_dir: Path) -> str:
    versions_root = Path(experiment_dir) / VERSIONS_DIR
    highest = 0
    if versions_root.is_dir():
        for entry in versions_root.iterdir():
            if entry.is_dir() and entry.name.startswith("v") and entry.name[1:].isdigit():
                highest = max(highest, int(entry.name[1:]))
    return f"v{highest + 1:06d}"


def _snapshot_sha256(files: dict[str, bytes]) -> str:
    return sha256_payload({name: sha256_bytes(content)
                           for name, content in sorted(files.items())})


def _publish_version(experiment_dir: Path, files: dict[str, bytes]) -> str:
    """Write a fully validated snapshot and atomically make it authoritative."""
    experiment_dir = Path(experiment_dir)
    versions_root = experiment_dir / VERSIONS_DIR
    versions_root.mkdir(parents=True, exist_ok=True)

    version_name = _next_version_name(experiment_dir)
    previous_version = current_version_name(experiment_dir)
    previous_hash = None
    if previous_version is not None:
        previous_hash = _snapshot_sha256(_read_snapshot(
            experiment_dir / VERSIONS_DIR / previous_version))
    payload_hash = _snapshot_sha256({name: content for name, content in files.items()
                                     if name != SNAPSHOT_META_FILE})
    meta = {"snapshot_meta_version": "edge-tribunal-snapshot-meta-v1",
            "version": version_name, "previous_version": previous_version,
            "previous_snapshot_sha256": previous_hash,
            "snapshot_payload_sha256": payload_hash}
    meta["snapshot_meta_sha256"] = sha256_payload(meta)
    files = dict(files, **{SNAPSHOT_META_FILE: strict_json_text(meta).encode("utf-8")})
    # Validate the snapshot before anything touches disk in its final home.
    events = [load_strict_json_text(line)
              for line in files[AUDIT_FILE].decode("utf-8").splitlines() if line.strip()]
    audit_log.verify_chain(events)
    state_payload = load_strict_json_text(files[STATE_FILE].decode("utf-8"))
    if state_payload.get("state") != events[-1]["state_after"]:
        raise TransactionError("state file disagrees with the audit chain")
    for filename, content in files.items():
        if filename.endswith(".json"):
            _validate_file_schema(filename, load_strict_json_text(content.decode("utf-8")))

    final_dir = versions_root / version_name
    staging_dir = versions_root / f".staging-{version_name}-{uuid.uuid4().hex}"
    renamed = False
    published = False
    try:
        _test_failpoint("pre-write")
        staging_dir.mkdir(parents=False, exist_ok=False)
        for filename, content in sorted(files.items()):
            if filename == AUDIT_FILE or filename == STATE_FILE:
                continue
            (staging_dir / filename).write_bytes(content)
        _test_failpoint("after-artifact-write")
        (staging_dir / AUDIT_FILE).write_bytes(files[AUDIT_FILE])
        _test_failpoint("before-manifest-write")
        (staging_dir / STATE_FILE).write_bytes(files[STATE_FILE])
        # Re-read and re-verify the staged bytes: what publishes is what was
        # validated, not what we hoped we wrote.
        staged = _read_snapshot(staging_dir)
        if staged != files:
            raise TransactionError("staged snapshot does not match validated content")
        _test_failpoint("before-rename")
        if final_dir.exists():
            raise TransactionError(f"version {version_name} already exists; refusing to overwrite")
        os.rename(staging_dir, final_dir)
        renamed = True
        _test_failpoint("after-rename")
        pointer_staging = experiment_dir / f".{CURRENT_POINTER}.staging-{uuid.uuid4().hex}"
        pointer_staging.write_text(version_name + "\n", encoding="utf-8")
        os.replace(pointer_staging, experiment_dir / CURRENT_POINTER)
        published = True
        return version_name
    except BaseException:
        # Controlled cleanup, then always re-raise (KeyboardInterrupt and
        # SystemExit included — they are re-raised, never swallowed).
        if not renamed:
            shutil.rmtree(staging_dir, ignore_errors=True)
        elif not published:
            # The snapshot landed but never became authoritative; remove it so
            # no half-published state can be mistaken for a real version.
            shutil.rmtree(final_dir, ignore_errors=True)
        raise


def _transition_files(
    experiment_dir: Path,
    *,
    operation: str,
    new_artifacts: dict[str, bytes],
    informational_events: list[tuple[str, Any]],
    transition_event_type: str,
    transition_payload: Any,
    actor: str,
    timestamp_utc: str,
) -> dict[str, bytes]:
    """Assemble the next snapshot: prior files + new artifacts + audit events."""
    state_before, state_after = state_machine.TRANSITION_FOR_OPERATION[operation]
    current = current_state(experiment_dir)
    state_machine.require_state(current, state_before, operation=operation)
    state_machine.validate_transition(state_before, state_after)

    version_dir = _current_dir(experiment_dir)
    files = _read_snapshot(version_dir)
    events = audit_log.read_events(version_dir / AUDIT_FILE)
    audit_log.verify_chain(events)
    if events[-1]["state_after"] != state_before:
        raise InvalidStateTransitionError(
            f"audit chain is in state {events[-1]['state_after']}, operation {operation!r} "
            f"requires {state_before}")

    for filename in new_artifacts:
        if filename in files:
            raise TransactionError(
                f"artifact {filename} already exists; artifacts are immutable and are never "
                f"overwritten")
    files.update(new_artifacts)

    experiment_id = events[0]["experiment_id"]
    previous_hash = audit_log.chain_head_hash(events)
    for event_type, payload in informational_events:
        event = audit_log.build_event(
            event_id=_deterministic_id("event", experiment_id,
                                       f"{len(events)}:{event_type}"),
            experiment_id=experiment_id,
            event_type=event_type, timestamp_utc=timestamp_utc, actor=actor,
            state_before=state_before, state_after=state_before,
            payload=payload, previous_event_sha256=previous_hash)
        events.append(event)
        previous_hash = event["event_sha256"]
    transition = audit_log.build_event(
        event_id=_deterministic_id("event", experiment_id,
                                   f"{len(events)}:{transition_event_type}"),
        experiment_id=experiment_id,
        event_type=transition_event_type, timestamp_utc=timestamp_utc, actor=actor,
        state_before=state_before, state_after=state_after,
        payload=transition_payload, previous_event_sha256=previous_hash)
    events.append(transition)

    files[AUDIT_FILE] = audit_log.serialize_events(events).encode("utf-8")
    files[STATE_FILE] = strict_json_text({
        "experiment_id": events[0]["experiment_id"],
        "state": state_after,
        "updated_at_utc": normalize_utc_timestamp(timestamp_utc),
    }).encode("utf-8")
    return files


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def init_experiment(
    experiment_dir: Path,
    plan: dict[str, Any],
    *,
    actor: str = "researcher",
    timestamp_utc: str | None = None,
) -> str:
    """Register a DRAFT experiment from a validated plan. Never overwrites."""
    experiment_dir = Path(experiment_dir)
    timestamp_utc = timestamp_utc or _now_utc()
    preregistration.validate_plan(plan)
    experiment_id = normalize_uuid(plan["identity"]["experiment_id"])
    event_type = "PLAN_AMENDED" if preregistration.is_amendment(plan) else "PLAN_CREATED"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    with _ExperimentLock(experiment_dir):
        if current_version_name(experiment_dir) is not None:
            raise TribunalError(f"an experiment already exists at {experiment_dir}; a sealed "
                                f"or draft experiment is never overwritten — amend into a new "
                                f"experiment directory instead")
        event = audit_log.build_event(
            event_id=_deterministic_id("event", experiment_id, f"0:{event_type}"),
            experiment_id=experiment_id,
            event_type=event_type, timestamp_utc=timestamp_utc, actor=actor,
            state_before=state_machine.DRAFT, state_after=state_machine.DRAFT,
            payload={"plan_sha256": preregistration.plan_sha256(plan)},
            previous_event_sha256=audit_log.GENESIS_HASH)
        files = {
            PLAN_FILE: strict_json_text(plan).encode("utf-8"),
            AUDIT_FILE: audit_log.serialize_events([event]).encode("utf-8"),
            STATE_FILE: strict_json_text({
                "experiment_id": experiment_id,
                "state": state_machine.DRAFT,
                "updated_at_utc": normalize_utc_timestamp(timestamp_utc),
            }).encode("utf-8"),
        }
        return _publish_version(experiment_dir, files)


def seal_experiment(
    experiment_dir: Path,
    *,
    actor: str = "researcher",
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Seal the plan: bind its hash, code identity, and ceiling irreversibly."""
    experiment_dir = Path(experiment_dir)
    timestamp_utc = timestamp_utc or _now_utc()
    with _ExperimentLock(experiment_dir):
        plan = load_artifact(experiment_dir, PLAN_FILE)
        preregistration.validate_plan(plan)
        events = load_events(experiment_dir)
        seal: dict[str, Any] = {
            "seal_version": SEAL_VERSION,
            "experiment_id": normalize_uuid(plan["identity"]["experiment_id"]),
            "plan_sha256": preregistration.plan_sha256(plan),
            "sealed_at_utc": normalize_utc_timestamp(timestamp_utc),
            "code_commit_sha": plan["code_contract"]["commit_sha"],
            "configuration_sha256": plan["code_contract"]["configuration_sha256"],
            "source_file_sha256": plan["code_contract"]["source_file_sha256"],
            "researcher": plan["identity"]["researcher"],
            "parent_experiment_id": plan["identity"].get("parent_experiment_id"),
            "amendment": preregistration.is_amendment(plan),
            "maximum_permitted_verdict": plan["promotion_ceiling"]["maximum_verdict"],
            "state_before": state_machine.DRAFT,
            "state_after": state_machine.SEALED,
            "previous_audit_event_sha256": audit_log.chain_head_hash(events),
        }
        seal["seal_sha256"] = sha256_payload(seal)
        files = _transition_files(
            experiment_dir, operation="seal",
            new_artifacts={SEAL_FILE: strict_json_text(seal).encode("utf-8")},
            informational_events=[],
            transition_event_type="PLAN_SEALED",
            transition_payload={"seal_sha256": seal["seal_sha256"],
                                "plan_sha256": seal["plan_sha256"]},
            actor=actor, timestamp_utc=timestamp_utc)
        _publish_version(experiment_dir, files)
        return seal


def _verify_plan_unmutated(plan: dict[str, Any], seal: dict[str, Any]) -> None:
    if preregistration.plan_sha256(plan) != seal["plan_sha256"]:
        raise PlanMutationError(
            "plan bytes no longer hash to the sealed plan_sha256: a sealed plan is immutable; "
            "register an amendment as a new experiment instead")


def _verify_seal_integrity(seal: dict[str, Any]) -> None:
    body = {key: value for key, value in seal.items() if key != "seal_sha256"}
    if sha256_payload(body) != seal.get("seal_sha256"):
        raise PlanMutationError("seal artifact is hash-inconsistent: it was modified after "
                                "sealing")


def bind_data(
    experiment_dir: Path,
    dataset_manifest: dict[str, Any],
    *,
    registry_root: Path,
    actor: str = "researcher",
    timestamp_utc: str | None = None,
    declare_forensic_reuse: bool = False,
    physical_file_bindings: list[dict[str, Any]] | None = None,
    dataset_root: Path | None = None,
) -> dict[str, Any]:
    """Bind the exact dataset after sealing, claiming its holdout interval."""
    experiment_dir = Path(experiment_dir)
    timestamp_utc = timestamp_utc or _now_utc()
    if physical_file_bindings is not None:
        if dataset_root is None:
            raise dataset_binding.DatasetBindingError(
                "dataset_root is required with physical_file_bindings")
        from engine.experiments.artifact_binding import bind_physical_files
        verified = bind_physical_files(
            physical_file_bindings, allowed_root=dataset_root,
            error_type=dataset_binding.DatasetBindingError)
        by_path = {item["logical_path"]: item for item in verified}
        dataset_manifest = dict(dataset_manifest)
        files: list[dict[str, Any]] = []
        for item in dataset_manifest.get("files", []):
            logical = normalize_logical_path(item["logical_path"])
            if logical not in by_path:
                raise dataset_binding.DatasetBindingError(
                    f"no physical binding supplied for {logical}")
            computed = by_path[logical]
            if item.get("sha256") != computed["sha256"]:
                raise dataset_binding.DatasetBindingError(f"hash mismatch for {logical}")
            files.append(dict(item, size_bytes=computed["size_bytes"]))
        if set(by_path) != {normalize_logical_path(item["logical_path"]) for item in files}:
            raise dataset_binding.DatasetBindingError("unexpected physical dataset binding")
        dataset_manifest["files"] = files
        dataset_manifest["dataset_bytes_verified"] = True
    with _ExperimentLock(experiment_dir):
        state_machine.require_state(current_state(experiment_dir), state_machine.SEALED,
                                    operation="bind-data")
        plan = load_artifact(experiment_dir, PLAN_FILE)
        seal = load_artifact(experiment_dir, SEAL_FILE)
        _verify_seal_integrity(seal)
        _verify_plan_unmutated(plan, seal)
        contract = plan["data_contract"]
        manifest_errors = dataset_binding.validate_dataset_manifest(dataset_manifest)
        if manifest_errors:
            raise dataset_binding.DatasetBindingError(
                "invalid dataset manifest:\n- " + "\n- ".join(manifest_errors))
        mismatches = dataset_binding.contract_mismatches(plan, dataset_manifest)
        if mismatches:
            raise dataset_binding.DatasetBindingError(
                "dataset manifest contradicts the sealed data contract (no holdout was "
                "claimed):\n- " + "\n- ".join(mismatches))
        claim = holdout_registry.claim_holdout(
            registry_root,
            dataset_fingerprint=dataset_binding.dataset_content_fingerprint(dataset_manifest),
            universe=contract["universe"],
            interval_start_utc=contract["untouched_evaluation_start_utc"],
            interval_end_utc=contract["untouched_evaluation_end_utc"],
            target_contract_sha256=sha256_payload(plan["target_contract"]),
            model_family=plan["code_contract"]["model_contract_version"],
            hypothesis_family=plan["multiple_testing_contract"]["family_id"],
            experiment_id=plan["identity"]["experiment_id"],
            plan_sha256=seal["plan_sha256"],
            claimed_at_utc=timestamp_utc,
            declare_forensic_reuse=declare_forensic_reuse,
        )
        binding = dataset_binding.build_dataset_binding(
            plan=plan, seal=seal, dataset_manifest=dataset_manifest,
            holdout_claim=claim,
            binding_id=_deterministic_id("binding", plan["identity"]["experiment_id"]),
            bound_at_utc=timestamp_utc)
        informational: list[tuple[str, Any]] = [
            ("HOLDOUT_CLAIMED", {"holdout_id": claim["holdout_id"],
                                 "status": claim["status"],
                                 "reuse_kind": claim["reuse_kind"]})]
        if claim["status"] == holdout_registry.FORENSIC_REUSE:
            informational.append(("FORENSIC_REUSE_DECLARED",
                                  {"holdout_id": claim["holdout_id"]}))
        try:
            files = _transition_files(
                experiment_dir, operation="bind-data",
                new_artifacts={BINDING_FILE: strict_json_text(binding).encode("utf-8")},
                informational_events=informational,
                transition_event_type="DATASET_BOUND",
                transition_payload={"binding_sha256": binding["binding_sha256"]},
                actor=actor, timestamp_utc=timestamp_utc)
            _publish_version(experiment_dir, files)
        except BaseException:
            # Compensate only a claim minted by this attempt.  Idempotently
            # recovered claims may belong to a prior interrupted publication.
            if not claim.get("idempotent"):
                holdout_registry.release_claim_if_owned(
                    registry_root, holdout_id=claim["holdout_id"],
                    experiment_id=plan["identity"]["experiment_id"],
                    plan_sha256=seal["plan_sha256"])
            raise
        return binding


def record_evidence(
    experiment_dir: Path,
    evidence_payload: dict[str, Any],
    *,
    actor: str = "researcher",
    timestamp_utc: str | None = None,
    artifact_bindings: list[dict[str, Any]] | None = None,
    artifact_root: Path | None = None,
    evaluation_rows_logical_path: str | None = None,
    robustness_rows_logical_path: str | None = None,
    test_receipts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Ingest and fully validate the structured evidence bundle."""
    experiment_dir = Path(experiment_dir)
    timestamp_utc = timestamp_utc or _now_utc()
    with _ExperimentLock(experiment_dir):
        state_machine.require_state(current_state(experiment_dir), state_machine.DATA_BOUND,
                                    operation="record-evidence")
        plan = load_artifact(experiment_dir, PLAN_FILE)
        seal = load_artifact(experiment_dir, SEAL_FILE)
        binding = load_artifact(experiment_dir, BINDING_FILE)
        _verify_seal_integrity(seal)
        _verify_plan_unmutated(plan, seal)
        payload = dict(evidence_payload)
        if "independent_verification" in payload:
            raise EvidenceValidationError(
                "independent_verification is Tribunal-generated and may not be supplied")
        if artifact_bindings is not None:
            if (artifact_root is None or evaluation_rows_logical_path is None
                    or robustness_rows_logical_path is None):
                raise EvidenceValidationError(
                    "artifact_root, evaluation_rows_logical_path, and "
                    "robustness_rows_logical_path are required")
            from engine.experiments.artifact_binding import bind_physical_files
            from engine.experiments.evaluation_rows import load_evaluation_rows
            from engine.experiments.evidence_recompute import (
                deterministic_block_bootstrap, recompute)
            verified = bind_physical_files(artifact_bindings, allowed_root=artifact_root)
            row_logical = normalize_logical_path(evaluation_rows_logical_path)
            raw_by_logical = {normalize_logical_path(item["logical_path"]): item
                              for item in artifact_bindings}
            if row_logical not in raw_by_logical:
                raise EvidenceValidationError("evaluation-row artifact binding is missing")
            robustness_logical = normalize_logical_path(robustness_rows_logical_path)
            if robustness_logical not in raw_by_logical:
                raise EvidenceValidationError("robustness-row artifact binding is missing")
            rows = load_evaluation_rows(
                Path(raw_by_logical[row_logical]["physical_path"]),
                class_order=plan["target_contract"]["class_order"],
                allowed_policies=plan["entry_policy_contract"]["sensitivity_populations"],
                allowed_boundaries=plan["entry_policy_contract"]["split_boundary_policies"],
                duplicate_policy=("reject" if "reject" in
                                  plan["data_contract"]["duplicate_row_policy"].lower()
                                  else plan["data_contract"]["duplicate_row_policy"]))
            calculated = recompute(rows, class_order=plan["target_contract"]["class_order"])
            comparisons = {
                "accepted_rows": payload["population"]["accepted_rows"],
                "class_counts": payload["population"]["class_counts"],
                "model_brier": payload["primary_model"]["multiclass_brier"],
                "comparator_brier": payload["primary_comparator"]["multiclass_brier"],
                "brier_improvement": payload["primary_model"]["brier_improvement"],
                "model_log_loss": payload["primary_model"]["multiclass_log_loss"],
                "comparator_log_loss": payload["primary_comparator"]["multiclass_log_loss"],
                "log_loss_improvement": payload["primary_model"]["log_loss_improvement"],
            }
            from engine.experiments.evidence_recompute import assert_summary_matches
            assert_summary_matches(
                {key: calculated[key] for key in comparisons}, comparisons,
                tolerance=1e-10)
            concentration_keys = [key for key in calculated if key.startswith("largest_")]
            assert_summary_matches(
                {key: calculated[key] for key in concentration_keys},
                {key: payload["concentration"][key] for key in concentration_keys},
                tolerance=1e-10)
            uncertainty = plan["uncertainty_contract"]
            bounds_by_block = {str(block): deterministic_block_bootstrap(
                rows, class_order=plan["target_contract"]["class_order"],
                block_length=block, replicate_count=uncertainty["replicate_count"],
                seed=uncertainty["random_seed"],
                confidence_level=uncertainty["one_sided_confidence"],
                two_sided_confidence_level=uncertainty["two_sided_confidence"])
                for block in uncertainty["block_sizes"]}
            bound_mapping = {
                "brier_lower_one_sided": "brier_improvement_lower_bound_one_sided",
                "brier_lower_two_sided": "brier_improvement_lower_bound_two_sided",
                "brier_upper_two_sided": "brier_improvement_upper_bound_two_sided",
                "log_loss_lower_one_sided": "log_loss_improvement_lower_bound_one_sided",
                "log_loss_lower_two_sided": "log_loss_improvement_lower_bound_two_sided",
                "log_loss_upper_two_sided": "log_loss_improvement_upper_bound_two_sided",
            }
            submitted_by_block = payload["primary_model"].get("bootstrap_by_block_length")
            if not isinstance(submitted_by_block, dict) or set(submitted_by_block) != set(bounds_by_block):
                raise EvidenceValidationError(
                    "every preregistered bootstrap block length must be reported")
            for block, bounds in bounds_by_block.items():
                assert_summary_matches(
                    bounds, {key: submitted_by_block[block][submitted]
                             for key, submitted in bound_mapping.items()}, tolerance=1e-10)
            if not test_receipts:
                raise EvidenceValidationError(
                    "physical test receipts are required for independent verification")
            from engine.experiments.test_receipts import verify_test_receipt
            required_commands = plan["code_contract"]["required_test_commands"]
            if len(test_receipts) != len(required_commands):
                raise EvidenceValidationError("physical receipts must exactly cover sealed commands")
            receipt_results = [verify_test_receipt(
                Path(item["receipt_path"]), Path(item["log_path"]),
                expected_commit=plan["code_contract"]["commit_sha"],
                required_command=command, now_utc=timestamp_utc)
                for item, command in zip(test_receipts, required_commands)]
            from engine.experiments.robustness_evidence import recompute_robustness_rows
            recomputed_cells = recompute_robustness_rows(
                Path(raw_by_logical[robustness_logical]["physical_path"]),
                contract=plan["robustness_contract"],
                class_order=plan["target_contract"]["class_order"])
            submitted_cells = {item["cell_id"]: item
                               for item in payload["robustness"]["cells"]}
            for cell in recomputed_cells:
                submitted = submitted_cells.get(cell["cell_id"])
                if submitted is None:
                    raise EvidenceValidationError("submitted robustness cell is missing")
                for key, expected in cell.items():
                    observed = submitted.get(key)
                    if isinstance(expected, float):
                        if not isinstance(observed, (int, float)) or abs(float(observed) - expected) > 1e-10:
                            raise EvidenceValidationError(
                                f"robustness cell {cell['cell_id']} {key} differs from recomputation")
                    elif observed != expected:
                        raise EvidenceValidationError(
                            f"robustness cell {cell['cell_id']} {key} differs from recomputation")
            payload["robustness"] = dict(payload["robustness"], cells=recomputed_cells)
            payload["artifacts"] = dict(payload["artifacts"])
            payload["artifacts"]["artifact_sha256"] = {
                item["logical_path"]: item["sha256"] for item in verified}
            payload["independent_verification"] = {
                "dataset_bytes_verified": binding.get("dataset_bytes_verified") is True,
                "evaluation_rows_verified": True,
                "metrics_recomputed": True,
                "artifact_bytes_verified": True,
                "concentrations_recomputed": True,
                "uncertainty_recomputed": True,
                "test_receipts_verified": True,
                "robustness_metrics_recomputed": True,
                "robustness_rows_logical_path": robustness_logical,
                "test_receipt_sha256": [item["receipt_sha256"]
                                         for item in receipt_results],
                "evaluation_rows_logical_path": row_logical,
            }
            payload.pop("evidence_sha256", None)
        validated = evidence_module.validate_evidence(
            payload, plan=plan, seal=seal, binding=binding)
        files = _transition_files(
            experiment_dir, operation="record-evidence",
            new_artifacts={EVIDENCE_FILE: strict_json_text(validated).encode("utf-8")},
            informational_events=[],
            transition_event_type="EVIDENCE_RECORDED",
            transition_payload={"evidence_sha256": validated["evidence_sha256"]},
            actor=actor, timestamp_utc=timestamp_utc)
        _publish_version(experiment_dir, files)
        return validated


def _fact(facts: dict[str, tuple[bool, str]], gate_id: str,
          check: Callable[[], bool], ok_reason: str) -> None:
    try:
        holds = bool(check())
        facts[gate_id] = (holds, ok_reason if holds else f"check failed: {ok_reason}")
    except TribunalError as exc:
        facts[gate_id] = (False, str(exc))
    except (KeyError, TypeError, ValueError) as exc:
        facts[gate_id] = (False, f"fact could not be established: {exc!r}")


def _collect_hard_facts(
    *, plan_bytes: bytes, plan: dict[str, Any], seal: dict[str, Any],
    binding: dict[str, Any], stored_evidence: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, tuple[bool, str]]:
    facts: dict[str, tuple[bool, str]] = {}

    def chain_ok() -> bool:
        audit_log.verify_chain(events)
        return events[-1]["state_after"] == state_machine.EVIDENCE_RECORDED
    _fact(facts, "hard_state_chain", chain_ok, "audit chain verifies and is in "
          "EVIDENCE_RECORDED")
    _fact(facts, "hard_audit_chain", lambda: (audit_log.verify_chain(events) or True),
          "audit hash chain verifies")
    _fact(facts, "hard_plan_hash",
          lambda: sha256_payload(load_strict_json_text(plan_bytes.decode("utf-8")))
          == seal["plan_sha256"],
          "plan bytes hash to the sealed plan_sha256")

    def seal_ok() -> bool:
        _verify_seal_integrity(seal)
        return True
    _fact(facts, "hard_seal_integrity", seal_ok, "seal is hash-consistent")
    _fact(facts, "hard_dataset_binding",
          lambda: dataset_binding.verify_binding_integrity(binding)
          and binding["seal_sha256"] == seal["seal_sha256"],
          "binding is hash-consistent and references this seal")
    _fact(facts, "hard_no_undeclared_amendment",
          lambda: bool(seal["amendment"]) == preregistration.is_amendment(plan),
          "amendment status is declared consistently")
    _fact(facts, "hard_no_clean_holdout_reuse",
          lambda: not (binding["holdout_reuse_kind"] != "none"
                       and binding["holdout_claim_status"]
                       == holdout_registry.SEALED_FOR_EXPERIMENT),
          "no reused interval was claimed as untouched")
    producer = stored_evidence.get("producer", {})
    _fact(facts, "hard_code_commit_match",
          lambda: producer.get("code_commit_sha") == plan["code_contract"]["commit_sha"],
          "evidence code commit matches preregistration")
    _fact(facts, "hard_configuration_match",
          lambda: producer.get("configuration_sha256")
          == plan["code_contract"]["configuration_sha256"],
          "evidence configuration matches preregistration")
    _fact(facts, "hard_model_contract_match",
          lambda: producer.get("model_contract_version")
          == plan["code_contract"]["model_contract_version"],
          "evidence model contract matches preregistration")
    bound_hashes = {entry["logical_path"]: entry["sha256"] for entry in binding["files"]}
    _fact(facts, "hard_dataset_hash_match",
          lambda: stored_evidence.get("dataset", {}).get("dataset_binding_sha256")
          == binding["binding_sha256"]
          and stored_evidence.get("dataset", {}).get("dataset_file_sha256") == bound_hashes,
          "evidence dataset identity matches the binding")

    def evidence_ok() -> bool:
        errors = evidence_module.evidence_errors(
            stored_evidence, plan=plan, seal=seal, binding=binding)
        if errors:
            raise TribunalError("evidence bundle invalid: " + "; ".join(errors[:5]))
        body = {key: value for key, value in stored_evidence.items()
                if key != "evidence_sha256"}
        return stored_evidence.get("evidence_sha256") == sha256_payload(body)
    _fact(facts, "hard_evidence_complete", evidence_ok,
          "evidence bundle is complete, valid, and hash-consistent")
    quality = stored_evidence.get("data_quality", {})
    _fact(facts, "hard_probabilities_valid",
          lambda: quality.get("probability_validity_confirmed") is True
          and quality.get("schema_validation_passed") is True,
          "producer confirmed probability validity and schema validation")

    def finite_ok() -> bool:
        sha256_payload({key: value for key, value in stored_evidence.items()
                        if key != "evidence_sha256"})
        return True
    _fact(facts, "hard_finite_metrics", finite_ok, "no NaN or infinity anywhere in evidence")
    _fact(facts, "hard_mandatory_controls",
          lambda: not (preregistration.REQUIRED_SECONDARY_COMPARATOR_KINDS
                       - set(stored_evidence.get("negative_controls", {}))),
          "every mandatory negative control is present")
    cells = stored_evidence.get("robustness", {}).get("cells", [])
    _fact(facts, "hard_mandatory_policies",
          lambda: set(preregistration.ENTRY_POLICY_ORDER)
          <= {cell.get("dimensions", {}).get("entry_policy") for cell in cells},
          "every mandatory entry-policy view is present")
    _fact(facts, "hard_mandatory_block_lengths",
          lambda: set(plan["robustness_contract"]["required_block_lengths"])
          <= {cell.get("dimensions", {}).get("block_length") for cell in cells},
          "every mandatory block length is present")

    def paths_ok() -> bool:
        artifacts = stored_evidence.get("artifacts", {})
        normalize_logical_path(artifacts.get("summary_path", ""))
        normalize_logical_path(artifacts.get("manifest_path", ""))
        for path in artifacts.get("artifact_sha256", {}):
            normalize_logical_path(path)
        return True
    _fact(facts, "hard_no_path_traversal", paths_ok, "no artifact path escapes its root")

    def unique_ids_ok() -> bool:
        cell_ids = [cell.get("cell_id") for cell in cells]
        hypothesis_ids = stored_evidence.get("multiplicity", {}).get("hypothesis_ids", [])
        return len(cell_ids) == len(set(cell_ids)) and \
            len(hypothesis_ids) == len(set(hypothesis_ids))
    _fact(facts, "hard_no_duplicate_evidence_ids", unique_ids_ok,
          "no duplicate evidence identifiers")
    independent = stored_evidence.get("independent_verification")
    if isinstance(independent, dict):
        for gate_id, key in (
            ("hard_dataset_bytes_verified", "dataset_bytes_verified"),
            ("hard_evaluation_rows_verified", "evaluation_rows_verified"),
            ("hard_metrics_recomputed", "metrics_recomputed"),
            ("hard_uncertainty_recomputed", "uncertainty_recomputed"),
            ("hard_test_receipts_verified", "test_receipts_verified"),
        ):
            _fact(facts, gate_id, lambda key=key: independent.get(key) is True,
                  f"Tribunal independently established {key}")
    return facts


def evaluate(
    experiment_dir: Path,
    *,
    registry_root: Path | None = None,
    actor: str = "researcher",
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Evaluate every gate and issue the single, final, deterministic verdict."""
    experiment_dir = Path(experiment_dir)
    timestamp_utc = timestamp_utc or _now_utc()
    with _ExperimentLock(experiment_dir):
        state_machine.require_state(current_state(experiment_dir),
                                    state_machine.EVIDENCE_RECORDED, operation="evaluate")
        if registry_root is None:
            raise TribunalError("registry_root is mandatory for verdict issuance")
        version_dir = _current_dir(experiment_dir)
        plan_bytes = (version_dir / PLAN_FILE).read_bytes()
        plan = load_artifact(experiment_dir, PLAN_FILE)
        seal = load_artifact(experiment_dir, SEAL_FILE)
        binding = load_artifact(experiment_dir, BINDING_FILE)
        stored_evidence = load_artifact(experiment_dir, EVIDENCE_FILE)
        events = load_events(experiment_dir)
        from engine.experiments.registry_reconciliation import reconcile_binding
        reconcile_binding(binding, registry_root)

        facts = _collect_hard_facts(
            plan_bytes=plan_bytes, plan=plan, seal=seal, binding=binding,
            stored_evidence=stored_evidence, events=events)
        hard_results = gates.evaluate_hard_validity_gates(facts)

        robustness_report = robustness.evaluate_robustness_matrix(
            robustness_contract=plan["robustness_contract"],
            cells=stored_evidence["robustness"]["cells"])
        concentration_report = robustness.evaluate_concentration(
            concentration_limits=plan["concentration_limits"],
            concentration_evidence=stored_evidence["concentration"],
            accepted_rows=stored_evidence["population"]["accepted_rows"])
        multiplicity_contract = plan["multiple_testing_contract"]
        multiplicity_evidence = stored_evidence["multiplicity"]
        multiplicity_report = multiplicity.apply_correction(
            method=multiplicity_contract["correction_method"],
            hypothesis_ids=list(multiplicity_evidence["hypothesis_ids"]),
            p_values=list(multiplicity_evidence["p_values"]),
            familywise_alpha=multiplicity_contract["familywise_alpha"],
            family_size=multiplicity_contract["family_size"],
            primary_hypothesis_index=multiplicity_contract["primary_hypothesis_index"])

        namespace: dict[str, Any] = dict(stored_evidence)
        namespace["derived"] = {
            "robustness_pass_ratio": robustness_report["pass_ratio"],
            "robustness_mandatory_failures": len(robustness_report["mandatory_cell_failures"]),
            "concentration_breaches": concentration_report["breach_count"],
            "concentration_dangerous":
                1 if concentration_report["status"] == "dangerously_concentrated" else 0,
            "multiplicity_primary_rejected":
                1 if multiplicity_report["primary_rejected"] else 0,
        }
        statistical_results = gates.evaluate_statistical_gates(
            plan["acceptance_gates"], namespace)

        verdict_binding = dict(binding)
        independent_verification = dict(
            stored_evidence.get("independent_verification", {}))
        independent_verification["registry_reconciled"] = True
        independent_verification["robustness_plan_complete"] = (
            independent_verification.get("robustness_metrics_recomputed") is True
            and not robustness_report["missing_cells"])
        verdict_binding["independent_verification"] = independent_verification
        verdict_payload = verdict_module.decide_verdict(
            plan=plan, binding=verdict_binding, hard_gate_results=hard_results,
            statistical_gate_results=statistical_results,
            robustness_report=robustness_report,
            concentration_report=concentration_report,
            multiplicity_report=multiplicity_report,
            evidence_sha256=stored_evidence.get("evidence_sha256", "0" * 64),
            verdict_id=_deterministic_id("verdict", binding["experiment_id"]),
            decided_at_utc=timestamp_utc)
        scorecard = reporting.build_scorecard(verdict_payload)

        artifact_hashes = {
            PLAN_FILE: sha256_bytes(plan_bytes),
            SEAL_FILE: sha256_bytes((version_dir / SEAL_FILE).read_bytes()),
            BINDING_FILE: sha256_bytes((version_dir / BINDING_FILE).read_bytes()),
            EVIDENCE_FILE: sha256_bytes((version_dir / EVIDENCE_FILE).read_bytes()),
        }
        gates_event_payload = {
            "hard_gates_passed": sum(1 for gate in hard_results if gate["status"] == gates.PASS),
            "hard_gates_total": len(hard_results),
            "statistical_gates_passed": sum(
                1 for gate in statistical_results if gate["status"] == gates.PASS),
            "statistical_gates_total": len(statistical_results),
        }
        # The verification artifact commits to the audit head *including* the
        # events this transition appends; assemble events first to know it.
        registry_receipt = holdout_registry.consume_holdout_idempotent(
            registry_root, binding["holdout_id"])
        registry_reconciliation = {
            "reconciliation_version": "edge-tribunal-registry-reconciliation-v1",
            "status": "COMPLETE", "experiment_id": binding["experiment_id"],
            "binding_sha256": binding["binding_sha256"],
            **registry_receipt,
        }
        registry_reconciliation["reconciliation_sha256"] = sha256_payload(
            registry_reconciliation)
        preview_files = _transition_files(
            experiment_dir, operation="evaluate",
            new_artifacts={VERDICT_FILE: strict_json_text(verdict_payload).encode("utf-8"),
                           SCORECARD_FILE: strict_json_text(scorecard).encode("utf-8"),
                           REGISTRY_RECONCILIATION_FILE:
                               strict_json_text(registry_reconciliation).encode("utf-8")},
            informational_events=[("GATES_EVALUATED", gates_event_payload),
                                  ("HOLDOUT_CONSUMED", registry_receipt),
                                  ("REGISTRY_RECONCILED", registry_reconciliation)],
            transition_event_type="VERDICT_ISSUED",
            transition_payload={"verdict": verdict_payload["verdict"],
                                "verdict_sha256": verdict_payload["verdict_sha256"]},
            actor=actor, timestamp_utc=timestamp_utc)
        final_events = [load_strict_json_text(line)
                        for line in preview_files[AUDIT_FILE].decode("utf-8").splitlines()
                        if line.strip()]
        verification = reporting.build_verification(
            verdict_payload=verdict_payload,
            audit_head_sha256=audit_log.chain_head_hash(final_events),
            audit_event_count=len(final_events),
            artifact_hashes=artifact_hashes)
        report_markdown = reporting.build_report_markdown(
            plan=plan, seal=seal, binding=binding,
            verdict_payload=verdict_payload, verification=verification)
        preview_files[VERIFICATION_FILE] = strict_json_text(verification).encode("utf-8")
        preview_files[REPORT_FILE] = report_markdown.encode("utf-8")
        _publish_version(experiment_dir, preview_files)

        return verdict_payload


def archive(
    experiment_dir: Path,
    *,
    actor: str = "researcher",
    timestamp_utc: str | None = None,
) -> None:
    experiment_dir = Path(experiment_dir)
    timestamp_utc = timestamp_utc or _now_utc()
    with _ExperimentLock(experiment_dir):
        files = _transition_files(
            experiment_dir, operation="archive", new_artifacts={},
            informational_events=[], transition_event_type="EXPERIMENT_ARCHIVED",
            transition_payload={}, actor=actor, timestamp_utc=timestamp_utc)
        _publish_version(experiment_dir, files)


def verify_experiment(experiment_dir: Path, *, registry_root: Path | None = None) -> dict[str, Any]:
    """Read-only end-to-end verification of every artifact and the audit chain."""
    experiment_dir = Path(experiment_dir)
    problems: list[str] = []
    state = None
    artifact_hashes: dict[str, str] = {}
    try:
        version_dir = _current_dir(experiment_dir)
    except TribunalError as exc:
        return {"ok": False, "state": None, "problems": [str(exc)], "artifact_sha256": {}}
    try:
        snapshot = _read_snapshot(version_dir)
    except TribunalError as exc:
        return {"ok": False, "state": None, "problems": [str(exc)],
                "artifact_sha256": {}}
    artifact_hashes = {name: sha256_bytes(content) for name, content in snapshot.items()}
    versions_root = experiment_dir / VERSIONS_DIR
    version_dirs = sorted(entry for entry in versions_root.iterdir()
                          if entry.is_dir() and entry.name.startswith("v"))
    expected_names = [f"v{index:06d}" for index in range(1, len(version_dirs) + 1)]
    if [entry.name for entry in version_dirs] != expected_names:
        problems.append("snapshot history has missing, duplicate, or non-contiguous versions")
    prior_name: str | None = None
    prior_hash: str | None = None
    for version in version_dirs:
        try:
            version_files = _read_snapshot(version)
            meta_raw = version_files.get(SNAPSHOT_META_FILE)
            if meta_raw is None:
                problems.append(f"{version.name}: missing snapshot history commitment")
                break
            meta = load_strict_json_text(meta_raw.decode("utf-8"))
            claimed_meta = meta.get("snapshot_meta_sha256")
            meta_body = {key: value for key, value in meta.items()
                         if key != "snapshot_meta_sha256"}
            if claimed_meta != sha256_payload(meta_body):
                problems.append(f"{version.name}: snapshot metadata hash mismatch")
            if (meta.get("version") != version.name
                    or meta.get("previous_version") != prior_name
                    or meta.get("previous_snapshot_sha256") != prior_hash):
                problems.append(f"{version.name}: previous-version commitment mismatch")
            payload = {name: content for name, content in version_files.items()
                       if name != SNAPSHOT_META_FILE}
            if meta.get("snapshot_payload_sha256") != _snapshot_sha256(payload):
                problems.append(f"{version.name}: snapshot payload commitment mismatch")
            prior_name = version.name
            prior_hash = _snapshot_sha256(version_files)
        except (TribunalError, CanonicalJsonError) as exc:
            problems.append(f"{version.name}: {exc}")
    if version_dirs and current_version_name(experiment_dir) != version_dirs[-1].name:
        problems.append("CURRENT does not point to the highest committed snapshot")

    events: list[dict[str, Any]] = []
    try:
        events = audit_log.read_events(version_dir / AUDIT_FILE)
        audit_log.verify_chain(events)
    except AuditIntegrityError as exc:
        problems.append(f"audit chain: {exc}")
    try:
        state = current_state(experiment_dir)
        if state in (state_machine.DATA_BOUND, state_machine.EVIDENCE_RECORDED,
                     state_machine.VERDICT_ISSUED, state_machine.ARCHIVED) \
                and registry_root is None:
            problems.append("registry_root is mandatory for DATA_BOUND and later verification")
        if events and state != events[-1]["state_after"]:
            problems.append(f"state file says {state}, audit chain says "
                            f"{events[-1]['state_after']}")
        if state in (state_machine.VERDICT_ISSUED, state_machine.ARCHIVED):
            reconciliation = load_artifact(experiment_dir, REGISTRY_RECONCILIATION_FILE)
            claimed = reconciliation.get("reconciliation_sha256")
            body = {key: value for key, value in reconciliation.items()
                    if key != "reconciliation_sha256"}
            if claimed != sha256_payload(body) or reconciliation.get("status") != "COMPLETE":
                problems.append("registry reconciliation artifact is invalid or incomplete")
            if registry_root is not None:
                binding = load_artifact(experiment_dir, BINDING_FILE)
                from engine.experiments.registry_reconciliation import reconcile_binding
                receipt = reconcile_binding(binding, registry_root)
                if receipt["status"] != holdout_registry.CONSUMED:
                    problems.append("registry holdout is not CONSUMED")
    except TribunalError as exc:
        problems.append(f"state: {exc}")

    plan = seal = binding = stored_evidence = None
    try:
        plan = load_artifact(experiment_dir, PLAN_FILE)
        preregistration.validate_plan(plan)
    except TribunalError as exc:
        problems.append(f"plan: {exc}")
    if SEAL_FILE in snapshot:
        try:
            seal = load_artifact(experiment_dir, SEAL_FILE)
            _verify_seal_integrity(seal)
            if plan is not None:
                _verify_plan_unmutated(plan, seal)
        except TribunalError as exc:
            problems.append(f"seal: {exc}")
    if BINDING_FILE in snapshot:
        try:
            binding = load_artifact(experiment_dir, BINDING_FILE)
            if not dataset_binding.verify_binding_integrity(binding):
                problems.append("binding: hash-inconsistent")
            elif seal is not None and binding["seal_sha256"] != seal["seal_sha256"]:
                problems.append("binding: does not reference the seal")
        except TribunalError as exc:
            problems.append(f"binding: {exc}")
    if EVIDENCE_FILE in snapshot and plan is not None and seal is not None \
            and binding is not None:
        try:
            stored_evidence = load_artifact(experiment_dir, EVIDENCE_FILE)
            errors = evidence_module.evidence_errors(
                stored_evidence, plan=plan, seal=seal, binding=binding)
            problems.extend(f"evidence: {error}" for error in errors)
        except TribunalError as exc:
            problems.append(f"evidence: {exc}")
    if VERDICT_FILE in snapshot:
        try:
            verdict_payload = load_artifact(experiment_dir, VERDICT_FILE)
            if not verdict_module.verify_verdict_integrity(verdict_payload):
                problems.append("verdict: hash-inconsistent or malformed")
            elif stored_evidence is not None and \
                    verdict_payload["evidence_sha256"] != stored_evidence.get("evidence_sha256"):
                problems.append("verdict: does not reference the recorded evidence")
        except TribunalError as exc:
            problems.append(f"verdict: {exc}")
    for filename, content in snapshot.items():
        if filename.endswith(".json"):
            try:
                _validate_file_schema(filename, load_strict_json_text(content.decode("utf-8")))
            except (TribunalError, CanonicalJsonError) as exc:
                problems.append(str(exc))
    return {"ok": not problems, "state": state, "problems": problems,
            "artifact_sha256": artifact_hashes}


def reconcile_registry(experiment_dir: Path, registry_root: Path) -> dict[str, Any]:
    """Reconcile the idempotent registry side of an interrupted transition."""
    binding = load_artifact(experiment_dir, BINDING_FILE)
    from engine.experiments.registry_reconciliation import (
        reconcile_binding, reconcile_verdict)
    if current_state(experiment_dir) in (state_machine.VERDICT_ISSUED,
                                         state_machine.ARCHIVED):
        return reconcile_verdict(binding, registry_root)
    return reconcile_binding(binding, registry_root)


def show_state(experiment_dir: Path) -> dict[str, Any]:
    """Read-only snapshot summary."""
    experiment_dir = Path(experiment_dir)
    version_dir = _current_dir(experiment_dir)
    snapshot = _read_snapshot(version_dir)
    return {
        "experiment_dir": str(experiment_dir),
        "version": current_version_name(experiment_dir),
        "state": current_state(experiment_dir),
        "artifacts": {name: sha256_bytes(content) for name, content in sorted(snapshot.items())},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_json_file(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise TribunalError(f"file does not exist: {path}")
    payload = load_strict_json_text(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TribunalError(f"{path} is not a JSON object")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m engine.experiments.edge_tribunal",
        description="Edge Tribunal: preregistered falsification and research-promotion "
                    "governance. Verdicts are research decisions, never trading "
                    "authorizations.")
    parser.add_argument("--debug", action="store_true",
                        help="show full tracebacks for unexpected defects")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--experiment-dir", required=True)
        sub.add_argument("--actor", default="researcher")
        sub.add_argument("--timestamp", default=None,
                         help="deterministic UTC timestamp override")
        return sub

    init_cmd = add("init", "register a DRAFT experiment from a plan")
    init_cmd.add_argument("--plan", required=True)
    add("seal", "seal the plan irreversibly")
    bind_cmd = add("bind-data", "bind the dataset and claim the holdout")
    bind_cmd.add_argument("--dataset-manifest", required=True)
    bind_cmd.add_argument("--registry-root", required=True)
    bind_cmd.add_argument("--declare-forensic-reuse", action="store_true")
    record_cmd = add("record-evidence", "ingest the structured evidence bundle")
    record_cmd.add_argument("--evidence", required=True)
    evaluate_cmd = add("evaluate", "evaluate all gates and issue the verdict")
    evaluate_cmd.add_argument("--registry-root", required=True)
    add("archive", "archive a verdict-issued experiment")
    verify_cmd = subparsers.add_parser("verify", help="read-only artifact verification")
    verify_cmd.add_argument("--experiment-dir", required=True)
    verify_cmd.add_argument("--registry-root", required=True)
    reconcile_cmd = subparsers.add_parser("reconcile", help="reconcile holdout registry state")
    reconcile_cmd.add_argument("--experiment-dir", required=True)
    reconcile_cmd.add_argument("--registry-root", required=True)
    report_cmd = subparsers.add_parser("report", help="print the issued report")
    report_cmd.add_argument("--experiment-dir", required=True)
    show_cmd = subparsers.add_parser("show-state", help="read-only state summary")
    show_cmd.add_argument("--experiment-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    debug = getattr(args, "debug", False)
    try:
        if args.command == "init":
            plan = _load_json_file(args.plan)
            version = init_experiment(Path(args.experiment_dir), plan, actor=args.actor,
                                      timestamp_utc=args.timestamp)
            print(f"initialized DRAFT experiment at {args.experiment_dir} ({version})")
        elif args.command == "seal":
            seal = seal_experiment(Path(args.experiment_dir), actor=args.actor,
                                   timestamp_utc=args.timestamp)
            print(f"sealed plan {seal['plan_sha256']}")
        elif args.command == "bind-data":
            manifest = _load_json_file(args.dataset_manifest)
            binding = bind_data(Path(args.experiment_dir), manifest,
                                registry_root=Path(args.registry_root), actor=args.actor,
                                timestamp_utc=args.timestamp,
                                declare_forensic_reuse=args.declare_forensic_reuse)
            print(f"bound dataset {binding['dataset_manifest_sha256']} "
                  f"(holdout {binding['holdout_id']}, "
                  f"status {binding['holdout_claim_status']})")
        elif args.command == "record-evidence":
            payload = _load_json_file(args.evidence)
            validated = record_evidence(Path(args.experiment_dir), payload, actor=args.actor,
                                        timestamp_utc=args.timestamp)
            print(f"recorded evidence {validated['evidence_sha256']}")
        elif args.command == "evaluate":
            verdict_payload = evaluate(Path(args.experiment_dir),
                                       registry_root=Path(args.registry_root),
                                       actor=args.actor, timestamp_utc=args.timestamp)
            print(f"VERDICT: {verdict_payload['verdict']}")
            print(f"maximum next stage: {verdict_payload['maximum_next_stage']}")
        elif args.command == "archive":
            archive(Path(args.experiment_dir), actor=args.actor,
                    timestamp_utc=args.timestamp)
            print("experiment archived")
        elif args.command == "verify":
            result = verify_experiment(
                Path(args.experiment_dir),
                registry_root=Path(args.registry_root) if args.registry_root else None)
            print(strict_json_text(result), end="")
            return 0 if result["ok"] else 1
        elif args.command == "reconcile":
            result = reconcile_registry(Path(args.experiment_dir), Path(args.registry_root))
            print(strict_json_text(result), end="")
        elif args.command == "report":
            experiment_dir = Path(args.experiment_dir)
            state = current_state(experiment_dir)
            if state not in (state_machine.VERDICT_ISSUED, state_machine.ARCHIVED):
                raise VerdictIntegrityError(
                    f"no verdict has been issued (state: {state}); report refuses to exist "
                    f"before its verdict")
            print((_current_dir(experiment_dir) / REPORT_FILE).read_text(encoding="utf-8"))
        elif args.command == "show-state":
            print(strict_json_text(show_state(Path(args.experiment_dir))), end="")
        return 0
    except (LockError, TribunalError) as exc:
        if debug:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

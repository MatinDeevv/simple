from __future__ import annotations
import hashlib, subprocess
from datetime import datetime, timezone
from pathlib import Path
import pytest
from engine.core import run_manifest as rm

def _git(root: Path, *args: str): return subprocess.run(["git","-c","user.email=t@e.com","-c","user.name=t",*args],cwd=root,check=True,capture_output=True,text=True)
def _repo(tmp: Path) -> tuple[Path,Path]:
    root=tmp/"repo"; root.mkdir(); _git(root,"init","-q"); source=root/"source.py"; source.write_text("x=1\n"); _git(root,"add","source.py"); _git(root,"commit","-qm","baseline"); return root,source
def _evidence(root:Path) -> list[dict]:
    commit,_=rm.git_info(root); now=datetime.now(timezone.utc).isoformat(); digest="a"*64
    return [{"command":"python -m pytest tests -q","exit_code":0,"commit_sha":commit,"started_at_utc":now,"completed_at_utc":now,"artifact_sha256":digest,"python_version":"3.11","dependency_snapshot_sha256":digest}]

def test_bare_boolean_cannot_promote(tmp_path):
    root,source=_repo(tmp_path); manifest=rm.build_run_manifest(root=root,frozen_contract_version="v1",required_tests_passed=True,source_files=[source],holdout_status="clean_holdout")
    assert not manifest["promotion_eligible"] and "test evidence" in manifest["promotion_blockers"][-1]
def test_evidence_bound_to_commit_promotes(tmp_path):
    root,source=_repo(tmp_path); manifest=rm.build_run_manifest(root=root,frozen_contract_version="v1",source_files=[source],holdout_status="clean_holdout",test_evidence=_evidence(root))
    assert manifest["promotion_eligible"] and rm.verify_manifest_integrity(manifest)
def test_logical_path_hashes_staged_bytes(tmp_path):
    root,source=_repo(tmp_path); staged=root/"tmp.bin"; staged.write_bytes(b"published")
    manifest=rm.build_run_manifest(root=root,frozen_contract_version="v1",source_files=[source],output_artifacts=[rm.ArtifactBinding(staged,"artifacts/final.bin")])
    assert manifest["output_artifact_sha256"] == {"artifacts/final.bin":hashlib.sha256(b"published").hexdigest()}
def test_bad_logical_paths_rejected(tmp_path):
    root,source=_repo(tmp_path); out=root/"x"; out.write_text("x")
    with pytest.raises(rm.RunManifestError): rm.build_run_manifest(root=root,frozen_contract_version="v1",source_files=[source],output_artifacts=[rm.ArtifactBinding(out,"../x")])
    with pytest.raises(rm.RunManifestError): rm.build_run_manifest(root=root,frozen_contract_version="v1",source_files=[source],output_artifacts=[rm.ArtifactBinding(out,"x"),rm.ArtifactBinding(out,"x")])
def test_verified_read_rejects_tampering_and_forensic_mode_is_explicit(tmp_path):
    root,source=_repo(tmp_path); payload=rm.build_run_manifest(root=root,frozen_contract_version="v1",source_files=[source]); path=rm.write_manifest(payload,tmp_path/"manifest.json"); payload["run_id"]="bad"; path.write_text(__import__('json').dumps(payload))
    with pytest.raises(rm.RunManifestError): rm.read_manifest(path)
    assert rm.read_manifest(path,verify=False)["run_id"] == "bad"

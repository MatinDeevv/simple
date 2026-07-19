from __future__ import annotations
from engine.core import run_manifest as rm
import subprocess
from datetime import datetime, timezone
import pytest

def _repo(tmp_path):
    root=tmp_path/"repo"; root.mkdir(); source=root/"s.py"; source.write_text("x=1")
    subprocess.run(["git","init","-q"],cwd=root,check=True); subprocess.run(["git","-c","user.email=t@e.com","-c","user.name=t","add","s.py"],cwd=root,check=True); subprocess.run(["git","-c","user.email=t@e.com","-c","user.name=t","commit","-qm","baseline"],cwd=root,check=True); return root,source
def _evidence(root):
    commit,_=rm.git_info(root); now=datetime.now(timezone.utc).isoformat(); return [{"command":"pytest","exit_code":0,"commit_sha":commit,"started_at_utc":now,"completed_at_utc":now,"artifact_sha256":"a"*64,"python_version":"3.11","dependency_snapshot_sha256":"a"*64}]
def test_wrong_commit_and_nonzero_evidence_never_promote(tmp_path):
    root,source=_repo(tmp_path); evidence=_evidence(root); evidence[0]["commit_sha"]="b"*40
    assert not rm.build_run_manifest(root=root,frozen_contract_version="v1",source_files=[source],test_evidence=evidence)["promotion_eligible"]
    evidence=_evidence(root); evidence[0]["exit_code"]=1
    assert not rm.build_run_manifest(root=root,frozen_contract_version="v1",source_files=[source],test_evidence=evidence)["promotion_eligible"]
def test_bad_evidence_hash_rejected(tmp_path):
    root,source=_repo(tmp_path); evidence=_evidence(root); evidence[0]["artifact_sha256"]="bad"
    with pytest.raises(rm.RunManifestError): rm.build_run_manifest(root=root,frozen_contract_version="v1",source_files=[source],test_evidence=evidence)

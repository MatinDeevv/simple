from __future__ import annotations
import hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
import pytest
from engine.core import run_manifest as rm

def _repo(tmp_path):
    root=tmp_path/"repo"; root.mkdir(); source=root/"s.py"; source.write_text("x=1\n")
    subprocess.run(["git","init","-q"],cwd=root,check=True); subprocess.run(["git","-c","user.email=t@e","-c","user.name=t","add","s.py"],cwd=root,check=True); subprocess.run(["git","-c","user.email=t@e","-c","user.name=t","commit","-qm","base"],cwd=root,check=True); return root,source
def _binding(tmp_path,root,deps):
    commit,_=rm.git_info(root); dep=hashlib.sha256(rm.canonical_json(deps).encode()).hexdigest(); log=tmp_path/"run.log"; log.write_text("passed\n"); now=datetime.now(timezone.utc).isoformat()
    p={"receipt_version":"simple-test-receipt-v2","command_argv":["python","-m","pytest"],"exit_code":0,"started_at_utc":now,"completed_at_utc":now,"commit_sha":commit,"python_version":"3.11","dependency_snapshot_sha256":dep,"logical_log_path":"logs/run.log","log_path":"run.log","log_sha256":rm.sha256_file(log),"runner_version":"2","working_tree_status":"clean","environment_policy":"minimal-test-v1"}; p["receipt_sha256"]=hashlib.sha256(rm.canonical_json(p).encode()).hexdigest(); receipt=tmp_path/"receipt.json"; receipt.write_text(json.dumps(p)); return rm.ReceiptBinding(receipt,"receipts/receipt.json"),log
def test_physical_log_bytes_are_bound(tmp_path):
    root,source=_repo(tmp_path); deps={"engine":"x"}; binding,log=_binding(tmp_path,root,deps); log.write_text("tampered\n")
    with pytest.raises(rm.RunManifestError,match="log hash mismatch"): rm.build_run_manifest(root=root,frozen_contract_version="v",source_files=[source],dependency_versions=deps,test_receipts=[binding])
def test_self_attested_evidence_cannot_promote(tmp_path):
    root,source=_repo(tmp_path); now=datetime.now(timezone.utc).isoformat(); commit,_=rm.git_info(root); deps={"engine":"x"}; dep=hashlib.sha256(rm.canonical_json(deps).encode()).hexdigest(); evidence=[{"command":"pytest","exit_code":0,"commit_sha":commit,"started_at_utc":now,"completed_at_utc":now,"artifact_sha256":"a"*64,"python_version":"3.11","dependency_snapshot_sha256":dep}]
    result=rm.build_run_manifest(root=root,frozen_contract_version="v",source_files=[source],dependency_versions=deps,test_evidence=evidence,required_tests_passed=True,holdout_status="clean_holdout")
    assert not result["promotion_eligible"]
def test_receipt_cannot_define_required_command_policy(tmp_path):
    root,source=_repo(tmp_path); deps={"engine":"x"}; binding,_=_binding(tmp_path,root,deps)
    result=rm.build_run_manifest(root=root,frozen_contract_version="v",source_files=[source],dependency_versions=deps,test_receipts=[binding],required_tests_passed=True,holdout_status="clean_holdout")
    assert not result["promotion_eligible"] and any("at least one required" in item for item in result["promotion_blockers"])
def test_preregistered_argv_must_exactly_match_receipt(tmp_path):
    root,source=_repo(tmp_path); deps={"engine":"x"}; binding,_=_binding(tmp_path,root,deps)
    result=rm.build_run_manifest(root=root,frozen_contract_version="v",source_files=[source],dependency_versions=deps,test_receipts=[binding],required_test_commands=[["python","-m","pytest","tests/tiny.py"]],required_tests_passed=True,holdout_status="clean_holdout")
    assert not result["promotion_eligible"]
def test_unknown_receipt_field_fails_strict_ingestion(tmp_path):
    root,source=_repo(tmp_path); deps={"engine":"x"}; binding,_=_binding(tmp_path,root,deps); payload=json.loads(binding.receipt_path.read_text()); payload["extra"]=True; payload["receipt_sha256"]=hashlib.sha256(rm.canonical_json({k:v for k,v in payload.items() if k!="receipt_sha256"}).encode()).hexdigest(); binding.receipt_path.write_text(json.dumps(payload))
    with pytest.raises(rm.RunManifestError,match="strict v2"): rm.build_run_manifest(root=root,frozen_contract_version="v",source_files=[source],dependency_versions=deps,test_receipts=[binding],required_test_commands=[["python","-m","pytest"]])
def test_atomic_default_refuses_existing_destination(tmp_path):
    root,source=_repo(tmp_path); manifest=rm.build_run_manifest(root=root,frozen_contract_version="v",source_files=[source]); out=tmp_path/"manifest.json"; rm.write_manifest(manifest,out)
    with pytest.raises(rm.RunManifestError,match="overwrite"): rm.write_manifest(manifest,out)

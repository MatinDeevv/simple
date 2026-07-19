from __future__ import annotations
import hashlib, importlib.util, json, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pytest

SCRIPTS=Path(__file__).parents[1]/"plugins/simple-agent-framework/scripts"
sys.path.insert(0,str(SCRIPTS)); import receipt_contract as rc

def _receipt(tmp_path: Path):
    log=tmp_path/"test.log"; log.write_text("ok\n")
    now=datetime.now(timezone.utc).isoformat()
    p={"receipt_version":rc.VERSION,"command_argv":["python","-m","pytest"],"exit_code":0,"started_at_utc":now,"completed_at_utc":now,"commit_sha":"a"*40,"python_version":"3.11","dependency_snapshot_sha256":"b"*64,"logical_log_path":".agents/receipts/test.log","log_path":"test.log","log_sha256":hashlib.sha256(log.read_bytes()).hexdigest(),"runner_version":"2","working_tree_status":"clean","environment_policy":"minimal-test-v1"}
    p["receipt_sha256"]=hashlib.sha256(rc.canonical(p)).hexdigest(); path=tmp_path/"receipt.json"; path.write_text(json.dumps(p)); return path,p

def test_strict_receipt_and_log_verify(tmp_path): assert rc.verify(_receipt(tmp_path)[0])["exit_code"]==0
def test_unknown_field_and_traversal_fail(tmp_path):
    path,p=_receipt(tmp_path); p["extra"]=1; path.write_text(json.dumps(p))
    with pytest.raises(rc.ReceiptError): rc.verify(path)
    path,p=_receipt(tmp_path); p["log_path"]="../test.log"; p["receipt_sha256"]=hashlib.sha256(rc.canonical({k:v for k,v in p.items() if k!="receipt_sha256"})).hexdigest(); path.write_text(json.dumps(p))
    with pytest.raises(rc.ReceiptError): rc.verify(path)
def test_duplicate_keys_and_future_time_fail(tmp_path):
    path,p=_receipt(tmp_path); path.write_text('{"receipt_version":"x","receipt_version":"y"}')
    with pytest.raises(rc.ReceiptError): rc.verify(path)
    path,p=_receipt(tmp_path); p["completed_at_utc"]=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat(); p["receipt_sha256"]=hashlib.sha256(rc.canonical({k:v for k,v in p.items() if k!="receipt_sha256"})).hexdigest(); path.write_text(json.dumps(p))
    with pytest.raises(rc.ReceiptError): rc.verify(path)

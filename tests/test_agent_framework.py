"""Focused safety contracts for the repository-local agent plugin."""
from __future__ import annotations

import importlib.util
import sys
import json
import hashlib
import shutil, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MCP = ROOT / "plugins" / "simple-agent-framework" / "mcp" / "research_mcp.py"
spec = importlib.util.spec_from_file_location("simple_research_test", MCP)
assert spec and spec.loader
mcp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp)


def test_mcp_config_contains_no_machine_specific_path() -> None:
    config = json.loads((ROOT / "plugins" / "simple-agent-framework" / ".mcp.json").read_text())
    rendered = json.dumps(config).lower()
    assert "c:\\users\\" not in rendered
    assert "c:\\python" not in rendered


def test_runtime_validation_rejects_unknown_and_wrong_type() -> None:
    assert mcp._matches(mcp._tool_schema("research_search_papers"), {"query": "abc", "extra": 1})
    assert mcp._matches(mcp._tool_schema("market_dukascopy_history_url"), {"instrument": "EURUSD", "year": True, "month": 1, "day": 1, "hour_utc": 0})
    assert mcp._matches(mcp._tool_schema("research_search_papers"), {"query": "abc"}) is None


def test_dukascopy_network_policy_rejects_nonofficial_destinations() -> None:
    valid = "https://datafeed.dukascopy.com/datafeed/EURUSD/2024/00/01/00h_ticks.bi5"
    assert mcp._dukascopy_url(valid) == valid
    for bad in ("http://datafeed.dukascopy.com/datafeed/EURUSD/2024/00/01/00h_ticks.bi5", "https://datafeed.dukascopy.com.evil.test/datafeed/EURUSD/2024/00/01/00h_ticks.bi5", "https://localhost/datafeed/EURUSD/2024/00/01/00h_ticks.bi5", "https://datafeed.dukascopy.com:444/datafeed/EURUSD/2024/00/01/00h_ticks.bi5", "https://datafeed.dukascopy.com/datafeed/EURUSD/2024/12/01/00h_ticks.bi5"):
        try:
            mcp._dukascopy_url(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)


def test_receipt_self_hash_contract() -> None:
    payload = {"command_argv": ["python", "-c", "pass"], "exit_code": 0, "started_at_utc": "2026-01-01T00:00:00+00:00", "completed_at_utc": "2026-01-01T00:00:01+00:00", "commit_sha": "a" * 40, "python_version": "3.11", "log_path": "test.log", "log_sha256": "b" * 64}
    payload["receipt_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    assert payload["receipt_sha256"] == hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()

def test_mcp_launches_from_copied_plugin_and_external_cwd(tmp_path) -> None:
    copied=tmp_path/"copied-plugin"; shutil.copytree(ROOT/"plugins/simple-agent-framework",copied)
    outside=tmp_path/"outside"; outside.mkdir()
    request=json.dumps({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}})+"\n"
    proc=subprocess.run([sys.executable,str(copied/"mcp/research_mcp.py")],cwd=outside,input=request,text=True,capture_output=True,timeout=15)
    assert proc.returncode==0,proc.stderr
    assert json.loads(proc.stdout)["result"]["serverInfo"]["name"]=="simple-research"

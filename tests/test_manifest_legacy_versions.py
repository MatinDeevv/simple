from __future__ import annotations
import hashlib, json
from pathlib import Path
import pytest
from engine.core import run_manifest as rm

def _write(path: Path, version: str):
    payload = {"manifest_schema_version": version, "value": 1}
    payload["manifest_sha256"] = hashlib.sha256(rm.canonical_json(payload).encode()).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")

def test_known_v1_is_forensic_readable(tmp_path):
    path=tmp_path/"v1.json"; _write(path, rm.V1)
    assert rm.read_manifest(path)["manifest_schema_version"] == rm.V1

@pytest.mark.parametrize("version", ["random", "fxsim-research-run-manifest-v3"])
def test_unknown_and_future_versions_fail(tmp_path, version):
    path=tmp_path/"bad.json"; _write(path, version)
    with pytest.raises(rm.RunManifestError): rm.read_manifest(path)

def test_duplicate_keys_fail(tmp_path):
    path=tmp_path/"dup.json"; path.write_text('{"manifest_schema_version":"x","manifest_schema_version":"y"}')
    with pytest.raises(rm.RunManifestError, match="duplicate"): rm.read_manifest(path, verify=False)

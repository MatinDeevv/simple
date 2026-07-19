from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from engine.core import run_manifest as rm
from engine.core import schema_validate as sv
from engine.models.classical import simulate_integrator as integrator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "engine" / "config" / "schemas"
ALL_SCHEMAS = sorted(SCHEMA_DIR.glob("*.schema.json"))


def _load(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def test_at_least_the_five_required_schemas_exist() -> None:
    required_names = {
        "stat-arb-emission.schema.json", "stat-arb-summary.schema.json",
        "legal-event-study.schema.json", "integrator-checkpoint.schema.json",
        "research-run-manifest.schema.json",
    }
    present = {path.name for path in ALL_SCHEMAS}
    assert required_names <= present


@pytest.mark.parametrize("schema_path", ALL_SCHEMAS, ids=lambda p: p.name)
def test_every_schema_document_is_well_formed(schema_path: Path) -> None:
    document = json.loads(schema_path.read_text(encoding="utf-8"))
    error = sv.validate_schema_document(document)
    assert error is None, f"{schema_path.name}: {error}"


def test_validate_schema_document_rejects_malformed_documents() -> None:
    assert sv.validate_schema_document("not a dict") is not None
    assert sv.validate_schema_document({}) is not None
    assert sv.validate_schema_document({"schema_version": "v1", "title": "t",
                                        "type": "array", "properties": {}}) is not None
    assert sv.validate_schema_document({"schema_version": "v1", "title": "t", "type": "object",
                                        "properties": {"a": {"type": "string"}},
                                        "required": ["missing_field"]}) is not None


def test_validate_instance_catches_missing_required_and_wrong_type() -> None:
    schema = _load("integrator-checkpoint.schema.json")
    valid = {
        "version": "integrator-1.1.0", "pair_scope": ["EURUSD"], "state_index": 1,
        "next_arrival_index": 2, "state_timestamp": "2024-01-01T00:00:00+00:00",
        "x_hat": [0.1], "v_hat": [0.0], "restore_rule": "resume once",
    }
    assert sv.validate_instance(schema, valid) == []

    missing = dict(valid)
    del missing["state_index"]
    errors = sv.validate_instance(schema, missing)
    assert any("state_index" in error for error in errors)

    wrong_type = dict(valid)
    wrong_type["state_index"] = "not an int"
    errors = sv.validate_instance(schema, wrong_type)
    assert any("state_index" in error for error in errors)


def test_validate_instance_enforces_enum() -> None:
    schema = _load("legal-event-study.schema.json")
    row = {
        "event_id": "e1", "source_document_id": "d1", "jurisdiction": "US", "authority": "SEC",
        "legal_stage": "enacted", "known_at": "2024-01-01T00:00:00+00:00",
        "assessment_created_at": "2024-01-01T00:00:00+00:00", "decision_at": "2024-01-01T00:01:00+00:00",
        "assessment_sha256": "a" * 64, "parent_assessment_sha256": None,
        "prediction_time": "2024-01-01T00:01:00+00:00", "target_time": "2024-01-01T01:01:00+00:00",
        "pair": "EURUSD", "scenario_weighted_pressure": 0.5, "predicted_direction": 1,
        "post_event_log_return": 0.001, "pre_event_log_return": 0.0002,
        "expected_post_event_log_return": 0.0001, "baseline_adjusted_abnormal_log_return": 0.0009,
        "event_content_sha256": "b" * 64,
    }
    assert sv.validate_instance(schema, row) == []
    bad = dict(row)
    bad["predicted_direction"] = 2
    errors = sv.validate_instance(schema, bad)
    assert any("predicted_direction" in error for error in errors)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), np.float64("nan")])
def test_non_finite_numbers_are_rejected(value: float) -> None:
    schema = {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}
    assert any("$.x" in error for error in sv.validate_instance(schema, {"x": value}))


def test_integer_rejects_bool_and_nullable_union_is_supported() -> None:
    integer = {"type": "integer"}
    assert sv.validate_instance(integer, True)
    nullable = {"type": ["string", "null"]}
    assert sv.validate_instance(nullable, None) == []


def test_real_run_manifest_validates_against_its_schema(tmp_path: Path) -> None:
    import subprocess
    subprocess.run(["git", "-c", "user.email=t@e.com", "-c", "user.name=t", "init", "-q"],
                   cwd=tmp_path, check=True)
    source = tmp_path / "source.py"
    source.write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-c", "user.email=t@e.com", "-c", "user.name=t", "add", "source.py"],
                   cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=t@e.com", "-c", "user.name=t", "commit", "-q", "-m", "init"],
                   cwd=tmp_path, check=True)
    manifest = rm.build_run_manifest(root=tmp_path, frozen_contract_version="v1",
                                     required_tests_passed=True, holdout_status="clean_holdout",
                                     source_files=[source])
    schema = _load("research-run-manifest.schema.json")
    errors = sv.validate_instance(schema, manifest)
    assert errors == [], errors


def test_real_integrator_checkpoint_validates_against_its_schema(tmp_path: Path) -> None:
    times = np.arange(10, dtype=np.int64) * integrator.DT_NOM_NS
    x_hat = np.array([0.1] * len(integrator.PAIR_INDICES))
    v_hat = np.array([0.0] * len(integrator.PAIR_INDICES))
    checkpoint_path = tmp_path / "checkpoint.json"
    integrator.write_checkpoint(checkpoint_path, next_arrival_index=5, times=times,
                                x_hat=x_hat, v_hat=v_hat)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    schema = _load("integrator-checkpoint.schema.json")
    errors = sv.validate_instance(schema, payload)
    assert errors == [], errors

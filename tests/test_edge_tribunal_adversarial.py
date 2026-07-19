from __future__ import annotations

import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from engine.experiments import edge_tribunal as et
from engine.experiments.canonical import (
    canonical_json_bytes,
    is_git_commit_sha,
    is_sha256_hex,
    load_strict_json_text,
    normalize_logical_path,
    normalize_utc_timestamp,
    normalize_uuid,
    sha256_payload,
)
from engine.experiments.errors import CanonicalJsonError, PathSecurityError
from test_edge_tribunal_evidence import run_pipeline_to_bound


def test_key_order_and_unicode_normalization_are_deterministic() -> None:
    assert sha256_payload({"b": 2, "a": 1}) == sha256_payload({"a": 1, "b": 2})
    assert normalize_logical_path("folder\\caf\u00e9.json") == "folder/caf\u00e9.json"
    assert normalize_logical_path("folder/cafe\u0301.json") == "folder/caf\u00e9.json"


def test_deep_json_is_rejected_at_documented_limit() -> None:
    payload: object = 0
    for _ in range(66):
        payload = [payload]
    with pytest.raises(CanonicalJsonError, match="nesting"):
        canonical_json_bytes(payload)


def test_duplicate_keys_and_partial_json_are_rejected() -> None:
    with pytest.raises(CanonicalJsonError, match="duplicate"):
        load_strict_json_text('{"a":1,"a":2}')
    with pytest.raises(CanonicalJsonError, match="invalid JSON"):
        load_strict_json_text('{"a":')


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_numbers_are_rejected(value: float) -> None:
    with pytest.raises(CanonicalJsonError, match="NaN and infinity"):
        canonical_json_bytes({"metric": value})


def test_giant_integer_is_rejected_with_a_controlled_error() -> None:
    giant = 10 ** 10000
    with pytest.raises(CanonicalJsonError, match="integer exceeds"):
        canonical_json_bytes({"value": giant})


@pytest.mark.parametrize(
    "path",
    ["../secret", "a/../secret", "/absolute", "C:\\secret", "a//b", "a/./b", "\\\\server\\share"],
)
def test_absolute_traversal_and_ambiguous_paths_are_rejected(path: str) -> None:
    with pytest.raises(PathSecurityError):
        normalize_logical_path(path)


def test_hash_validation_rejects_wrong_length_and_uppercase() -> None:
    assert is_sha256_hex("a" * 64)
    assert not is_sha256_hex("a" * 63)
    assert not is_sha256_hex("A" * 64)
    assert is_git_commit_sha("b" * 40)
    assert not is_git_commit_sha("B" * 40)


@pytest.mark.parametrize("value", ["not-a-uuid", "../escape", "00000000-0000-4000-8000-0000000000012"])
def test_malformed_experiment_ids_are_rejected(value: str) -> None:
    with pytest.raises(CanonicalJsonError, match="malformed UUID"):
        normalize_uuid(value)


def test_timestamps_require_timezone_and_normalize_to_utc() -> None:
    with pytest.raises(CanonicalJsonError, match="timezone"):
        normalize_utc_timestamp("2026-01-02T00:00:00")
    assert normalize_utc_timestamp("2026-01-02T01:00:00+01:00") == "2026-01-02T00:00:00+00:00"
    assert normalize_utc_timestamp(datetime(2026, 1, 2, tzinfo=timezone.utc)) == "2026-01-02T00:00:00+00:00"


def test_unsupported_python_objects_are_rejected() -> None:
    with pytest.raises(CanonicalJsonError, match="unsupported type"):
        canonical_json_bytes({"payload": object()})


def test_copied_experiment_remains_verifiable_without_location_binding(tmp_path: Path) -> None:
    context = run_pipeline_to_bound(tmp_path / "source")
    copied = tmp_path / "copied"
    shutil.copytree(context["experiment_dir"], copied)
    assert et.verify_experiment(context["experiment_dir"],
                                registry_root=context["registry_root"])["ok"] is True
    assert et.verify_experiment(copied, registry_root=context["registry_root"])["ok"] is True
    assert et.show_state(context["experiment_dir"])["artifacts"] == et.show_state(copied)["artifacts"]


def test_unexpected_json_artifact_is_reported_by_verifier(tmp_path: Path) -> None:
    context = run_pipeline_to_bound(tmp_path)
    current = et._current_dir(context["experiment_dir"])
    (current / "unexpected.json").write_text("{}\n", encoding="utf-8")
    result = et.verify_experiment(context["experiment_dir"])
    assert result["ok"] is False
    assert any("unexpected" in problem for problem in result["problems"])


def test_symlink_escape_is_never_a_valid_logical_path(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows account")
    assert link.resolve() == outside.resolve()
    with pytest.raises(PathSecurityError):
        normalize_logical_path("link/../outside")


def test_evidence_command_is_data_and_never_executed(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    payload = {"execution_command": f"python -c create {marker}"}
    canonical_json_bytes(payload)
    assert marker.exists() is False

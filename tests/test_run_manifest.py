from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from engine.core import run_manifest as rm


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "user.email=test@example.com", "-c", "user.name=test", *args],
        cwd=root, capture_output=True, text=True, check=True,
    )


def _init_repo(root: Path) -> None:
    _git(root, "init", "-q")
    (root / "source.py").write_text("x = 1\n", encoding="utf-8")
    _git(root, "add", "source.py")
    _git(root, "commit", "-q", "-m", "initial")


def test_git_info_reports_clean_then_dirty_then_unavailable(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    commit, status = rm.git_info(repo)
    assert status == "clean"
    assert commit and len(commit) == 40

    (repo / "source.py").write_text("x = 2\n", encoding="utf-8")
    commit2, status2 = rm.git_info(repo)
    assert status2 == "dirty"
    assert commit2 == commit  # HEAD hasn't moved, only the worktree changed

    no_git = tmp_path / "no_git"
    no_git.mkdir()
    commit3, status3 = rm.git_info(no_git)
    assert commit3 is None
    assert status3 == "unavailable"


def _clean_repo_with_source(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    return repo, repo / "source.py"


def test_promotion_eligible_when_every_gate_is_satisfied(tmp_path: Path) -> None:
    repo, source = _clean_repo_with_source(tmp_path)
    manifest = rm.build_run_manifest(
        root=repo, frozen_contract_version="v1", required_tests_passed=True,
        holdout_status="clean_holdout", source_files=[source],
    )
    assert manifest["promotion_eligible"] is True
    assert manifest["promotion_blockers"] == []
    assert manifest["git_status"] == "clean"
    assert manifest["dirty_worktree"] is False


def test_promotion_blocked_by_dirty_worktree(tmp_path: Path) -> None:
    repo, source = _clean_repo_with_source(tmp_path)
    source.write_text("x = 999\n", encoding="utf-8")
    manifest = rm.build_run_manifest(
        root=repo, frozen_contract_version="v1", required_tests_passed=True,
        holdout_status="not_used", source_files=[source],
    )
    assert manifest["promotion_eligible"] is False
    assert any("not clean" in blocker for blocker in manifest["promotion_blockers"])


def test_promotion_blocked_by_burned_holdout_acknowledgement(tmp_path: Path) -> None:
    repo, source = _clean_repo_with_source(tmp_path)
    manifest = rm.build_run_manifest(
        root=repo, frozen_contract_version="v1", required_tests_passed=True,
        holdout_status="burned_acknowledged", source_files=[source],
    )
    assert manifest["promotion_eligible"] is False
    assert any("burned-holdout" in blocker for blocker in manifest["promotion_blockers"])


def test_promotion_blocked_by_missing_source_hashes(tmp_path: Path) -> None:
    repo, _source = _clean_repo_with_source(tmp_path)
    manifest = rm.build_run_manifest(
        root=repo, frozen_contract_version="v1", required_tests_passed=True,
        holdout_status="not_used", source_files=None,
    )
    assert manifest["promotion_eligible"] is False
    assert any("no source file hashes" in blocker for blocker in manifest["promotion_blockers"])


def test_promotion_blocked_when_tests_not_recorded_passed(tmp_path: Path) -> None:
    repo, source = _clean_repo_with_source(tmp_path)
    manifest = rm.build_run_manifest(
        root=repo, frozen_contract_version="v1", required_tests_passed=False,
        holdout_status="not_used", source_files=[source],
    )
    assert manifest["promotion_eligible"] is False
    assert any("tests were not recorded" in blocker for blocker in manifest["promotion_blockers"])


def test_git_unavailable_blocks_promotion_even_with_every_other_gate_satisfied(tmp_path: Path) -> None:
    no_git = tmp_path / "no_git"
    no_git.mkdir()
    source = no_git / "source.py"
    source.write_text("x = 1\n", encoding="utf-8")
    manifest = rm.build_run_manifest(
        root=no_git, frozen_contract_version="v1", required_tests_passed=True,
        holdout_status="clean_holdout", source_files=[source],
    )
    assert manifest["git_status"] == "unavailable"
    assert manifest["dirty_worktree"] is None
    assert manifest["promotion_eligible"] is False


def test_unsupported_holdout_status_raises(tmp_path: Path) -> None:
    repo, source = _clean_repo_with_source(tmp_path)
    with pytest.raises(rm.RunManifestError):
        rm.build_run_manifest(root=repo, frozen_contract_version="v1", required_tests_passed=True,
                              holdout_status="whatever", source_files=[source])


def test_missing_recorded_path_raises_instead_of_silently_skipping(tmp_path: Path) -> None:
    repo, _source = _clean_repo_with_source(tmp_path)
    with pytest.raises(rm.RunManifestError):
        rm.build_run_manifest(root=repo, frozen_contract_version="v1", required_tests_passed=True,
                              holdout_status="not_used", source_files=[repo / "does_not_exist.py"])


def test_manifest_self_verifies(tmp_path: Path) -> None:
    repo, source = _clean_repo_with_source(tmp_path)
    manifest = rm.build_run_manifest(root=repo, frozen_contract_version="v1", required_tests_passed=True,
                                     holdout_status="not_used", source_files=[source])
    assert rm.verify_manifest_integrity(manifest) is True


def test_tampering_with_any_field_is_detected(tmp_path: Path) -> None:
    repo, source = _clean_repo_with_source(tmp_path)
    manifest = rm.build_run_manifest(root=repo, frozen_contract_version="v1", required_tests_passed=False,
                                     holdout_status="not_used", source_files=[source])
    assert manifest["promotion_eligible"] is False  # tests not passed -> genuinely blocked
    tampered = dict(manifest)
    tampered["promotion_eligible"] = True  # attacker flips the verdict without redoing the hash
    tampered["promotion_blockers"] = []
    assert rm.verify_manifest_integrity(tampered) is False


def test_missing_hash_field_fails_verification() -> None:
    assert rm.verify_manifest_integrity({"run_id": "x"}) is False


def test_write_manifest_refuses_a_tampered_payload(tmp_path: Path) -> None:
    repo, source = _clean_repo_with_source(tmp_path)
    manifest = rm.build_run_manifest(root=repo, frozen_contract_version="v1", required_tests_passed=True,
                                     holdout_status="not_used", source_files=[source])
    manifest["run_id"] = "tampered"
    with pytest.raises(rm.RunManifestError):
        rm.write_manifest(manifest, tmp_path / "out" / "manifest.json")


def test_write_and_read_manifest_roundtrip(tmp_path: Path) -> None:
    repo, source = _clean_repo_with_source(tmp_path)
    manifest = rm.build_run_manifest(root=repo, frozen_contract_version="v1", required_tests_passed=True,
                                     holdout_status="clean_holdout", source_files=[source])
    out_path = rm.write_manifest(manifest, tmp_path / "out" / "manifest.json")
    reloaded = rm.read_manifest(out_path)
    assert reloaded == manifest
    assert rm.verify_manifest_integrity(reloaded) is True


def test_configuration_sha256_is_deterministic_and_order_independent(tmp_path: Path) -> None:
    repo, source = _clean_repo_with_source(tmp_path)
    manifest_a = rm.build_run_manifest(root=repo, frozen_contract_version="v1", required_tests_passed=True,
                                       holdout_status="not_used", source_files=[source],
                                       configuration={"a": 1, "b": 2})
    manifest_b = rm.build_run_manifest(root=repo, frozen_contract_version="v1", required_tests_passed=True,
                                       holdout_status="not_used", source_files=[source],
                                       configuration={"b": 2, "a": 1})
    manifest_c = rm.build_run_manifest(root=repo, frozen_contract_version="v1", required_tests_passed=True,
                                       holdout_status="not_used", source_files=[source],
                                       configuration={"a": 1, "b": 3})
    assert manifest_a["configuration_sha256"] == manifest_b["configuration_sha256"]
    assert manifest_a["configuration_sha256"] != manifest_c["configuration_sha256"]

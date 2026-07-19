from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from engine.experiments import holdout_registry as hr
from engine.experiments.errors import HoldoutReuseError, LockError, TribunalError

FINGERPRINT = "a" * 64
TARGET = "b" * 64
PLAN_HASH = "c" * 64
START = "2024-07-01T00:00:00+00:00"
END = "2024-12-31T00:00:00+00:00"


def claim(root: Path, *, experiment="00000000-0000-4000-8000-000000000001",
          fingerprint=FINGERPRINT, start=START, end=END, family="family-a",
          model="model-v1", forensic=False, holdout_id=None):
    return hr.claim_holdout(
        root, dataset_fingerprint=fingerprint, universe=["EURUSD", "GBPUSD"],
        interval_start_utc=start, interval_end_utc=end,
        target_contract_sha256=TARGET, model_family=model,
        hypothesis_family=family, experiment_id=experiment, plan_sha256=PLAN_HASH,
        claimed_at_utc="2026-01-02T00:00:00+00:00",
        declare_forensic_reuse=forensic, holdout_id=holdout_id)


def test_first_claim_succeeds(tmp_path: Path) -> None:
    result = claim(tmp_path)
    assert result["status"] == hr.SEALED_FOR_EXPERIMENT
    assert result["reuse_kind"] == "none"
    assert result["promotion_blocked"] is False
    assert hr.holdout_status(tmp_path, result["holdout_id"]) == hr.SEALED_FOR_EXPERIMENT


def test_exact_second_claim_is_detected(tmp_path: Path) -> None:
    claim(tmp_path)
    with pytest.raises(HoldoutReuseError, match="not untouched"):
        claim(tmp_path, experiment="00000000-0000-4000-8000-000000000002")


def test_partial_overlap_is_detected(tmp_path: Path) -> None:
    claim(tmp_path)
    with pytest.raises(HoldoutReuseError, match="partial"):
        claim(tmp_path, experiment="00000000-0000-4000-8000-000000000002",
              start="2024-10-01T00:00:00+00:00", end="2025-03-01T00:00:00+00:00")


def test_disjoint_interval_is_a_fresh_claim(tmp_path: Path) -> None:
    claim(tmp_path)
    result = claim(tmp_path, experiment="00000000-0000-4000-8000-000000000002",
                   start="2025-01-01T00:00:00+00:00", end="2025-06-01T00:00:00+00:00")
    assert result["reuse_kind"] == "none"


def test_renaming_or_changing_family_does_not_evade_reuse(tmp_path: Path) -> None:
    claim(tmp_path)
    with pytest.raises(HoldoutReuseError):
        claim(tmp_path, experiment="00000000-0000-4000-8000-000000000003",
              family="totally-different-family")


def test_new_model_commit_does_not_make_data_untouched(tmp_path: Path) -> None:
    claim(tmp_path)
    with pytest.raises(HoldoutReuseError):
        claim(tmp_path, experiment="00000000-0000-4000-8000-000000000004",
              model="model-v2-completely-new")


def test_forensic_reuse_is_allowed_but_promotion_blocked(tmp_path: Path) -> None:
    claim(tmp_path)
    result = claim(tmp_path, experiment="00000000-0000-4000-8000-000000000005",
                   forensic=True)
    assert result["status"] == hr.FORENSIC_REUSE
    assert result["reuse_kind"] == "exact"
    assert result["promotion_blocked"] is True


def test_consume_and_reuse_after_consumption(tmp_path: Path) -> None:
    result = claim(tmp_path)
    hr.consume_holdout(tmp_path, result["holdout_id"])
    assert hr.holdout_status(tmp_path, result["holdout_id"]) == hr.CONSUMED
    with pytest.raises(HoldoutReuseError):
        claim(tmp_path, experiment="00000000-0000-4000-8000-000000000006")
    forensic = claim(tmp_path, experiment="00000000-0000-4000-8000-000000000006",
                     forensic=True)
    assert forensic["promotion_blocked"] is True


def test_invalidated_holdout_blocks_clean_claim(tmp_path: Path) -> None:
    result = claim(tmp_path)
    hr.invalidate_holdout(tmp_path, result["holdout_id"])
    assert hr.holdout_status(tmp_path, result["holdout_id"]) == hr.INVALIDATED
    with pytest.raises(HoldoutReuseError):
        claim(tmp_path, experiment="00000000-0000-4000-8000-000000000007")


def test_registry_tampering_is_detected(tmp_path: Path) -> None:
    claim(tmp_path)
    registry_path = tmp_path / hr.REGISTRY_FILENAME
    document = json.loads(registry_path.read_text(encoding="utf-8"))
    document["entries"][0]["status"] = hr.UNUSED  # rewrite history
    registry_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(TribunalError, match="tampered"):
        hr.load_registry(tmp_path)
    with pytest.raises(TribunalError, match="tampered"):
        claim(tmp_path, experiment="00000000-0000-4000-8000-000000000008")


def test_registry_revision_tampering_is_detected(tmp_path: Path) -> None:
    claim(tmp_path)
    registry_path = tmp_path / hr.REGISTRY_FILENAME
    document = json.loads(registry_path.read_text(encoding="utf-8"))
    document["revision"] += 1
    registry_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(TribunalError, match="integrity hash mismatch"):
        hr.load_registry(tmp_path)


def test_concurrent_claims_permit_only_one_clean_claimant(tmp_path: Path) -> None:
    outcomes: list[str] = []
    lock = threading.Lock()

    def attempt(experiment_id: str) -> None:
        try:
            claim(tmp_path, experiment=experiment_id, holdout_id=None)
            with lock:
                outcomes.append("claimed")
        except HoldoutReuseError:
            with lock:
                outcomes.append("reuse-rejected")

    threads = [threading.Thread(target=attempt,
                                args=(f"00000000-0000-4000-8000-0000000000{index:02x}",))
               for index in range(1, 6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count("claimed") == 1
    assert outcomes.count("reuse-rejected") == len(threads) - 1


def test_stale_lock_is_broken_safely(tmp_path: Path) -> None:
    lock_path = tmp_path / hr.LOCK_FILENAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("pid=0\n", encoding="utf-8")
    old = time.time() - 3600
    import os
    os.utime(lock_path, (old, old))
    result = claim(tmp_path)  # default stale threshold is 600s, lock is 1h old
    assert result["status"] == hr.SEALED_FOR_EXPERIMENT


def test_fresh_lock_blocks_with_lock_error(tmp_path: Path) -> None:
    lock_path = tmp_path / hr.LOCK_FILENAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("pid=0\n", encoding="utf-8")
    with pytest.raises(LockError):
        hr.claim_holdout(
            tmp_path, dataset_fingerprint=FINGERPRINT, universe=["EURUSD"],
            interval_start_utc=START, interval_end_utc=END,
            target_contract_sha256=TARGET, model_family="m",
            hypothesis_family="f", experiment_id="00000000-0000-4000-8000-000000000001",
            plan_sha256=PLAN_HASH, claimed_at_utc="2026-01-02T00:00:00+00:00",
            lock_timeout_seconds=0.2)


def test_unused_registration_then_exact_claim_binds_it(tmp_path: Path) -> None:
    # Manually seed an UNUSED entry, then claim it exactly.
    document = hr.load_registry(tmp_path)
    document["entries"].append({
        "holdout_id": "11111111-1111-4111-8111-111111111111",
        "dataset_fingerprint": FINGERPRINT, "universe": ["EURUSD", "GBPUSD"],
        "interval_start_utc": START, "interval_end_utc": END,
        "target_contract_sha256": TARGET, "model_family": "model-v1",
        "hypothesis_family": "family-a", "status": hr.UNUSED,
        "experiment_id": None, "plan_sha256": None, "claimed_at_utc": None})
    hr._save_registry(tmp_path, document)
    result = claim(tmp_path)
    assert result["holdout_id"] == "11111111-1111-4111-8111-111111111111"
    assert result["status"] == hr.SEALED_FOR_EXPERIMENT


def test_release_restores_pre_registered_unused_entry(tmp_path: Path) -> None:
    document = hr.load_registry(tmp_path)
    holdout_id = "11111111-1111-4111-8111-111111111111"
    document["entries"].append({
        "holdout_id": holdout_id, "dataset_fingerprint": FINGERPRINT,
        "universe": ["EURUSD", "GBPUSD"], "interval_start_utc": START,
        "interval_end_utc": END, "target_contract_sha256": TARGET,
        "model_family": "model-v1", "hypothesis_family": "family-a",
        "status": hr.UNUSED, "experiment_id": None, "plan_sha256": None,
        "claimed_at_utc": None})
    hr._save_registry(tmp_path, document)
    result = claim(tmp_path)
    assert hr.release_claim_if_owned(tmp_path, holdout_id=result["holdout_id"],
                                     experiment_id="00000000-0000-4000-8000-000000000001",
                                     plan_sha256=PLAN_HASH)
    restored = hr.load_registry(tmp_path)["entries"][0]
    assert restored["status"] == hr.UNUSED
    assert restored["experiment_id"] is None


def test_many_registered_holdouts_overlap_query(tmp_path: Path) -> None:
    # 100 disjoint holdouts on distinct fingerprints; overlap detection stays exact.
    for index in range(100):
        fingerprint = f"{index:064x}"
        claim(tmp_path, fingerprint=fingerprint,
              experiment=f"00000000-0000-4000-8000-{index:012x}")
    document = hr.load_registry(tmp_path)
    assert len(document["entries"]) == 100
    overlaps = hr.find_overlaps(document, dataset_fingerprint=f"{42:064x}",
                                interval_start_utc=START, interval_end_utc=END)
    assert len(overlaps) == 1 and overlaps[0]["overlap_kind"] == "exact"


def test_deterministic_holdout_id(tmp_path: Path) -> None:
    first = claim(tmp_path)
    second = claim(tmp_path / "other-registry")
    assert first["holdout_id"] == second["holdout_id"]

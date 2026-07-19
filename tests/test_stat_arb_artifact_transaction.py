from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from engine.core.run_manifest import verify_manifest_integrity
from engine.models.statistical import stat_arb


FAST_CONFIG = stat_arb.ArenaConfig(
    warmup_steps=64,
    bootstrap_samples=8,
    bootstrap_block_sensitivity_minutes=(8, 16, 32),
    basket_mode=stat_arb.RELATIVE_VALUE_MODE,
    relative_value_currency_exposure_budget=0.35,
    basket_neutral_zone_z=0.05,
    low_regime=stat_arb.RegimeParameters(32.0, 32.0, 8, 0.94, 0.50, 8, 2.0, 2),
    high_regime=stat_arb.RegimeParameters(16.0, 16.0, 4, 0.82, 0.75, 6, 1.35, 1),
)


def _result() -> stat_arb.ArenaResult:
    times, prices = stat_arb.synthetic_input()
    return stat_arb.run_arrays(times, prices, test_start_index=500, config=FAST_CONFIG)


def _patch_canonical_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    for pair in stat_arb.PAIRS:
        (canonical / f"{pair}.parquet").write_bytes(pair.encode("ascii"))
    monkeypatch.setattr(stat_arb, "CANONICAL_DIR", canonical)


def test_successful_publication_is_one_complete_directory_and_returned_summary_matches(tmp_path: Path,
                                                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_canonical_inputs(monkeypatch, tmp_path)
    result = _result()
    paths = stat_arb.write_result(result, tmp_path / "out", "synthetic", run_id="00000000-0000-4000-8000-000000000001")
    run_dir = Path(paths["run_directory"])
    assert run_dir.is_dir()
    assert {path.name for path in run_dir.iterdir()} == {
        "minute.parquet", "graph.parquet", "daily.parquet", "summary.json", "manifest.json",
    }
    persisted = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
    assert persisted == result.summary
    manifest = json.loads(Path(paths["manifest"]).read_text(encoding="utf-8"))
    assert verify_manifest_integrity(manifest)
    assert not manifest["required_tests_passed"]
    assert not manifest["promotion_eligible"]


def test_invalid_summary_and_nonfinite_json_leave_no_final_directory(tmp_path: Path,
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_canonical_inputs(monkeypatch, tmp_path)
    with pytest.raises(stat_arb.ContractError):
        stat_arb._strict_json_text({"bad": np.inf})
    result = _result()
    result.summary["interpretation"] = float("nan")
    with pytest.raises(stat_arb.ContractError):
        stat_arb.write_result(result, tmp_path / "out", "synthetic", run_id="00000000-0000-4000-8000-000000000002")
    runs_root = tmp_path / "out" / "stat_arb_v0_2_1_runs"
    assert not (runs_root / "00000000-0000-4000-8000-000000000002").exists()
    assert not list(runs_root.glob(".00000000-0000-4000-8000-000000000002.staging-*"))


def test_validation_manifest_and_parquet_failures_do_not_publish(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_canonical_inputs(monkeypatch, tmp_path)
    out = tmp_path / "out"
    original_validate = stat_arb._validate_frame_rows

    def reject_emission(frame, schema_name):
        if schema_name == "stat-arb-emission.schema.json":
            raise stat_arb.ContractError("synthetic invalid emission")
        return original_validate(frame, schema_name)

    monkeypatch.setattr(stat_arb, "_validate_frame_rows", reject_emission)
    with pytest.raises(stat_arb.ContractError):
        stat_arb.write_result(_result(), out, "synthetic", run_id="00000000-0000-4000-8000-000000000003")
    assert not (out / "stat_arb_v0_2_1_runs" / "00000000-0000-4000-8000-000000000003").exists()

    monkeypatch.setattr(stat_arb, "_validate_frame_rows", original_validate)
    monkeypatch.setattr(stat_arb, "verify_manifest_integrity", lambda _manifest: False)
    with pytest.raises(stat_arb.ContractError):
        stat_arb.write_result(_result(), out, "synthetic", run_id="00000000-0000-4000-8000-000000000004")
    assert not (out / "stat_arb_v0_2_1_runs" / "00000000-0000-4000-8000-000000000004").exists()

    monkeypatch.setattr(stat_arb, "verify_manifest_integrity", verify_manifest_integrity)
    original_to_parquet = __import__("pandas").DataFrame.to_parquet

    def fail_parquet(self, path, *args, **kwargs):
        if str(path).endswith("minute.parquet"):
            raise OSError("synthetic parquet write failure")
        return original_to_parquet(self, path, *args, **kwargs)

    monkeypatch.setattr(__import__("pandas").DataFrame, "to_parquet", fail_parquet)
    with pytest.raises(OSError):
        stat_arb.write_result(_result(), out, "synthetic", run_id="00000000-0000-4000-8000-000000000005")
    assert not (out / "stat_arb_v0_2_1_runs" / "00000000-0000-4000-8000-000000000005").exists()


def test_preexisting_final_run_directory_is_never_overwritten(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_canonical_inputs(monkeypatch, tmp_path)
    run_id = "00000000-0000-4000-8000-000000000006"
    final = tmp_path / "out" / "stat_arb_v0_2_1_runs" / run_id
    final.mkdir(parents=True)
    marker = final / "marker.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(stat_arb.ContractError):
        stat_arb.write_result(_result(), tmp_path / "out", "synthetic", run_id=run_id)
    assert marker.read_text(encoding="utf-8") == "preserve"

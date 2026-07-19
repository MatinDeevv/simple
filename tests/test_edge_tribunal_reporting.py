from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from engine.experiments import edge_tribunal as et
from engine.experiments import reporting
from engine.experiments.canonical import load_strict_json_text, sha256_payload
from engine.experiments.errors import VerdictIntegrityError
from test_edge_tribunal_evidence import T_EVAL, T_RECORD, finalize_evidence, make_evidence, run_pipeline_to_bound


def run_to_verdict(root: Path) -> dict:
    context = run_pipeline_to_bound(root)
    payload = make_evidence(context["plan"], context["seal"], context["binding"])
    finalize_evidence(payload)
    et.record_evidence(context["experiment_dir"], payload, timestamp_utc=T_RECORD)
    return et.evaluate(context["experiment_dir"], registry_root=context["registry_root"], timestamp_utc=T_EVAL)


def experiment(root: Path) -> Path:
    return root / "experiment"


def artifact(root: Path, name: str) -> dict:
    return load_strict_json_text((et._current_dir(experiment(root)) / name).read_text(encoding="utf-8"))


def test_evaluation_publishes_agreeing_machine_and_human_reports(tmp_path: Path) -> None:
    verdict = run_to_verdict(tmp_path)
    current = et._current_dir(experiment(tmp_path))
    assert artifact(tmp_path, "verdict.json") == verdict
    assert artifact(tmp_path, "scorecard.json")["verdict"] == verdict["verdict"]
    assert artifact(tmp_path, "verification.json")["verdict_sha256"] == verdict["verdict_sha256"]
    report = (current / "report.md").read_text(encoding="utf-8")
    assert f"## VERDICT: {verdict['verdict']}" in report


def test_report_is_deterministic_for_identical_inputs(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    assert run_to_verdict(first) == run_to_verdict(second)
    for name in ("verdict.json", "scorecard.json", "verification.json", "report.md"):
        assert (et._current_dir(experiment(first)) / name).read_bytes() == (et._current_dir(experiment(second)) / name).read_bytes()


def test_report_leads_with_failures_and_no_trading_authorization(tmp_path: Path) -> None:
    run_to_verdict(tmp_path)
    report = (et._current_dir(experiment(tmp_path)) / "report.md").read_text(encoding="utf-8")
    required = (
        "## VERDICT:", "VALIDITY STATUS", "PRIMARY GATES", "HARD BLOCKERS",
        "MAXIMUM NEXT STAGE", "## WHY THIS IS NOT A TRADING AUTHORIZATION",
        "## 1. Failed gates", "## 2. Insufficient gates",
        "## 15. Audit-chain verification and artifact hashes",
    )
    for heading in required:
        assert heading in report
    assert report.index("## 1. Failed gates") < report.index("## 13. Hard validity gates")
    assert "does not authorize order placement" in report


def test_scorecard_and_verification_are_strict_finite_json(tmp_path: Path) -> None:
    run_to_verdict(tmp_path)
    for name in ("verdict.json", "scorecard.json", "verification.json"):
        raw = (et._current_dir(experiment(tmp_path)) / name).read_text(encoding="utf-8")
        def reject_constant(value: str) -> None:
            raise AssertionError(f"non-finite JSON constant: {value}")
        assert json.loads(raw, parse_constant=reject_constant)


def test_reporting_rejects_tampered_verdict(tmp_path: Path) -> None:
    verdict = run_to_verdict(tmp_path)
    tampered = dict(verdict, verdict_sha256="0" * 64)
    with pytest.raises(VerdictIntegrityError):
        reporting.build_scorecard(tampered)
    with pytest.raises(VerdictIntegrityError):
        reporting.build_report_markdown(
            plan=artifact(tmp_path, "plan.json"), seal=artifact(tmp_path, "seal.json"),
            binding=artifact(tmp_path, "dataset-binding.json"), verdict_payload=tampered,
            verification=artifact(tmp_path, "verification.json"))


def test_verify_detects_tampered_evidence_and_is_read_only(tmp_path: Path) -> None:
    run_to_verdict(tmp_path)
    evidence_path = et._current_dir(experiment(tmp_path)) / "evidence.json"
    evidence = load_strict_json_text(evidence_path.read_text(encoding="utf-8"))
    evidence["producer"]["execution_command"] = "tampered"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = et.verify_experiment(experiment(tmp_path))
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert result["ok"] is False
    assert before == after


def test_report_cli_refuses_before_verdict(tmp_path: Path) -> None:
    context = run_pipeline_to_bound(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "engine.experiments.edge_tribunal", "report",
         "--experiment-dir", str(context["experiment_dir"])],
        text=True, capture_output=True, check=False)
    assert result.returncode == 1
    assert "no verdict has been issued" in result.stderr
    assert "Traceback" not in result.stderr


def test_report_hashes_are_bound_to_verdict(tmp_path: Path) -> None:
    verdict = run_to_verdict(tmp_path)
    scorecard = artifact(tmp_path, "scorecard.json")
    verification = artifact(tmp_path, "verification.json")
    assert scorecard["verdict_sha256"] == verdict["verdict_sha256"]
    assert verification["verdict_sha256"] == verdict["verdict_sha256"]
    assert scorecard["scorecard_sha256"] == sha256_payload({k: v for k, v in scorecard.items() if k != "scorecard_sha256"})

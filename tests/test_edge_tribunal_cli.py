from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from engine.experiments import edge_tribunal as et
from engine.experiments.canonical import strict_json_text
from test_edge_tribunal_evidence import T_EVAL, T_RECORD, finalize_evidence, make_evidence, run_pipeline_to_bound


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "engine.experiments.edge_tribunal", *args],
        cwd=cwd, env=env, text=True, capture_output=True, check=False)


def snapshot(root: Path) -> dict[Path, bytes]:
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_cli_help_lists_every_command() -> None:
    result = run_cli("--help")
    assert result.returncode == 0
    for name in ("init", "seal", "bind-data", "record-evidence", "evaluate", "verify", "report", "show-state", "archive"):
        assert name in result.stdout
        child = run_cli(name, "--help")
        assert child.returncode == 0
        assert "usage:" in child.stdout.lower()


def test_cli_ordinary_error_is_concise_and_debug_preserves_traceback(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    normal = run_cli("seal", "--experiment-dir", str(missing))
    debug = run_cli("--debug", "seal", "--experiment-dir", str(missing))
    assert normal.returncode == 1 and "error:" in normal.stderr and "Traceback" not in normal.stderr
    assert debug.returncode != 0 and "Traceback" in debug.stderr


def test_cli_invalid_sequence_and_report_before_verdict_fail(tmp_path: Path) -> None:
    seal = run_cli("seal", "--experiment-dir", str(tmp_path))
    assert seal.returncode == 1
    context = run_pipeline_to_bound(tmp_path / "bound")
    report = run_cli("report", "--experiment-dir", str(context["experiment_dir"]))
    assert report.returncode == 1 and "no verdict has been issued" in report.stderr


def test_cli_verify_and_show_state_are_byte_for_byte_read_only(tmp_path: Path) -> None:
    context = run_pipeline_to_bound(tmp_path)
    before = snapshot(context["experiment_dir"])
    verify = run_cli("verify", "--experiment-dir", str(context["experiment_dir"]),
                     "--registry-root", str(context["registry_root"]))
    show = run_cli("show-state", "--experiment-dir", str(context["experiment_dir"]))
    after = snapshot(context["experiment_dir"])
    assert verify.returncode == 0 and json.loads(verify.stdout)["ok"] is True
    assert show.returncode == 0 and json.loads(show.stdout)["state"] == "DATA_BOUND"
    assert before == after


def test_cli_has_no_repository_root_cwd_assumption(tmp_path: Path) -> None:
    context = run_pipeline_to_bound(tmp_path / "experiment")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    result = run_cli("show-state", "--experiment-dir", str(context["experiment_dir"]), cwd=elsewhere)
    assert result.returncode == 0
    assert json.loads(result.stdout)["state"] == "DATA_BOUND"


def test_cli_record_evaluate_report_archive_workflow_uses_only_synthetic_tmp_path(tmp_path: Path) -> None:
    context = run_pipeline_to_bound(tmp_path)
    evidence = make_evidence(context["plan"], context["seal"], context["binding"])
    finalize_evidence(evidence)
    evidence_path = tmp_path / "synthetic-evidence.json"
    evidence_path.write_text(strict_json_text(evidence), encoding="utf-8")
    record = run_cli("record-evidence", "--experiment-dir", str(context["experiment_dir"]),
                     "--evidence", str(evidence_path), "--timestamp", T_RECORD)
    evaluate = run_cli("evaluate", "--experiment-dir", str(context["experiment_dir"]),
                       "--registry-root", str(context["registry_root"]), "--timestamp", T_EVAL)
    report = run_cli("report", "--experiment-dir", str(context["experiment_dir"]))
    archive = run_cli("archive", "--experiment-dir", str(context["experiment_dir"]),
                      "--timestamp", "2026-01-02T00:30:00+00:00")
    assert record.returncode == evaluate.returncode == report.returncode == archive.returncode == 0
    assert "VERDICT:" in evaluate.stdout
    assert "WHY THIS IS NOT A TRADING AUTHORIZATION" in report.stdout
    assert et.current_state(context["experiment_dir"]) == "ARCHIVED"


def test_four_synthetic_examples_complete_with_exact_verdicts(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples/edge-tribunal/run_synthetic_examples.py"),
         "--output-root", str(tmp_path / "outcomes")],
        cwd=tmp_path, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["synthetic_only"] is True
    assert [item["verdict"] for item in payload["results"]] == [
        "INVALID_EXPERIMENT", "REJECTED", "INCONCLUSIVE", "RESEARCH_ONLY"]
    assert all(item["trading_authorization"] is False for item in payload["results"])

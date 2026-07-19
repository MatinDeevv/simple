from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from engine.experiments import edge_tribunal as et
from engine.experiments import state_machine as sm
from engine.experiments.errors import (
    InvalidStateTransitionError,
    PlanMutationError,
    TribunalError,
    VerdictIntegrityError,
)

from test_edge_tribunal_evidence import (
    T_EVAL,
    T_RECORD,
    T_SEAL,
    finalize_evidence,
    make_evidence,
    run_pipeline_to_bound,
)
from test_edge_tribunal_preregistration import make_plan


def test_every_legal_transition_succeeds() -> None:
    for state_before, state_after in sm.ALLOWED_TRANSITIONS.items():
        sm.validate_transition(state_before, state_after)


def test_every_skipped_transition_fails() -> None:
    for index, state_before in enumerate(sm.STATES):
        for state_after in sm.STATES[index + 2:]:
            with pytest.raises(InvalidStateTransitionError):
                sm.validate_transition(state_before, state_after)


def test_every_backward_transition_fails() -> None:
    for index, state_before in enumerate(sm.STATES):
        for state_after in sm.STATES[:index]:
            with pytest.raises(InvalidStateTransitionError):
                sm.validate_transition(state_before, state_after)


def test_repeated_transition_fails() -> None:
    for state in sm.STATES:
        with pytest.raises(InvalidStateTransitionError, match="repeated"):
            sm.validate_transition(state, state)


def test_unknown_state_fails() -> None:
    with pytest.raises(InvalidStateTransitionError):
        sm.validate_transition("DRAFT", "LIVE_READY")


def test_pipeline_walks_every_state(tmp_path: Path) -> None:
    context = run_pipeline_to_bound(tmp_path)
    experiment_dir = context["experiment_dir"]
    assert et.current_state(experiment_dir) == sm.DATA_BOUND
    evidence = finalize_evidence(
        make_evidence(context["plan"], context["seal"], context["binding"]))
    et.record_evidence(experiment_dir, evidence, timestamp_utc=T_RECORD)
    assert et.current_state(experiment_dir) == sm.EVIDENCE_RECORDED
    et.evaluate(experiment_dir, registry_root=context["registry_root"],
                timestamp_utc=T_EVAL)
    assert et.current_state(experiment_dir) == sm.VERDICT_ISSUED
    et.archive(experiment_dir, timestamp_utc="2026-01-02T00:25:00+00:00")
    assert et.current_state(experiment_dir) == sm.ARCHIVED


def test_seal_twice_fails(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"
    et.init_experiment(experiment_dir, make_plan(),
                       timestamp_utc="2026-01-02T00:00:00+00:00")
    et.seal_experiment(experiment_dir, timestamp_utc=T_SEAL)
    with pytest.raises(InvalidStateTransitionError):
        et.seal_experiment(experiment_dir, timestamp_utc="2026-01-02T00:06:00+00:00")


def test_evaluate_before_evidence_fails(tmp_path: Path) -> None:
    context = run_pipeline_to_bound(tmp_path)
    with pytest.raises(InvalidStateTransitionError):
        et.evaluate(context["experiment_dir"], timestamp_utc=T_EVAL)


def test_bind_before_seal_fails(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"
    et.init_experiment(experiment_dir, make_plan(),
                       timestamp_utc="2026-01-02T00:00:00+00:00")
    with pytest.raises(InvalidStateTransitionError):
        et.bind_data(experiment_dir, json.loads(
            (Path(__file__).resolve().parents[1] / "examples" / "edge-tribunal" /
             "synthetic-dataset.json").read_text(encoding="utf-8")),
            registry_root=tmp_path / "registry",
            timestamp_utc="2026-01-02T00:10:00+00:00")


def test_archive_before_verdict_fails(tmp_path: Path) -> None:
    context = run_pipeline_to_bound(tmp_path)
    with pytest.raises(InvalidStateTransitionError):
        et.archive(context["experiment_dir"], timestamp_utc=T_EVAL)


def test_repeated_verdict_issuance_fails(tmp_path: Path) -> None:
    context = run_pipeline_to_bound(tmp_path)
    evidence = finalize_evidence(
        make_evidence(context["plan"], context["seal"], context["binding"]))
    et.record_evidence(context["experiment_dir"], evidence, timestamp_utc=T_RECORD)
    et.evaluate(context["experiment_dir"], timestamp_utc=T_EVAL)
    with pytest.raises(InvalidStateTransitionError):
        et.evaluate(context["experiment_dir"], timestamp_utc="2026-01-02T00:30:00+00:00")


def test_sealed_plan_mutation_is_detected(tmp_path: Path) -> None:
    context = run_pipeline_to_bound(tmp_path, experiment_name="mutation")
    experiment_dir = context["experiment_dir"]
    version_dir = experiment_dir / "versions" / et.current_version_name(experiment_dir)
    plan_path = version_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["acceptance_gates"][0]["threshold"] = -100.0
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    evidence = finalize_evidence(
        make_evidence(context["plan"], context["seal"], context["binding"]))
    with pytest.raises(PlanMutationError):
        et.record_evidence(experiment_dir, evidence, timestamp_utc=T_RECORD)
    result = et.verify_experiment(experiment_dir)
    assert not result["ok"]
    assert any("sealed" in problem or "seal" in problem for problem in result["problems"])


def test_verification_is_read_only(tmp_path: Path) -> None:
    context = run_pipeline_to_bound(tmp_path)
    experiment_dir = context["experiment_dir"]
    before = et.show_state(experiment_dir)
    result = et.verify_experiment(experiment_dir)
    assert result["ok"]
    after = et.show_state(experiment_dir)
    assert before == after


def test_report_before_verdict_fails(tmp_path: Path) -> None:
    context = run_pipeline_to_bound(tmp_path)
    exit_code = et.main(["report", "--experiment-dir", str(context["experiment_dir"])])
    assert exit_code == 1


def test_init_never_overwrites_an_existing_experiment(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"
    et.init_experiment(experiment_dir, make_plan(),
                       timestamp_utc="2026-01-02T00:00:00+00:00")
    with pytest.raises(TribunalError, match="never overwritten"):
        et.init_experiment(experiment_dir, make_plan(),
                           timestamp_utc="2026-01-02T00:01:00+00:00")


def test_artifacts_are_never_replaced_across_versions(tmp_path: Path) -> None:
    context = run_pipeline_to_bound(tmp_path)
    experiment_dir = context["experiment_dir"]
    versions = sorted((experiment_dir / "versions").iterdir())
    assert len(versions) == 3  # init, seal, bind
    plan_bytes = {version.name: (version / "plan.json").read_bytes()
                  for version in versions}
    assert len(set(plan_bytes.values())) == 1  # the plan never changed byte-wise
    # Prior versions still hold their original artifacts (immutable history).
    assert not (versions[0] / "seal.json").exists()
    assert (versions[1] / "seal.json").exists()
    assert (versions[2] / "dataset-binding.json").exists()

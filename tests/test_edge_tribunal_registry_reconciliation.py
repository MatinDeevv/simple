from engine.experiments import holdout_registry as hr
from test_edge_tribunal_holdout_registry import claim
from test_edge_tribunal_evidence import make_dataset_manifest
from test_edge_tribunal_preregistration import make_plan
from test_edge_tribunal_evidence import make_evidence, finalize_evidence, T_RECORD, T_EVAL
from engine.experiments import edge_tribunal as et
from engine.experiments.errors import TribunalError
import pytest


def test_exact_claim_retry_is_idempotent(tmp_path):
    first = claim(tmp_path)
    second = claim(tmp_path)
    assert second["idempotent"] is True
    assert first["holdout_id"] == second["holdout_id"]
    assert len(hr.load_registry(tmp_path)["entries"]) == 1


def test_idempotent_claim_returns_original_token_after_unrelated_revision(tmp_path):
    first = claim(tmp_path)
    claim(tmp_path, fingerprint="d" * 64,
          experiment="00000000-0000-4000-8000-000000000099")
    retry = claim(tmp_path)
    assert retry["claim_token"] == first["claim_token"]
    assert retry["registry_revision"] == first["registry_revision"]


def test_idempotent_consumption(tmp_path):
    item = claim(tmp_path)
    first = hr.consume_holdout_idempotent(tmp_path, item["holdout_id"])
    second = hr.consume_holdout_idempotent(tmp_path, item["holdout_id"])
    assert first["final_status"] == second["final_status"] == hr.CONSUMED


def test_failed_binding_publication_does_not_burn_claim(tmp_path, monkeypatch):
    experiment = tmp_path / "experiment"; registry = tmp_path / "registry"
    et.init_experiment(experiment, make_plan(), timestamp_utc="2026-01-02T00:00:00+00:00")
    et.seal_experiment(experiment, timestamp_utc="2026-01-02T00:05:00+00:00")
    original = et._test_failpoint
    monkeypatch.setattr(et, "_test_failpoint",
                        lambda stage: (_ for _ in ()).throw(OSError("disk"))
                        if stage == "before-rename" else original(stage))
    with pytest.raises(OSError, match="disk"):
        et.bind_data(experiment, make_dataset_manifest(), registry_root=registry,
                     timestamp_utc="2026-01-02T00:10:00+00:00")
    assert hr.load_registry(registry)["entries"] == []


def test_verdict_requires_registry_and_publishes_complete_receipt(tmp_path):
    from test_edge_tribunal_evidence import run_pipeline_to_bound
    context = run_pipeline_to_bound(tmp_path)
    evidence = finalize_evidence(make_evidence(
        context["plan"], context["seal"], context["binding"]))
    et.record_evidence(context["experiment_dir"], evidence, timestamp_utc=T_RECORD)
    with pytest.raises(TribunalError, match="registry_root is mandatory"):
        et.evaluate(context["experiment_dir"], timestamp_utc=T_EVAL)
    et.evaluate(context["experiment_dir"], registry_root=context["registry_root"],
                timestamp_utc=T_EVAL)
    receipt = et.load_artifact(context["experiment_dir"],
                               et.REGISTRY_RECONCILIATION_FILE)
    assert receipt["status"] == "COMPLETE"
    assert hr.holdout_status(context["registry_root"],
                             context["binding"]["holdout_id"]) == hr.CONSUMED
    assert et.verify_experiment(context["experiment_dir"],
                                registry_root=context["registry_root"])["ok"]

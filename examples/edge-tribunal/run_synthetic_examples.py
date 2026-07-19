"""Run the four deterministic, synthetic-only Edge Tribunal outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.experiments import edge_tribunal as tribunal
from engine.experiments.canonical import sha256_payload, strict_json_text
from engine.experiments.robustness import planned_cell_registry

HERE = Path(__file__).resolve().parent
T_INIT = "2026-01-02T00:00:00+00:00"
T_SEAL = "2026-01-02T00:05:00+00:00"
T_BIND = "2026-01-02T00:10:00+00:00"
T_RECORD = "2026-01-02T00:15:00+00:00"
T_EVALUATE = "2026-01-02T00:20:00+00:00"


def load(name: str) -> dict[str, Any]:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def finalize(evidence: dict[str, Any]) -> None:
    body = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    evidence["evidence_sha256"] = sha256_payload(body)


def mutate(evidence: dict[str, Any], mutation: str) -> None:
    if mutation == "none" or mutation == "tamper_configuration_after_recording":
        return
    if mutation == "primary_model_loses":
        model = evidence["primary_model"]
        model.update({
            "brier_improvement": -0.004,
            "brier_improvement_lower_bound_one_sided": -0.006,
            "brier_improvement_lower_bound_two_sided": -0.007,
            "brier_improvement_upper_bound_two_sided": -0.001,
            "log_loss_improvement": -0.008,
            "log_loss_improvement_lower_bound_one_sided": -0.012,
            "log_loss_improvement_lower_bound_two_sided": -0.013,
            "log_loss_improvement_upper_bound_two_sided": -0.002,
        })
    elif mutation == "insufficient_target_clusters":
        evidence["population"]["target_clusters"] = 20
    else:
        raise ValueError(f"unknown synthetic mutation: {mutation}")
    finalize(evidence)


def run_scenario(root: Path, scenario: dict[str, str]) -> dict[str, Any]:
    experiment_dir = root / scenario["name"]
    registry_root = root / f"{scenario['name']}-registry"
    plan = load("synthetic-plan.json")
    manifest = load("synthetic-dataset.json")
    evidence = load("synthetic-evidence.json")
    evidence["robustness"]["cells"] = [
        {"cell_id": cell_id, "dimensions": dimensions, "status": "scored",
         "sample_count": 300, "target_clusters": 120,
         "brier_improvement": 0.003, "log_loss_improvement": 0.006}
        for cell_id, dimensions in planned_cell_registry(
            plan["robustness_contract"]).items()]

    tribunal.init_experiment(experiment_dir, plan, timestamp_utc=T_INIT)
    seal = tribunal.seal_experiment(experiment_dir, timestamp_utc=T_SEAL)
    binding = tribunal.bind_data(
        experiment_dir, manifest, registry_root=registry_root,
        timestamp_utc=T_BIND)
    evidence["experiment"]["binding_sha256"] = binding["binding_sha256"]
    evidence["experiment"]["plan_sha256"] = seal["plan_sha256"]
    evidence["experiment"]["seal_sha256"] = seal["seal_sha256"]
    evidence["dataset"]["dataset_binding_sha256"] = binding["binding_sha256"]
    mutate(evidence, scenario["mutation"])
    finalize(evidence)
    tribunal.record_evidence(
        experiment_dir, evidence, timestamp_utc=T_RECORD)

    if scenario["mutation"] == "tamper_configuration_after_recording":
        evidence_path = tribunal._current_dir(experiment_dir) / tribunal.EVIDENCE_FILE
        recorded = json.loads(evidence_path.read_text(encoding="utf-8"))
        recorded["producer"]["configuration_sha256"] = "f" * 64
        evidence_path.write_text(strict_json_text(recorded), encoding="utf-8")

    verdict = tribunal.evaluate(
        experiment_dir, registry_root=registry_root,
        timestamp_utc=T_EVALUATE)
    if verdict["verdict"] != scenario["expected_verdict"]:
        raise RuntimeError(
            f"{scenario['name']}: expected {scenario['expected_verdict']}, "
            f"received {verdict['verdict']}")
    return {
        "scenario": scenario["name"],
        "verdict": verdict["verdict"],
        "maximum_next_stage": verdict["maximum_next_stage"],
        "trading_authorization": verdict["trading_authorization"],
        "verdict_sha256": verdict["verdict_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    results = [run_scenario(args.output_root, item)
               for item in load("synthetic-scenarios.json")["scenarios"]]
    print(strict_json_text({"synthetic_only": True, "results": results}), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

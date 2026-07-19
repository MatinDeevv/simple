from engine.experiments.robustness import evaluate_robustness_matrix
from engine.experiments.robustness import planned_cell_registry
from engine.experiments.robustness_evidence import recompute_robustness_rows, VERSION
from test_edge_tribunal_preregistration import make_plan
import json


def test_mandatory_explicit_missing_is_failure():
    contract = {"minimum_samples_per_cell": 1, "mandatory_cells": ["must"],
                "insufficient_allowed_cells": [], "missing_cell_policy": "count_as_failed",
                "minimum_pass_proportion": 0.0}
    report = evaluate_robustness_matrix(
        robustness_contract=contract,
        cells=[{"cell_id": "must", "status": "missing"}])
    assert report["mandatory_cell_failures"] == ["must"]


def test_all_2268_planned_cells_are_physically_derived(tmp_path):
    contract = make_plan()["robustness_contract"]
    planned = planned_cell_registry(contract)
    assert len(planned) == 2268
    path = tmp_path / "robustness.jsonl"
    path.write_text("\n".join(json.dumps({"version": VERSION, "kind": "cell",
                                           "cell_id": cell_id, "dimensions": dimensions,
                                           "status": "missing"})
                              for cell_id, dimensions in planned.items()) + "\n",
                    encoding="utf-8")
    result = recompute_robustness_rows(
        path, contract=contract, class_order=["negative", "neutral", "positive"])
    assert len(result) == 2268
    assert all(cell["status"] == "missing" for cell in result)

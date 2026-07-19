from copy import deepcopy
from engine.experiments.dataset_binding import dataset_content_fingerprint
from test_edge_tribunal_evidence import make_dataset_manifest
from test_edge_tribunal_preregistration import make_plan
from engine.experiments import edge_tribunal as et
import hashlib


def test_metadata_and_order_do_not_change_content_identity():
    first = make_dataset_manifest()
    first["files"].append({"logical_path": "data/other", "sha256": "9" * 64,
                           "row_count": 3})
    second = deepcopy(first)
    second["dataset_name"] = "renamed"
    second["provenance"] = "different words"
    second["untouched_evidence"] = "different assertion"
    second["files"].reverse()
    for index, item in enumerate(second["files"]):
        item["logical_path"] = f"renamed/{index}"
    assert dataset_content_fingerprint(first) == dataset_content_fingerprint(second)


def test_byte_hash_change_changes_content_identity():
    first = make_dataset_manifest()
    second = deepcopy(first)
    second["files"][0]["sha256"] = "1" * 64
    assert dataset_content_fingerprint(first) != dataset_content_fingerprint(second)


def test_unverified_caller_row_count_cannot_change_content_identity():
    first = make_dataset_manifest()
    second = deepcopy(first)
    first["files"][0]["row_count"] = 1
    second["files"][0]["row_count"] = 999
    assert dataset_content_fingerprint(first) == dataset_content_fingerprint(second)


def test_bind_data_physically_verifies_bytes(tmp_path):
    data = tmp_path / "dataset.bin"; data.write_bytes(b"synthetic only")
    data2 = tmp_path / "dataset2.bin"; data2.write_bytes(b"synthetic two")
    manifest = make_dataset_manifest()
    manifest["files"][0]["sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    manifest["files"][1]["sha256"] = hashlib.sha256(data2.read_bytes()).hexdigest()
    experiment = tmp_path / "experiment"
    et.init_experiment(experiment, make_plan(), timestamp_utc="2026-01-02T00:00:00+00:00")
    et.seal_experiment(experiment, timestamp_utc="2026-01-02T00:05:00+00:00")
    binding = et.bind_data(
        experiment, manifest, registry_root=tmp_path / "registry",
        timestamp_utc="2026-01-02T00:10:00+00:00", dataset_root=tmp_path,
        physical_file_bindings=[{"physical_path": str(data),
                                 "logical_path": manifest["files"][0]["logical_path"]},
                                {"physical_path": str(data2),
                                 "logical_path": manifest["files"][1]["logical_path"]}])
    assert binding["dataset_bytes_verified"] is True
    assert binding["files"][0]["size_bytes"] == len(b"synthetic only")

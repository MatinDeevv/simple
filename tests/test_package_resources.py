from __future__ import annotations
import tomllib
from pathlib import Path
ROOT=Path(__file__).parents[1]
def test_required_resources_exist_and_package_metadata_declares_them():
    assert all((ROOT/path).is_file() for path in ("engine/config/instruments.json","engine/config/legal-event-schema.json","engine/config/frozen-archives.json"))
    assert len(list((ROOT/"engine/config/schemas").glob("*.schema.json"))) >= 1
    data=tomllib.loads((ROOT/"pyproject.toml").read_text())
    assert data["tool"]["setuptools"]["package-data"]["engine"]
def test_runtime_dependencies_have_exact_core_pins():
    data=tomllib.loads((ROOT/"pyproject.toml").read_text()); pins={line.split("==")[0].lower() for line in (ROOT/"requirements-core.txt").read_text().splitlines() if "==" in line}
    assert {item.split(">=")[0].lower() for item in data["project"]["dependencies"]} <= pins

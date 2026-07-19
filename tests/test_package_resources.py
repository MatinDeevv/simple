from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _build_wheel(destination: Path) -> Path:
    subprocess.run([sys.executable, "-m", "pip", "wheel", ".", "--no-deps",
                    "--wheel-dir", str(destination)], cwd=ROOT, check=True, capture_output=True, text=True)
    wheels = list(destination.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def test_wheel_contains_required_resources_and_no_generated_data(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path)
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert "engine/config/instruments.json" in names
    assert "engine/config/legal-event-schema.json" in names
    schemas = {path.name for path in (ROOT / "engine" / "config" / "schemas").glob("*.json")}
    assert {f"engine/config/schemas/{name}" for name in schemas} <= names
    assert not any(name.startswith(("data/raw/", "data/canonical/", "data/derived/", "artifacts/", ".venv"))
                   or "__pycache__" in name for name in names)

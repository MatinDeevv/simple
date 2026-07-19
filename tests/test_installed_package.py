from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _build_wheel(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "wheel", ".", "--no-deps",
                    "--wheel-dir", str(destination)], cwd=ROOT, check=True, capture_output=True, text=True)
    wheels = list(destination.glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def test_real_wheel_installs_and_self_checks_outside_checkout(tmp_path: Path) -> None:
    wheel = _build_wheel(tmp_path / "wheel")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert len(digest) == 64
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv)], check=True)
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    auractl = venv / ("Scripts/auractl.exe" if os.name == "nt" else "bin/auractl")
    subprocess.run([str(python), "-m", "pip", "install", "--no-deps", str(wheel)], check=True,
                   capture_output=True, text=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    for command in ([str(python), "-c", "import engine; from engine.core.contracts import canonical_pair_order; print(canonical_pair_order(__import__('pathlib').Path(engine.__file__).resolve().parents[1]))"],
                    [str(auractl), "--help"], [str(auractl), "stat-arb", "--self-check"],
                    [str(auractl), "integrator", "--self-check"], [str(auractl), "legal-event", "--self-check"]):
        result = subprocess.run(command, cwd=outside, env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
    unknown = subprocess.run([str(auractl), "not-a-command"], cwd=outside, env=env,
                             capture_output=True, text=True)
    assert unknown.returncode != 0
    assert "invalid choice" in unknown.stderr.lower()

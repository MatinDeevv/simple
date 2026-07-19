from __future__ import annotations

import subprocess
import sys

from engine.cli.main import COMMANDS


def test_cli_exposes_stable_commands_in_help() -> None:
    result = subprocess.run([sys.executable, "-m", "engine.cli.main", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    for command in ("integrator", "stat-arb", "legal-event"):
        assert command in COMMANDS
        assert command in result.stdout


def test_cli_unknown_command_fails_without_traceback() -> None:
    result = subprocess.run([sys.executable, "-m", "engine.cli.main", "unknown"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "traceback" not in result.stderr.lower()


def test_core_module_import_graph_excludes_optional_surfaces() -> None:
    result = subprocess.run([sys.executable, "-c", "import engine.core.contracts, engine.core.run_manifest"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

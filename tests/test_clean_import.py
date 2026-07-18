"""Every tracked pipeline module must import from a clean checkout.

A clean checkout has only tracked files: no data_canonical/, no
data_derived/, no dukascopy_data/, no local caches. This test exports the
current HEAD via ``git archive`` into a temporary directory and imports each
pipeline module there in a subprocess, so a module that reaches for
generated data or a stray absolute path at import time fails loudly instead
of silently working because the developer's own checkout happens to have
data_canonical/ populated.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import verify_repository  # noqa: E402


def test_all_tracked_pipeline_modules_import_without_generated_data() -> None:
    with tempfile.TemporaryDirectory(prefix="test_clean_import_") as tmp:
        result = verify_repository.check_clean_checkout_imports(Path(tmp))
    assert result.passed, result.detail


def test_clean_checkout_does_not_track_generated_data_directories() -> None:
    with tempfile.TemporaryDirectory(prefix="test_clean_import_") as tmp:
        clean_dir = Path(tmp) / "clean_checkout"
        clean_dir.mkdir()
        verify_repository._export_clean_checkout(clean_dir)
        assert not (clean_dir / "data_canonical").exists()
        assert not (clean_dir / "data_derived").exists()
        assert not (clean_dir / "dukascopy_data").exists()
        assert (clean_dir / "config" / "instruments.json").exists()
        assert (clean_dir / "pipeline" / "contracts.py").exists()


@pytest.mark.parametrize("module_name", [
    "contracts", "ingest", "estimate_dynamics", "estimate_coupling",
    "simulate_integrator", "stat_arb", "legal_event", "render_quantum_figures",
    "quantum_kernel", "quantum_lindblad", "quantum_mps", "quantum_reservoir",
    "quantum_trajectories", "quantum_process_tomography",
])
def test_module_is_present_in_clean_checkout_module_list(module_name: str) -> None:
    with tempfile.TemporaryDirectory(prefix="test_clean_import_") as tmp:
        clean_dir = Path(tmp) / "clean_checkout"
        clean_dir.mkdir()
        verify_repository._export_clean_checkout(clean_dir)
        assert (clean_dir / "pipeline" / f"{module_name}.py").is_file()

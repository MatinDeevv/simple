"""Every tracked source module must import from a clean checkout.

A clean checkout has only application and research source: no generated data,
artifacts, or local caches. This test creates a code-only temporary copy and
imports each module there in a subprocess, so a module that reaches for
generated data or a stray absolute path at import time fails loudly instead
of silently working because the developer's own checkout happens to have
data/canonical/ populated.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from fxresearch.tools import verify_repository


def test_all_tracked_source_modules_import_without_generated_data() -> None:
    with tempfile.TemporaryDirectory(prefix="test_clean_import_") as tmp:
        result = verify_repository.check_clean_checkout_imports(Path(tmp))
    assert result.passed, result.detail


def test_clean_checkout_does_not_track_generated_data_directories() -> None:
    with tempfile.TemporaryDirectory(prefix="test_clean_import_") as tmp:
        clean_dir = Path(tmp) / "clean_checkout"
        clean_dir.mkdir()
        verify_repository._export_clean_checkout(clean_dir)
        assert not (clean_dir / "data").exists()
        assert not (clean_dir / "artifacts").exists()
        assert (clean_dir / "fxresearch" / "config" / "instruments.json").exists()
        assert (clean_dir / "fxresearch" / "core" / "contracts.py").exists()


@pytest.mark.parametrize("module_name", [
    *verify_repository.CLEAN_IMPORT_MODULES,
])
def test_module_is_present_in_clean_checkout_module_list(module_name: str) -> None:
    with tempfile.TemporaryDirectory(prefix="test_clean_import_") as tmp:
        clean_dir = Path(tmp) / "clean_checkout"
        clean_dir.mkdir()
        verify_repository._export_clean_checkout(clean_dir)
        package, leaf = module_name.rsplit(".", 1)
        assert (clean_dir / package.replace(".", "/") / f"{leaf}.py").is_file()

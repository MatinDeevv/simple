"""Repository-wide guard: no production module may hardcode its own copy of
the canonical ten-pair FX instrument order.

``config/instruments.json`` (via ``pipeline/contracts.canonical_pair_order``)
is the single source of truth. This test uses an AST-based scan (see
``scripts/repo_pair_order_scan.py``) so it tolerates comments, docstrings,
individual pair-name references, and explicitly-marked test fixtures (this
file and ``tests/test_contracts.py`` both define ``EXPECTED_PAIRS`` on
purpose, to validate the tracked contract end to end), while still catching
a real drifted duplicate such as the one found and fixed in
``pipeline/estimate_dynamics.py`` (``PAIRS_ALL`` used to be a second literal
copy of the order).
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
PIPELINE_DIR = Path(__file__).resolve().parents[1] / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from repo_pair_order_scan import find_duplicate_pair_order_definitions  # noqa: E402
from contracts import canonical_pair_order  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PAIRS = (
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD",
    "USDCNH", "USDCHF", "EURGBP", "EURJPY", "GBPJPY",
)


def test_tracked_contract_matches_the_fixture_used_by_this_scan() -> None:
    assert canonical_pair_order(ROOT) == EXPECTED_PAIRS


def test_no_production_module_hardcodes_a_duplicate_pair_order() -> None:
    findings = find_duplicate_pair_order_definitions(ROOT, EXPECTED_PAIRS)
    assert findings == [], f"duplicate hardcoded pair-order definitions found: {findings}"


def test_scanner_detects_an_exact_duplicate(tmp_path: Path) -> None:
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "offender.py").write_text(
        "PAIRS = ('EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', "
        "'USDCNH', 'USDCHF', 'EURGBP', 'EURJPY', 'GBPJPY')\n",
        encoding="utf-8",
    )
    findings = find_duplicate_pair_order_definitions(tmp_path, EXPECTED_PAIRS)
    assert len(findings) == 1
    assert "offender.py" in findings[0]
    assert "duplicates the tracked instrument order exactly" in findings[0]


def test_scanner_detects_a_reordered_duplicate(tmp_path: Path) -> None:
    (tmp_path / "pipeline").mkdir()
    reordered = tuple(reversed(EXPECTED_PAIRS))
    (tmp_path / "pipeline" / "offender.py").write_text(
        f"PAIRS = {reordered!r}\n", encoding="utf-8",
    )
    findings = find_duplicate_pair_order_definitions(tmp_path, EXPECTED_PAIRS)
    assert len(findings) == 1
    assert "reordering of the tracked instrument set" in findings[0]


def test_scanner_tolerates_individual_pair_references_and_comments(tmp_path: Path) -> None:
    (tmp_path / "pipeline").mkdir()
    (tmp_path / "pipeline" / "fine.py").write_text(
        "# canonical order is EURUSD, USDJPY, GBPUSD, AUDUSD, USDCAD, "
        "USDCNH, USDCHF, EURGBP, EURJPY, GBPJPY\n"
        "PRIMARY = 'EURUSD'\n"
        "MAJORS = ('EURUSD', 'USDJPY', 'GBPUSD')\n",
        encoding="utf-8",
    )
    findings = find_duplicate_pair_order_definitions(tmp_path, EXPECTED_PAIRS)
    assert findings == []


def test_scanner_ignores_the_tests_directory(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_fixture.py").write_text(
        f"EXPECTED_PAIRS = {EXPECTED_PAIRS!r}\n", encoding="utf-8",
    )
    findings = find_duplicate_pair_order_definitions(tmp_path, EXPECTED_PAIRS)
    assert findings == []

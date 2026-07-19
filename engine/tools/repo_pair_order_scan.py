"""Scan production Python modules for hardcoded duplicates of the canonical
ten-pair FX instrument order.

``engine/config/instruments.json`` (loaded through ``engine/core/contracts.py``) is the
single source of truth for instrument order. A module that re-declares the
same ten-pair sequence as a literal list/tuple risks silently drifting from
the tracked contract. This scanner uses the AST (not a text grep) so it does
not fire on comments, docstrings, or individual pair-name references, and it
skips the ``tests/`` directory, where an explicitly-named fixture tuple is
the documented, allowed way to validate the tracked contract end to end.
"""

from __future__ import annotations

import ast
from pathlib import Path

EXCLUDED_DIR_NAMES = {"tests", "artifacts", "__pycache__", ".venv-quantum", ".git", "node_modules"}


def _production_python_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for source_dir in (root / "engine", root / "research"):
        if source_dir.is_dir():
            files.extend(sorted(source_dir.rglob("*.py")))
    return files


def _literal_string_sequence(node: ast.AST) -> tuple[str, ...] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    values: list[str] = []
    for element in node.elts:
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            values.append(element.value)
        else:
            return None
    return tuple(values)


def find_duplicate_pair_order_definitions(root: Path, canonical_order: tuple[str, ...]) -> list[str]:
    """Return ``["path:line: detail", ...]`` for hardcoded duplicates found outside tests/."""
    canonical_set = set(canonical_order)
    findings: list[str] = []
    for path in _production_python_files(root):
        if any(part in EXCLUDED_DIR_NAMES for part in path.relative_to(root).parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            findings.append(f"{path.relative_to(root)}:{exc.lineno}: file failed to parse: {exc.msg}")
            continue
        for node in ast.walk(tree):
            sequence = _literal_string_sequence(node)
            if sequence is None or len(sequence) != 10:
                continue
            if tuple(sequence) == canonical_order:
                findings.append(f"{path.relative_to(root)}:{node.lineno}: "
                                f"literal ten-pair tuple duplicates the tracked instrument order exactly")
            elif set(sequence) == canonical_set:
                findings.append(f"{path.relative_to(root)}:{node.lineno}: "
                                f"literal ten-pair tuple is a reordering of the tracked instrument set")
    return findings


def main() -> int:
    import sys

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from engine.core.contracts import canonical_pair_order

    pairs = canonical_pair_order(root)
    findings = find_duplicate_pair_order_definitions(root, pairs)
    for finding in findings:
        print(finding)
    if findings:
        print(f"[FAIL] {len(findings)} duplicate pair-order definition(s) found outside tests/")
        return 1
    print("[PASS] no duplicate hardcoded pair-order definitions outside tests/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

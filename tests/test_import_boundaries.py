from __future__ import annotations
import ast
from pathlib import Path
ROOT=Path(__file__).parents[1]
FORBIDDEN={"engine.quantum","qiskit","qiskit_aer","matplotlib"}
def test_core_modules_do_not_import_optional_quantum_or_visualization_dependencies():
    offenders=[]
    for path in (ROOT/"engine").rglob("*.py"):
        rel=path.relative_to(ROOT).as_posix()
        if "/quantum/" in f"/{rel}" or "/visualization/" in f"/{rel}" or rel == "engine/cli/main.py": continue
        tree=ast.parse(path.read_text(encoding="utf-8"),filename=rel)
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import): names=[x.name for x in node.names]
            if isinstance(node,ast.ImportFrom) and node.module: names=[node.module]
            if any(name == bad or name.startswith(bad+".") for name in names for bad in FORBIDDEN): offenders.append((rel,names))
    assert not offenders

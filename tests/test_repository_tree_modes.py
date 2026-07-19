from __future__ import annotations
import subprocess, sys
from pathlib import Path

ROOT=Path(__file__).parents[1]

def _run(root:Path,tree:str):
    return subprocess.run([sys.executable,"-m","engine.tools.verify_repository","--tree",tree,"--skip-slow"],cwd=root,text=True,capture_output=True,timeout=120)

def test_head_ignores_unstaged_helper_and_index_uses_staged_helper(tmp_path):
    clone=tmp_path/"repo"
    subprocess.run(["git","clone","--shared","--quiet",str(ROOT),str(clone)],check=True)
    helper=clone/"engine/core/schema_validate.py"; helper.write_text("raise RuntimeError('selected-tree-marker')\n")
    head=_run(clone,"head"); assert head.returncode==0,head.stdout+head.stderr
    subprocess.run(["git","add","engine/core/schema_validate.py"],cwd=clone,check=True)
    index=_run(clone,"index"); assert index.returncode!=0
    assert "selected-tree-marker" in index.stderr

def test_untracked_helper_cannot_affect_head_or_index(tmp_path):
    clone=tmp_path/"repo"; subprocess.run(["git","clone","--shared","--quiet",str(ROOT),str(clone)],check=True)
    (clone/"engine/core/untracked_helper.py").write_text("raise RuntimeError('must not load')\n")
    assert _run(clone,"head").returncode==0
    assert _run(clone,"index").returncode==0

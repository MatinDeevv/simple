from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import pytest

SCRIPT = Path(__file__).parents[1] / "plugins/simple-agent-framework/scripts/nvidia_coding_loop.py"
SPEC = importlib.util.spec_from_file_location("nvidia_patch_scope", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = MODULE
assert SPEC and SPEC.loader; SPEC.loader.exec_module(MODULE)

PATCH = "diff --git a/engine/good.py b/engine/good.py\n--- a/engine/good.py\n+++ b/engine/good.py\n@@ -1 +1 @@\n-old\n+new\n"

def test_explicit_scope_accepts_only_declared_path():
    MODULE.validate_patch(PATCH, ["engine/good.py"])
    with pytest.raises(MODULE.EngineError): MODULE.validate_patch(PATCH, ["engine/other.py"])

def test_library_api_fails_closed_without_scope():
    with pytest.raises(MODULE.EngineError, match="explicit allowed scope"):
        MODULE.validate_patch(PATCH)

@pytest.mark.parametrize("marker", ["GIT binary patch", "new file mode 120000", "new file mode 160000", "rename from old.py"])
def test_dangerous_patch_types_fail(marker):
    with pytest.raises(MODULE.EngineError): MODULE.validate_patch(PATCH + marker + "\n", ["engine/good.py"])

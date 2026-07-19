from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import pytest

SCRIPT = Path(__file__).parents[1] / "plugins/simple-agent-framework/scripts/nvidia_coding_loop.py"
SPEC = importlib.util.spec_from_file_location("nvidia_endpoint_policy", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = MODULE
assert SPEC and SPEC.loader; SPEC.loader.exec_module(MODULE)

def test_default_endpoint_is_pinned():
    assert MODULE.validate_endpoint(MODULE.DEFAULT_ENDPOINT) == MODULE.DEFAULT_ENDPOINT

@pytest.mark.parametrize("url", [
    "http://integrate.api.nvidia.com/v1", "https://localhost/v1",
    "https://integrate.api.nvidia.com.evil.test/v1", "https://integrate.api.nvidia.com:444/v1",
    "https://user@integrate.api.nvidia.com/v1", "https://integrate.api.nvidia.com/v1?q=x",
    "https://integrate.api.nvidia.com:bad/v1",
])
def test_untrusted_endpoints_fail(url):
    with pytest.raises(MODULE.EngineError): MODULE.validate_endpoint(url)

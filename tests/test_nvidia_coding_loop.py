import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "plugins/simple-agent-framework/scripts/nvidia_coding_loop.py"
SPEC = importlib.util.spec_from_file_location("nvidia_coding_loop", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Response:
    def __init__(self, model: str):
        self.payload = {"choices": [{"message": {"content": "ok"}}], "model": model}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return json.dumps(self.payload).encode()


def test_client_falls_back_on_unavailable_model():
    calls = []

    def request(req, timeout):
        payload = json.loads(req.data)
        calls.append((payload["model"], timeout))
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 404, "missing", {}, io.BytesIO())
        return Response(payload["model"])

    client = MODULE.NvidiaClient("temporary", models=("kimi", "nemotron"), request=request)
    result = client.complete("system", "user")
    assert result.model == "nemotron"
    assert calls == [("kimi", 180), ("nemotron", 180)]


def test_redacts_nvidia_and_named_secrets():
    value = MODULE.redact("nvapi-abc_123 API_KEY=hello TOKEN:world safe")
    assert "nvapi" not in value
    assert "hello" not in value
    assert "world" not in value
    assert value.endswith("safe")


@pytest.mark.parametrize(
    "patch",
    [
        "not a diff",
        "diff --git a/../secret b/../secret\n",
        "diff --git a/.git/config b/.git/config\n",
        "diff --git a/good.py b/good.py\n+API_KEY=secret\n",
    ],
)
def test_rejects_unsafe_patch(patch):
    with pytest.raises(MODULE.EngineError):
        MODULE.validate_patch(patch)


def test_parses_fenced_json():
    assert MODULE._json_object('```json\n{"approved": true}\n```') == {"approved": True}


def test_test_command_is_argv_not_shell(tmp_path):
    passed, transcript = MODULE.run_tests(
        tmp_path,
        [["python", "-c", "print('safe')"]],
    )
    assert passed is True
    assert "safe" in transcript

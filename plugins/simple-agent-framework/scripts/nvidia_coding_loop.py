#!/usr/bin/env python3
"""Bounded NVIDIA Build coding loop for an already-isolated Git worktree."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

DEFAULT_ENDPOINT = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODELS = (
    "moonshotai/kimi-k2.6",
    "nvidia/nemotron-3-super-120b-a12b",
)
MAX_CONTEXT_BYTES = 160_000
MAX_PATCH_BYTES = 250_000
SECRET_PATTERN = re.compile(
    r"(?i)(nvapi-[A-Za-z0-9_-]+|(?:api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,;]+)"
)


class EngineError(RuntimeError):
    pass


@dataclass(frozen=True)
class Completion:
    model: str
    content: str


def redact(text: str) -> str:
    return SECRET_PATTERN.sub("[REDACTED]", text)


def _json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I | re.S)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise EngineError(f"model returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise EngineError("model response must be a JSON object")
    return value


class NvidiaClient:
    def __init__(
        self,
        api_key: str,
        endpoint: str = DEFAULT_ENDPOINT,
        models: Sequence[str] = DEFAULT_MODELS,
        request: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        if not api_key:
            raise EngineError("NVIDIA_API_KEY is missing")
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")
        self.models = tuple(models)
        self.request = request

    def complete(self, system: str, user: str, max_tokens: int = 8192) -> Completion:
        errors: list[str] = []
        for model in self.models:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": redact(system)},
                    {"role": "user", "content": redact(user)},
                ],
                "max_tokens": max_tokens,
                "temperature": 1.0,
                "top_p": 0.95,
                "stream": False,
            }
            request = urllib.request.Request(
                f"{self.endpoint}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                with self.request(request, timeout=180) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return Completion(model=model, content=body["choices"][0]["message"]["content"])
            except urllib.error.HTTPError as exc:
                errors.append(f"{model}: HTTP {exc.code}")
                if exc.code not in {404, 429, 500, 502, 503, 504}:
                    break
            except (KeyError, json.JSONDecodeError, OSError) as exc:
                errors.append(f"{model}: {type(exc).__name__}")
        raise EngineError("all NVIDIA models failed (" + "; ".join(errors) + ")")


def _git(root: Path, *args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=root, input=input_text, text=True,
        capture_output=True, check=False,
    )


def validate_worktree(root: Path) -> None:
    result = _git(root, "rev-parse", "--show-toplevel")
    if result.returncode or Path(result.stdout.strip()).resolve() != root.resolve():
        raise EngineError("--worktree must be a Git worktree root")
    common = _git(root, "rev-parse", "--git-common-dir")
    git_dir = _git(root, "rev-parse", "--git-dir")
    if common.returncode or git_dir.returncode:
        raise EngineError("unable to inspect Git worktree metadata")
    if Path(common.stdout.strip()).resolve() == Path(git_dir.stdout.strip()).resolve():
        raise EngineError("refusing the primary checkout; use new-task-worktree.ps1")
    if _git(root, "status", "--porcelain").stdout.strip():
        raise EngineError("worktree must be clean before the coding loop starts")


def repository_context(root: Path, context_paths: Sequence[str]) -> str:
    tree = _git(root, "ls-files").stdout
    chunks = [f"TRACKED FILES:\n{tree}"]
    for raw in context_paths:
        candidate = (root / raw).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise EngineError(f"context path escapes worktree: {raw}") from exc
        if not candidate.is_file():
            raise EngineError(f"context file not found: {raw}")
        chunks.append(f"FILE {raw}:\n{candidate.read_text(encoding='utf-8', errors='replace')}")
    result = redact("\n\n".join(chunks))
    if len(result.encode("utf-8")) > MAX_CONTEXT_BYTES:
        raise EngineError("repository context exceeds byte limit; pass fewer --context files")
    return result


def validate_patch(patch: str) -> None:
    encoded = patch.encode("utf-8")
    if not patch.startswith("diff --git ") or len(encoded) > MAX_PATCH_BYTES:
        raise EngineError("model patch is missing or exceeds the byte limit")
    for match in re.finditer(r"^diff --git a/(.+?) b/(.+?)$", patch, flags=re.M):
        for raw in match.groups():
            path = Path(raw)
            if path.is_absolute() or ".." in path.parts or path.parts[:1] == (".git",):
                raise EngineError(f"unsafe patch path: {raw}")
    if SECRET_PATTERN.search(patch):
        raise EngineError("model patch appears to contain a secret")


def apply_patch(root: Path, patch: str) -> None:
    validate_patch(patch)
    check = _git(root, "apply", "--check", "--whitespace=error-all", "-", input_text=patch)
    if check.returncode:
        raise EngineError("git apply --check failed: " + redact(check.stderr.strip()))
    applied = _git(root, "apply", "--whitespace=error-all", "-", input_text=patch)
    if applied.returncode:
        raise EngineError("git apply failed: " + redact(applied.stderr.strip()))


def run_tests(root: Path, commands: Sequence[Sequence[str]]) -> tuple[bool, str]:
    output: list[str] = []
    for command in commands:
        if not command:
            continue
        proc = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False, timeout=900)
        transcript = redact((proc.stdout + "\n" + proc.stderr).strip())[-20_000:]
        output.append(f"$ {' '.join(command)}\nexit={proc.returncode}\n{transcript}")
        if proc.returncode:
            return False, "\n\n".join(output)
    return True, "\n\n".join(output)


SYSTEM = """You are the primary coding engine in a safety-bounded Git workflow.
Return only valid JSON. Never emit credentials. Preserve repository instructions and scope.
For patch requests return {"summary":"...","patch":"a complete unified git diff"}.
Do not include generated artifacts, secrets, .env files, or unrelated changes."""


def coding_loop(
    client: NvidiaClient,
    root: Path,
    task: str,
    context_paths: Sequence[str],
    test_commands: Sequence[Sequence[str]],
    retries: int,
) -> dict[str, Any]:
    validate_worktree(root)
    context = repository_context(root, context_paths)
    plan = client.complete(
        SYSTEM,
        f"Plan this task in at most 8 concrete steps. Return JSON with a steps array.\nTASK:\n{task}\n\n{context}",
        max_tokens=2048,
    )
    plan_json = _json_object(plan.content)
    build = client.complete(
        SYSTEM,
        f"Implement the task as one minimal patch.\nTASK:\n{task}\nPLAN:\n{json.dumps(plan_json)}\n\n{context}",
    )
    build_json = _json_object(build.content)
    apply_patch(root, str(build_json.get("patch", "")))

    attempts = 0
    while True:
        passed, transcript = run_tests(root, test_commands)
        if passed:
            break
        if attempts >= retries:
            raise EngineError(f"tests failed after {attempts + 1} runs:\n{transcript}")
        repair = client.complete(
            SYSTEM,
            "Repair only the test failure with a new patch against the current worktree.\n"
            f"TASK:\n{task}\nTEST OUTPUT:\n{transcript}\nDIFF:\n{_git(root, 'diff').stdout}",
        )
        repair_json = _json_object(repair.content)
        apply_patch(root, str(repair_json.get("patch", "")))
        attempts += 1

    diff = _git(root, "diff", "--check")
    if diff.returncode:
        raise EngineError("git diff --check failed: " + redact(diff.stdout + diff.stderr))
    review = client.complete(
        "You are a strict code reviewer. Return only JSON: "
        '{"approved":true|false,"findings":["..."]}. Never approve secrets or unrelated files.',
        f"TASK:\n{task}\nTESTS:\n{transcript}\nDIFF:\n{_git(root, 'diff').stdout}",
        max_tokens=4096,
    )
    review_json = _json_object(review.content)
    if review_json.get("approved") is not True:
        raise EngineError("review rejected the change: " + json.dumps(review_json.get("findings", [])))
    return {
        "status": "approved",
        "planner_model": plan.model,
        "builder_model": build.model,
        "reviewer_model": review.model,
        "repair_attempts": attempts,
        "summary": build_json.get("summary", ""),
        "findings": review_json.get("findings", []),
    }


def _command(value: str) -> list[str]:
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not parsed or not all(isinstance(item, str) and item for item in parsed):
        raise argparse.ArgumentTypeError('--test must be a JSON string array, e.g. ["python","-m","pytest"]')
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--context", action="append", default=[])
    parser.add_argument("--test", action="append", type=_command, required=True)
    parser.add_argument("--retries", type=int, choices=range(0, 3), default=2)
    args = parser.parse_args(argv)
    models = tuple(filter(None, os.getenv("NVIDIA_MODELS", ",".join(DEFAULT_MODELS)).split(",")))
    try:
        result = coding_loop(
            NvidiaClient(os.getenv("NVIDIA_API_KEY", ""), os.getenv("NVIDIA_BASE_URL", DEFAULT_ENDPOINT), models),
            args.worktree.resolve(), args.task, args.context, args.test, args.retries,
        )
    except EngineError as exc:
        print(json.dumps({"status": "failed", "error": redact(str(exc))}), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

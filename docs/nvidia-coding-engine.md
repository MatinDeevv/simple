# NVIDIA coding engine

The repository's external coding loop uses NVIDIA Build through its
OpenAI-compatible chat-completions endpoint. It does not replace the model inside
Codex itself. It is a separate, bounded planner-builder-reviewer that operates
only in an isolated Git worktree.

## Safety contract

- `NVIDIA_API_KEY` comes from the process environment or the ignored root `.env`.
- The key is never included in prompts, output, receipts, source, or Git history.
- The primary checkout and dirty worktrees are rejected.
- The caller chooses context files and test command argument arrays explicitly.
- Model output can modify files only through a size-limited unified diff checked
  by `git apply --check`.
- Test failures get at most two model repair attempts.
- A clean test run, `git diff --check`, and reviewer approval are required.
- The engine leaves an approved diff for human review; it never commits or pushes.

## Configuration

```text
NVIDIA_API_KEY      required secret
NVIDIA_BASE_URL     default: https://integrate.api.nvidia.com/v1
NVIDIA_MODELS       comma-separated preference order
```

The default preference order is `moonshotai/kimi-k2.6`, then
`nvidia/nemotron-3-super-120b-a12b`. Unavailable and transiently failing models
fall through to the next configured model.

## Run

Create a clean task worktree with `new-task-worktree.ps1`, then invoke:

```powershell
& plugins/simple-agent-framework/scripts/invoke-nvidia-coding-loop.ps1 `
  -Task 'Add a bounded feature and tests' `
  -Worktree 'C:\path\to\task-worktree' `
  -Context @('AGENTS.md', 'relevant/module.py', 'tests/test_relevant.py') `
  -Test @('["python","-m","pytest","tests/test_relevant.py","-q"]')
```

Inspect the resulting diff, test receipts, and reviewer output before committing.

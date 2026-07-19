# Agent framework

Use `simple-repo-orchestrator` for multi-agent repo work. Roles: `simple-scout`,
`simple-builder`, `simple-tester`, `simple-reviewer`.

- Before edits: run `plugins/simple-agent-framework/scripts/preflight.ps1`.
- One agent, one branch/worktree. Do not switch branches or reset shared checkout.
- Preserve other-agent changes. Stage explicit paths only.
- Test before commit; store receipts in `.agents/receipts/` (ignored/local).
- Hooks are disabled by default. Never copy hook templates into active Codex config without user approval.
- Builder may create a worktree only via `new-task-worktree.ps1`; it refuses a dirty target.
- Tester records exact command, exit code, UTC time, HEAD, and output SHA-256.
- Reviewer blocks commits containing another agent's files, data, secrets, generated artifacts, or unchecked failures.

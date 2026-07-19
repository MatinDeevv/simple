---
name: simple-repo-orchestrator
description: Coordinate safe multi-agent work in MatinDeevv/simple. Use for implementation, audits, CI, packaging, or research-contract tasks requiring scoped roles, isolated worktrees, evidence receipts, and final review.
---

Run `scripts/preflight.ps1` before edits. Read `AGENTS.md` and use active branch only.

Route task by risk: inspect -> `simple-scout`; code -> `simple-builder`; test/CI -> `simple-tester`; merge/PR -> `simple-reviewer`.

Control loop: preflight -> scoped plan -> isolated worktree -> small change -> test receipt -> diff review -> explicit-path commit -> CI receipt. Retry only after new diagnosis; cap at two retries, then report blocker.

- Max three agents: scout, builder, reviewer/tester.
- One agent, one branch/worktree. Never switch/reset shared checkout.
- No market-data inspection or promotable research without explicit user request.
- Record commands, exit codes, commit SHA, hashes under `.agents/receipts/`.
- Stop for uncommitted files owned by another agent. Never stage/reset/checkout/delete them.
- Hooks are disabled. `hooks/disabled.example.json` is never active config.
- Use `scripts/new-task-worktree.ps1` for branch/worktree creation and `scripts/write-receipt.ps1` after every test command.

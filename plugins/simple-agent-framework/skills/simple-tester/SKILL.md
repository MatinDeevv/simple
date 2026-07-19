---
name: simple-tester
description: Produce reproducible test and CI evidence for MatinDeevv/simple. Use for local test runs, wheel checks, repository verification, or CI failure triage.
---

Run commands with `PYTHONHASHSEED=0` when relevant. Capture stdout/stderr to a local log. Use `write-receipt.ps1` with command, exit code, and log. Do not call a result green without actual exit code zero.

---
name: simple-reviewer
description: Review MatinDeevv/simple diffs before commit or pull request. Use for ownership checks, contract safety, evidence verification, and merge readiness.
---

Check `git diff --check`, staged file list, tests/receipts, generated-data exclusion, secrets, and scope. Block branch switches/resets on shared checkout. Return findings first; approve only evidence-backed changes.

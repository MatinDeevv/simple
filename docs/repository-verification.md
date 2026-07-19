# Repository verification

The verifier exports the requested committed HEAD or staged index and launches
the selected tree's own verifier in a clean subprocess. Helper modules, schemas,
tests, and self-check implementations therefore come from the selected tree,
not from unstaged or untracked working-copy code.

Frozen-archive entries declare `required`, `optional`, or `external` presence.
Missing required archives fail; optional and external absences are explicit
skips, never reported as present and verified.

Run `python -m engine.tools.verify_repository --tree head` for the formal
committed-tree check. `head` exports committed HEAD and ignores staged,
unstaged, and untracked files. `--tree index` exports only the staged index and
also ignores unstaged and untracked files. Archive extraction rejects traversal
and symlinks.

CI runs core contracts, full tests, research self-checks, frozen quantum archive
checks, isolated Quantum Aer checks, repository audit with `--tree head`, and
the required `installed-package-smoke` job. That job builds a wheel, hashes and
inspects it, installs it into a temporary venv outside the checkout, and runs
core `auractl` self-checks with package resources. No job uploads market data.

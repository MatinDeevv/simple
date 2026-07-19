# Repository verification

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

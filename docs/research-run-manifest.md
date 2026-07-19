# Research Run Manifest

`engine/core/run_manifest.py` produces a self-verifying JSON record of what
produced a research artifact. It is a general-purpose module usable by any
pipeline script; nothing currently calls it automatically from
`stat_arb.py`, `legal_event.py`, or `simulate_integrator.py` — wiring it into
a specific run is a follow-up, not part of this branch.

## Fields

```json
{
  "run_id": "uuid4",
  "created_at_utc": "ISO-8601",
  "git_commit": "40-hex sha, or null if git is unavailable",
  "git_status": "clean | dirty | unavailable",
  "dirty_worktree": "true | false | null (null only when git_status is unavailable)",
  "python_version": "sys.version's first token",
  "platform": "platform.platform()",
  "dependency_versions": {"...": "caller-supplied, not auto-detected"},
  "configuration_sha256": "sha256 of the canonical JSON of the caller's config dict",
  "source_file_sha256": {"relative/path.py": "sha256"},
  "input_artifact_sha256": {"relative/path": "sha256"},
  "output_artifact_sha256": {"relative/path": "sha256"},
  "random_seeds": {"...": "caller-supplied"},
  "frozen_contract_version": "caller-supplied version string",
  "holdout_status": "not_used | clean_holdout | burned_acknowledged",
  "required_tests_passed": "caller-supplied boolean -- this module never runs tests itself",
  "promotion_eligible": "derived, see below",
  "promotion_blockers": ["human-readable reasons promotion_eligible is false"],
  "manifest_sha256": "sha256 of the canonical JSON of every field above"
}
```

## Promotion eligibility

`promotion_eligible` is `True` only if **all** of the following hold:

1. `git_status == "clean"` (not `"dirty"`, and not `"unavailable"` — a
   synthetic CI environment with no git binary or no `.git` directory gets
   `git_status: "unavailable"` and is never promotion-eligible, by design).
2. `holdout_status != "burned_acknowledged"`.
3. `source_file_sha256` is non-empty (at least one source file was recorded).
4. `required_tests_passed is True`.

This module **never infers success from an artifact's existence.** It does
not run tests, does not run self-checks, and does not check whether
`output_artifact_sha256` is non-empty as a promotion criterion — a caller
that wants that must actually run the tests/self-checks and pass the
resulting boolean into `required_tests_passed` explicitly. A manifest with
five populated output-artifact hashes and `required_tests_passed=False` is
still not promotion-eligible.

## Integrity

`manifest_sha256` covers the canonical JSON (`sort_keys=True,
separators=(",", ":")`) of every other field. `verify_manifest_integrity(payload)`
recomputes it and compares. `write_manifest()` refuses to write a payload
whose hash doesn't match its own contents — so a manifest cannot be silently
hand-edited (e.g., flipping `promotion_eligible` to `True` without redoing
every gate) and then saved through this module; `tests/test_run_manifest.py`
has an explicit tampering-detection test for this.

## What this does not do

- It does not decide what "required tests" means for a given module, or run
  them. That is the caller's responsibility, on purpose: a manifest-writing
  step must not be the thing that also decides tests passed.
- It does not enforce that `output_artifact_sha256` paths were produced by
  *this* run rather than a stale file from a previous one — hashing an
  existing file only proves the file's current content, not its provenance.
  Pair this with `input_artifact_sha256`/`source_file_sha256` and a fresh
  `output_dir` per run if that guarantee is needed.
- It is not wired into any pipeline module's `main()` yet.

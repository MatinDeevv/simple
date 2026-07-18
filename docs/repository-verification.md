# Repository Verification

This document describes what `fxresearch/tools/verify_repository.py` and
`.github/workflows/ci.yml` actually check, and â€” just as importantly â€” what
they do not. It covers the reliability-hardening work on
`agent2/research-reliability-hardening`: clean-checkout imports, shared
contract audits, frozen-archive protection, and the burned-holdout guard.
It does not cover the statistical-arbitrage model, the legal-event model, or
the quantum-archive research content themselves; see `docs/stat-arb.md`,
`docs/legal-event.md`, and `docs/quantum-frontier.md` for those.

## Running it locally

```powershell
python fxresearch/tools/verify_repository.py
python fxresearch/tools/verify_repository.py --skip-slow   # cheap static checks only
```

Exit code is nonzero if any check fails. Each line is `[PASS]`, `[FAIL]`, or
`[SKIP]` with a one-line detail. `--skip-slow` omits the clean-checkout
import test, `pytest tests`, and every module's `--self-check` (all of which
spawn subprocesses and take real time); use it for a fast sanity pass while
iterating.

## What each check actually verifies

| Check | What it does | What it does NOT prove |
|---|---|---|
| tracked configuration files exist | `fxresearch/config/instruments.json`, `fxresearch/config/legal-event-schema.json`, `fxresearch/config/frozen-archives.json`, both requirements files, and `ci.yml` are tracked by git, present, and (for JSON) parse | that their *content* is semantically correct beyond JSON validity |
| clean-checkout imports | `git archive HEAD` into a temp dir, then `python -c "import <module>"` for every tracked `pipeline/*.py`, subprocess per module | that the module *runs* correctly, only that it *imports* without needing `data/canonical/`, `data/derived/`, or any other generated/gitignored path |
| core dependency versions | pinned versions in `requirements-core.txt` match `importlib.metadata.version(...)` in the current interpreter | that pinned versions are the *right* versions, only that the environment matches the pin |
| pytest tests/ | runs the full suite, including modules owned by other in-flight work | flaky/slow tests are still counted as failures; there is no retry |
| required module self-checks | `simulate_integrator.py`, `stat_arb.py`, `legal_event.py` `--self-check`, each on deterministic synthetic input | anything about real canonical-data behavior; self-checks never touch `data/canonical/` |
| quantum-archive self-checks | the six numpy-only quantum-archive modules' `--self-check` | quantum advantage or physical validity â€” see `docs/quantum-frontier.md`; these are a negative-results archive |
| quantum-aer self-check | `quantum_aer_noise.py --self-check`, only if `qiskit-aer` is importable | nothing, if skipped â€” an absent isolated environment is not a failure |
| state schemas valid | every `fxresearch/config/schemas/*.schema.json` is well-formed under `fxresearch/core/schema_validate.py`'s subset | that any specific *artifact* on disk currently conforms â€” see [Schema validation](#schema-validation) below |
| shared instrument order | `pipeline/contracts.canonical_pair_order` loads and resolves to ten unique pairs, and no production module hardcodes a duplicate copy (`fxresearch/tools/repo_pair_order_scan.py`) | â€” |
| no null bytes in tracked files | every tracked text-like file (`.py`, `.md`, `.json`, `.yml`, ...) contains no `\x00` | binary tracked files (images, parquet) are intentionally excluded |
| no empty required source files | every `pipeline/*.py`, `tests/*.py`, `scripts/*.py` on disk has non-whitespace content | â€” |
| frozen archives unchanged | `fxresearch/config/frozen-archives.json` registry: for each entry present on disk, its sha256 (and recorded `version` field) matches the frozen value | if the file is absent (expected in a clean checkout / CI, since `data/derived/` is gitignored), this is a pass, not a failure â€” absence is not validity, but it is also not a violation of "don't overwrite" |
| burned-holdout guard enabled | invokes `stat_arb.py` with no flags; asserts nonzero exit, a refusal message mentioning "burned", and that `data/derived/` contents are byte-for-byte unchanged before/after | this check depends on `fxresearch/models/statistical/stat_arb.py` internals (the `--allow-burned-holdout-research` flag) that are owned by other in-flight work; if that flag is ever renamed or removed, this check fails loudly by design |

## Schema validation

`fxresearch/core/schema_validate.py` implements a small, dependency-free subset of
JSON Schema (`type`, `required`, `properties`, `items`, `enum`,
`additionalProperties`) â€” not a full JSON Schema library. `fxresearch/config/schemas/`
holds five documents:

- `stat-arb-emission.schema.json`, `stat-arb-summary.schema.json` â€” `stat_arb.py`
  is under active development (OQ-14); these schemas deliberately require only
  the structurally stable identity/eligibility/probability fields, not every
  diagnostic or outcome column, and set `additionalProperties: true`.
- `legal-event-study.schema.json`
- `integrator-checkpoint.schema.json` â€” validated against a real checkpoint
  produced by `simulate_integrator.write_checkpoint` in `tests/test_schema_validation.py`.
- `research-run-manifest.schema.json` â€” validated against a real manifest
  produced by `pipeline/run_manifest.build_run_manifest` in the same test file;
  this one is `additionalProperties: false` (the manifest shape is fully owned
  by this branch, so it can be strict).

`fxresearch/tools/verify_repository.py`'s "state schemas valid" check only validates
that the schema *documents* are well-formed. It does not validate live
`data/derived/*.json` artifacts against them â€” those files are gitignored, so
a clean checkout doesn't have them, and `stat_arb.py` v0.2 has not produced
any yet (see decision log D-028 in `docs/state-schema.md`). Once v0.2
artifacts exist, validating them against `stat-arb-summary.schema.json` is a
natural extension, not yet wired in.

## CI jobs (`.github/workflows/ci.yml`)

Six jobs, all triggered on push to `main`, on every pull request, and via
manual `workflow_dispatch` (with an optional `run_extended_checks` input,
currently reserved for future slower property/stress checks â€” nothing is
gated behind it yet). All run in parallel (no artificial `needs:` chain), each
with its own timeout, its own dependency cache, and `actions/upload-artifact`
for its test/self-check output on every run (pass or fail):

- **core-contracts** â€” byte-compiles `pipeline`, `tests`, `scripts`; runs
  `tests/test_contracts.py`; runs `fxresearch/tools/verify_repository.py`.
- **core-tests** â€” the full `pytest tests -q` suite.
- **research-self-checks** â€” `simulate_integrator.py`, `stat_arb.py`,
  `legal_event.py`, each `--self-check`.
- **quantum-archive-checks** â€” the six numpy-only quantum-archive modules.
- **quantum-aer-checks** â€” `quantum_aer_noise.py --self-check`, in its own job
  with `requirements-quantum.txt` so a qiskit-aer install failure can never
  block the core jobs (or vice versa).
- **repository-audit** â€” `fxresearch/tools/verify_repository.py` plus this branch's own
  new test files (clean-import, duplicate-pair-order, evaluation-protocol,
  run-manifest, schema-validation, reproducibility-contracts).

None of the six core jobs require `data/canonical/`, `data/derived/`, or
network access beyond the dependency-install step.

## What CI does not prove

- That any research model is profitable, causal, or production-ready. See
  the "promotion_status" field in every arena's summary output â€” it says so
  directly.
- That the pinned dependency versions are free of vulnerabilities or bugs.
- That a real (non-synthetic) canonical-data run behaves the same as the
  self-checks; self-checks are deliberately synthetic-only so they never
  need `data/canonical/`.
- That two contributors' local environments match CI exactly â€” only that CI's
  own environment matches the pins in `requirements-core.txt`.

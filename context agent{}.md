# Codex session handoff — `simple`

Last refreshed: 2026-07-19 (local workspace state). This is an operational handoff, not a claim that the Agent 1B brief is fully accepted.

## Repository and current state

- Repository: `MatinDeevv/simple`
- Local root: `C:\Users\marti\Desktop\simple`
- Current branch: `agent1/policy-sensitivity-and-publication-hardening`
- Mandatory Agent 1B base: `199408ffc19edc791e60f1c2defcc86f4bebd973` (`199408f`)
- Current Agent 1B commit: `ffad2045ac118ceb9196a0f6e34a3216c15f9360` (`ffad204`)
- Commit message: `fix(stat-arb): align policy populations and final artifact provenance`
- Remote branch is pushed: `origin/agent1/policy-sensitivity-and-publication-hardening`
- `origin/main` and local `main` currently resolve to `199408f`.

The worktree is **not clean**. At handoff it contains pre-existing/unrelated local items:

- modified: `.gitignore`
- untracked: `.agents/`, `AGENTS.md`, `plugins/`

Do not delete, reset, or commit these without determining their owner and purpose. This handoff file is also untracked until intentionally committed.

## Repository shape and scope

- The code package is `engine/`; this is a research codebase, not a web app.
- Keep the repository root compact. Do not reintroduce `fxresearch/`, `research/`, or an `app/` wrapper.
- The Agent 1B task concerns stat-arb evaluation/provenance only. Do not inspect newer market data, run promotable experiments, tune with burned holdout, or make profitability/execution claims.
- The frozen target, frozen model probabilities and coefficients, thresholds, horizons, regime settings, basket constraints, and optimizer mathematics must not change.

## Agent 1B implementation already committed

Commit `ffad204` changes only the intended stat-arb area:

- `engine/models/statistical/stat_arb.py`
- `tests/test_stat_arb_artifact_transaction.py`
- `tests/test_stat_arb_policy_sensitivity.py` (new)
- `tests/test_stat_arb_large_frame_validation.py` (new)
- `docs/stat-arb.md`
- `docs/evaluation-protocol.md`

### Implemented behavior

1. **Policy-consistent sensitivity views**
   - Policies are evaluated in fixed order:
     `independent_research_entries`, `first_entry_per_signal_episode`,
     `non_overlapping_global`, `non_overlapping_component`,
     `non_overlapping_basket`, `one_representative_per_target_cluster`.
   - Each view refits the three-class comparator from policy-accepted training rows, while retaining frozen model probabilities for accepted OOS rows.
   - It reports train/OOS raw and accepted counts, episode/cluster counts, class counts, fallback tiers, comparator details, scores, and uncertainty.
   - There is no score-based policy selection.

2. **Explicit split-boundary contracts**
   - Non-independent policies expose `reset_at_oos_start` and `carry_chronological_state_into_oos` under `boundary_contracts`.
   - The default sensitivity view is `reset_at_oos_start`; carry-state is secondary/operational.

3. **Final-path manifest provenance**
   - Staged artifact bytes are hashed under final logical paths of the form `stat_arb_v0_2_1_runs/<run-id>/<relative-path>`.
   - Manifest hash is recomputed after the final artifact mapping is installed, integrity/schema are checked, then staging is atomically renamed.
   - Every final artifact is rehashed after rename. A mismatch removes the just-published final directory and raises `ContractError`.

4. **Bootstrap and validation hardening**
   - Multiclass bootstrap rejects invalid sample/block counts, malformed/non-finite/non-normalized probability matrices, invalid labels, and duplicate source indices.
   - Evaluation populations deliberately reject duplicate source indices: one physical minute cannot be duplicated as separate observations without an explicit entry identity.
   - Frame validation now performs broad vectorized checks rather than calling `DataFrame.iterrows()` for every record; strict serialization is limited to representative rows.

5. **Publication interruption cleanup**
   - Publication tracks whether rename completed. `BaseException` before publication removes staging and is re-raised; a verified final run is retained after publication.

## Evidence collected locally

Run with Python 3.11 and `PYTHONHASHSEED=0` where relevant:

```text
python -m pytest tests/test_entry_diagnostics.py tests/test_evaluation_protocol.py tests/test_stat_arb.py tests/test_stat_arb_artifact_transaction.py tests/test_stat_arb_policy_sensitivity.py tests/test_stat_arb_large_frame_validation.py -q
60 passed in 81.69s

python -m pytest tests/test_stat_arb_artifact_transaction.py tests/test_stat_arb_policy_sensitivity.py tests/test_stat_arb_large_frame_validation.py -q
7 passed in 53.60s

python -m engine.models.statistical.stat_arb --self-check
passed=true; prefix_emissions_bitwise_equal=true;
transform_inverse_max_error=0.0; gap_resets=1;
train_samples=387; oos_samples=313
```

`git diff --check` was clean when the Agent 1B change was committed.

The full suite command `python -m pytest tests -q` was started but the outer execution timeout stopped it after about 183 seconds. No test assertion failure was printed before termination; an `OSError: [Errno 22] Invalid argument` appeared because the process was force-terminated. Do **not** represent the full suite as passed.

## Important incomplete/verify-before-PR items

The Agent 1B brief is unusually strict. The next session should audit and complete these before claiming acceptance:

1. Run every required focused test as separate commands **twice**, record passed/failed/skipped/warnings/elapsed time, then run the full suite with enough time to finish.
2. Add and run direct interruption tests with a controlled `BaseException` at all mandated phases: pre-write, after Parquet writes, before manifest, before rename, after rename.
3. Add the required provenance adversarial tests: tampered final artifact detection, failed post-rename verification cleanup, external `out_dir` stable logical paths, Windows separator normalization, deterministic fixed-run paths, and byte preservation across staging/rename.
4. Preserve or add the named descriptive diagnostic `oos_filter_only_using_independent_training_comparator` if the old OOS-filter-only view is retained. It must never be called policy-consistent.
5. Tighten sparse-policy handling so only a specific, deterministic bootstrap-insufficiency condition is converted to `insufficient_bootstrap_support`; arbitrary `ContractError` must not be relabeled as insufficiency.
6. Confirm every required pre-bootstrap support metric/status exists, including unique source indices, target clusters, causal segments, entry gaps, and populated timeline-block counts.
7. Audit vectorized validation against the full brief: conditional-probability null grouping, `accepted => entry_eligible`, non-negative active/overlap/optimizer values, cluster IDs, graph/daily contracts, and timestamp contract.
8. Add direct comparator-mutation tests: rejected training-label mutations do not affect the baseline; accepted training-label mutations do; OOS label mutations do not affect any baseline; and the boundary fixture actually differs.
9. Verify primary bootstrap remains fail-closed while sparse secondary policies produce local statuses.

## Agent ownership / merge safety

Agent 1B must not modify Agent 2-owned files, including CI, packaging, CLI, generic manifest/schema validation, generic manifest schema, and their corresponding tests/docs. The committed `ffad204` file list above avoids those files.

There is a separate historical/concurrent package branch artifact:

- `b9488adbc093d9c2fe2914972a51dfd8bf65ea09` — Agent 2 package/reproducibility work
- `ad49b09c12045af1d473048fc54a746068830937` — accidental mixed aggregate on `agent2/package-ci-and-provenance-hardening`

Do not merge `ad49b09` into this branch without a deliberate review. Do not merge Agent 2 work as part of the Agent 1B PR.

## Remote / pull request status

The Agent 1B branch was pushed successfully. A server-supplied compare/PR URL is:

`https://github.com/MatinDeevv/simple/pull/new/agent1/policy-sensitivity-and-publication-hardening`

`gh` was unavailable locally and browser authentication returned HTTP 401, so no pull request or GitHub Actions result was verified. The strict brief requires a PR rather than a direct push to `main`; do not claim CI passed. An earlier user request to merge everything to `main` conflicts with the later Agent 1B instruction—obtain a fresh explicit decision before any merge to `main`.

## Recommended continuation sequence

1. Inspect `git status --short`, `git branch -vv`, `git log --oneline --decorate -8`, and `git diff 199408f..HEAD --` before editing.
2. Keep unrelated `.gitignore`, `.agents/`, `AGENTS.md`, and `plugins/` changes untouched.
3. Read the full Agent 1B brief, then audit `ffad204` specifically against its missing acceptance items above.
4. Implement only within Agent 1B file ownership; use stat-arb-local helpers rather than generic Agent 2 infrastructure.
5. Run the mandated commands with `PYTHONHASHSEED=0`; allow the full suite to complete; record exact results in the final report.
6. Run `git diff --check`, confirm no Agent 2 files changed and no post-2024 market data were inspected, push the branch, then open the PR.

## Required final-report headings for Agent 1B

Return exactly these numbered items once truly complete:

1. Base SHA
2. Branch
3. Final commit SHA
4. Files modified
5. Files created
6. Policy-consistent population design
7. Split-boundary design
8. Comparator refit evidence
9. Bootstrap insufficiency design
10. Duplicate-source policy
11. Manifest final-path design
12. Final hash verification design
13. Large-frame validation benchmark
14. Interruption cleanup
15. Tests added
16. Commands executed
17. Exact results
18. Determinism evidence
19. Remaining blockers
20. Pull-request URL
21. Confirmation that no Agent 2 files were modified
22. Confirmation that no post-2024 data were inspected

Never claim profitability, execution readiness, or completed CI without evidence.

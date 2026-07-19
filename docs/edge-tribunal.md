Edge Tribunal: Purpose and Non-Proofs
====================================

The Edge Tribunal is a governance engine for preregistered falsification and research promotion.
It enforces an irreversible state machine, cryptographic artifact chaining of artifact history, and strict
separation between research decisions and trading authorizations.

Purpose
-------
* Enforce preregistration before any outcome is examined.
* Require machine-evaluable gates before results are known.
* Bind to a specific dataset and holdout interval.
* Produce a verdict that is a research decision only, never a trading authorization.
* Enable irreversible amendment via new experiment IDs.
* Provide a tamper-evident audit chain for artifact integrity.

Non-Proofs (What a Verdict Does NOT Prove)
------------------------------------------
A passed Tribunal verdict does not prove future profitability.
A passed historical Tribunal verdict does not authorize trading.
A passed historical Tribunal verdict only permits the next preregistered stage.

Irreversible State Machine
--------------------------
States (in order):
  DRAFT -> SEALED -> DATA_BOUND -> EVIDENCE_RECORDED -> VERDICT_ISSUED -> ARCHIVED

Transitions are strictly sequential; no backward or skip transitions are allowed.
A sealed experiment cannot return to DRAFT; to change a sealed plan one must
create a new experiment that records a parent ID and an amendment reason.

Artifact/Hash Chain
-------------------
Each experiment lives in a directory of immutable, versioned snapshots:
  experiment_dir/
    CURRENT               -> name of the authoritative version directory
    versions/v000001/     -> snapshot: artifacts + state + audit
    versions/v000002/     -> ...

Every state transition builds the next snapshot in a sibling staging directory,
validates it (audit chain, schema contracts), and atomically renames it into
place, moving the CURRENT pointer. Prior snapshots are never modified, so no
artifact is ever silently replaced. The audit log is a hash-chained JSONL file
that makes tampering evident.

Preregistration
---------------
A plan (preregistration) is registered in the DRAFT state and sealed immutably.
It contains:
* hypothesis, economic mechanism, expected direction, holding horizon, failure mode.
* exact code contract (repo, commit SHA, optional tree SHA).
* data contract (dataset fingerprint, columns, frequency, execution-data availability).
* target contract, primary/secondary comparators, metrics.
* uncertainty contract (confidence intervals, p-value calculations).
* entry-policy contract, robustness contract, concentration limits.
* multiple-testing correction method and family size.
* acceptance gates (hard validity gates first, statistical evidence gates second).
* automatic-rejection conditions and promotion ceiling (maximum verdict).
All gates are machine-evaluable: a numeric threshold, an explicit comparison,
and a metric path into the evidence bundle. Vague criteria like "looks promising"
are structurally impossible to register.

Seal and Amendments
-------------------
After sealing, the plan is cryptographically bound to the seal artifact.
Any attempt to change the sealed plan without creating a new experiment is
detected by the hard-gate "hard_no_undeclared_amendment".
Amendments are only possible by starting a new experiment that:
* records the parent experiment ID,
* provides an amendment reason,
* starts again in DRAFT state.
The amended experiment receives a new experiment ID and a new seal.

Dataset Binding
---------------
Binding occurs after sealing and before evidence recording. It hash-locks the
exact evaluation data (file hashes, row counts, interval, pair scope) and
records data-quality counters, execution-data availability, and the holdout
claim. The binding also records a reduced promotion ceiling derived from what
the dataset lacks (e.g., BID-only data cannot support more than
FORWARD_TEST_ELIGIBLE). The binding is immutable and must match the seal.

Holdout Registry and Reuse
--------------------------
The persistent holdout registry guarantees that a given holdout interval on a
dataset fingerprint can be claimed as "untouched" exactly once. It records every
claim, detects exact and partial overlap, survives renames and new model
commits, and permits reuse only as an explicitly declared forensic reuse that
automatically blocks promotion. The registry uses a transactional update (temp
file + atomic replace) under an exclusive lock file and carries an integrity
hash over its entries.

Evidence Contract
-----------------
The Tribunal consumes exactly one structured JSON evidence bundle per experiment.
It never scrapes console output, never executes anything named in the bundle,
and never treats a missing result as a pass. Every mandatory comparator,
control, policy view, and block length must be physically present; otherwise
ingestion fails. The evidence bundle includes:
* producer, dataset, population, primary model and comparator,
* negative controls, robustness matrix, concentration metrics,
* data-quality counters, execution data (ask/spread/fill/commission/latency/impact/
  notional/conversion availability), multiplicity artifacts, and provenance.

Hard Versus Statistical Gates
-----------------------------
All gates are evaluated in two phases:

1. Hard validity gates (must all PASS):
   * hard_state_chain, hard_plan_hash, hard_seal_integrity,
   * hard_dataset_binding, hard_no_undeclared_amendment,
   * hard_no_clean_holdout_reuse, hard_code_commit_match,
   * hard_configuration_match, hard_model_contract_match,
   * hard_dataset_hash_match, hard_evidence_complete,
   * hard_audit_chain, hard_probabilities_valid, hard_finite_metrics,
   * hard_mandatory_controls, hard_mandatory_policies,
   * hard_mandatory_block_lengths, hard_no_path_traversal,
   * hard_no_duplicate_evidence_ids.
   A single hard failure yields verdict INVALID_EXPERIMENT.

2. Statistical evidence gates (evaluated after hard gates pass):
   * acceptance gates from the plan (primary/secondary metrics, robustness,
     concentration, power, etc.).
   * Each gate is required or optional; a required gate that FAILS leads to
     REJECTED; a required gate that is MISSING, INVALID, or INSUFFICIENT
     (according to the plan's on_insufficient policy) may lead to
     INCONCLUSIVE or INVALID.
   * Power-gate failures are treated as INSUFFICIENT (not a rejection)
     because they indicate lack of evidence, not falsification.

Multiplicity
------------
The pre-registered multiple-testing correction (NONE_SINGLE_PRIMARY,
BONFERRONI, or HOLM_BONFERRONI) is applied only to the preregistered primary
family, never to secondary exploratory metrics. The family size is fixed at
plan time and cannot shrink after testing. The procedure returns per-hypothesis
rejection decisions and an explicit primary outcome.

Robustness
----------
The robustness matrix is planned (which cells must exist, which are mandatory,
what pass proportion is required, how missing cells count) before evidence
exists. The evaluator never selects the best-performing cell, never drops a
missing cell from the denominator unless the plan preregistered that rule, and
never treats an insufficient cell as a pass.

Concentration
-------------
Concentration limits guard against evidence being dangerously concentrated in a
single episode/cluster/segment despite an adequate sample size. The contract
specifies limits; the evaluator reports status (pass, dangerously_concentrated,
inconclusive_small_sample) and the verdict engine treats a dangerously
concentrated status as a hard failure.

Power
-----
Power gates are statistical gates that, when failed, indicate insufficient
evidence rather than falsification. The verdict engine maps a failed power gate
to INCONCLUSIVE (or INSUFFICIENT status) rather than REJECTED, reflecting that
the experiment lacks the ability to detect an effect, not that the effect is
absent.

Exact Verdict Hierarchy (worst to best)
---------------------------------------
INVALID_EXPERIMENT, REJECTED, INCONCLUSIVE, RESEARCH_ONLY,
FORWARD_TEST_ELIGIBLE, PAPER_FORWARD_TEST_PASSED, MICRO_LIVE_REVIEW_REQUIRED

Historical evidence (the only kind available today) can never exceed
FORWARD_TEST_ELIGIBLE. The grammar cannot express LIVE_READY,
PRODUCTION_READY, PROFITABLE, or DEPLOY verdicts.

Research-Reality Levels 0-5
---------------------------
LEVEL_0_SYNTHETIC_VALIDATION_ONLY,
LEVEL_1_HISTORICAL_BID_ONLY_RESEARCH,
LEVEL_2_QUOTE_AND_COST_AWARE_HISTORICAL_RESEARCH,
LEVEL_3_PREREGISTERED_PAPER_FORWARD_TEST,
LEVEL_4_MICRO_LIVE_HUMAN_REVIEW,
LEVEL_5_PRODUCTION_GOVERNANCE

The Tribunal only ever recommends advancing one research-reality level at a
time. For a historical verdict of FORWARD_TEST_ELIGIBLE the maximum next
stage is LEVEL_3_PREREGISTERED_PAPER_FORWARD_TEST.

Transaction/Locking Design
--------------------------
All state-transition operations (seal, bind-data, record-evidence, evaluate,
archive) are transactional:
* A staging directory is prepared next to the current version.
* All artifacts are written there, fully validated (audit chain, schema).
* If any validation fails, the staging directory is removed and the prior
  snapshot remains authoritative via the CURRENT pointer.
* On success, the staging directory is atomically renamed into place and the
  CURRENT pointer is updated (atomic replace).
An exclusive lock file (experiment.lock) prevents concurrent transitions for the
same experiment.

Audit Limitations
-----------------
The audit log is hash-chained and tamper-evident **only** when a trusted copy of
the latest event hash is retained somewhere the log's owner cannot rewrite.
It is not a blockchain, not a digital signature, and does not prove any human
identity. An attacker who can replace the entire log and present a new hash
can evade detection; the design assumes the operator keeps at least one honest
copy of the latest hash.

CLI Synthetic Walkthrough
-------------------------
The module ships a standalone CLI and remains a potential future integration
target for the shared `auractl` interface. Example workflow with synthetic data:

  python -m engine.experiments.edge_tribunal init --experiment-dir ./exp0 --plan ./plan.json
  python -m engine.experiments.edge_tribunal seal --experiment-dir ./exp0
  python -m engine.experiments.edge_tribunal bind-data --experiment-dir ./exp0 --dataset-manifest ./manifest.json --registry-root ./holdouts
  python -m engine.experiments.edge_tribunal record-evidence --experiment-dir ./exp0 --evidence ./evidence.json
  python -m engine.experiments.edge_tribunal evaluate --experiment-dir ./exp0
  python -m engine.experiments.edge_tribunal show-state --experiment-dir ./exp0
  python -m engine.experiments.edge_tribunal report --experiment-dir ./exp0

The repository also executes all four deterministic synthetic verdict examples
without reading market data:

  python examples/edge-tribunal/run_synthetic_examples.py --output-root ./synthetic-outcomes

The runner asserts INVALID_EXPERIMENT for a post-recording configuration
mismatch, REJECTED for comparator loss, INCONCLUSIVE for insufficient target
clusters, and FORWARD_TEST_ELIGIBLE for a fresh fully passing synthetic case.

Future `auractl` Integration
------------------------------
The standalone CLI above implements the complete Tribunal lifecycle. A future
`auractl experiments` command (owned elsewhere) may delegate to this module.
No changes to the shared CLI in `engine/cli` are made here; the integration
point remains separate to preserve ownership boundaries.

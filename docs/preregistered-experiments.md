Preregistered Experiments: Falsifiable Hypotheses and Execution Realism
=======================================================================

Dataset names and provenance prose are not untouched-data identities. Reuse
detection is keyed to invariant content evidence and the holdout interval;
renaming a file, changing a title, or rewriting provenance cannot restore
untouched status. Prose asserting that data are untouched is an assertion, not
proof.

This document explains the core principles that govern preregistered experiments
in the Edge Tribunal framework, focusing on falsifiability, pre-result gating,
holdout usage, amendment policies, multiplicity, dependence, and the separate
notion of execution realism.

Falsifiable Hypotheses
----------------------
A preregistration must declare a *falsifiable* hypothesis before any outcome is
examined. The hypothesis includes:
* A clear statement of the expected effect.
* An economic mechanism describing why the effect should exist.
* An expected direction (e.g., "price will increase").
* An expected holding period (how long the position is held).
* A failure mode (what would falsify the hypothesis).
* A justification for why the effect might exist and why it might disappear.

The hypothesis is not a vague aspiration; it is a concrete claim that can be
contradicted by evidence. The Tribunal never treats a missed prediction as
proof of the opposite; it only records whether the hypothesis survived the
pre-registered gates.

Machine-Evaluatable Gates Before Results
----------------------------------------
Every acceptance gate in the plan is defined as a machine-evaluable rule:
* a numeric threshold,
* an explicit comparison (>=, >, <=, <, ==),
* a metric path into the evidence bundle (e.g., "primary_model.brier_improvement").

Because the gate is defined purely in terms of numbers and a comparison, the
Tribunal can evaluate it *before* any human looks at the outcome. This prevents
hindsight bias and guarantees that the decision rule is locked in advance.

Vague criteria such as "looks promising" or "good enough" cannot be expressed
in this formalism and are therefore impossible to register.

Holdout Reuse and Permanent Failed-Run Registration
---------------------------------------------------
The holdout interval on a dataset fingerprint may be claimed as "untouched"
exactly once. The persistent holdout registry records every claim and detects
exact or partial overlap with prior claims.

If a hypothesis fails its preregistered gates, the experiment is *permanently*
registered as a failed run. The record includes:
* the experiment ID,
* the plan SHA-256,
* the seal SHA-256,
* the binding SHA-256,
* the verdict (REJECTED or INVALID_EXPERIMENT),
* a timestamp, and
* the reason for failure.

This failed-run registration is immutable and serves as a public record that
the hypothesis did not survive scrutiny. It prevents the same hypothesis from
being retested on the same hidden data without explicit disclosure.

Amendments and New Experiment IDs
---------------------------------
A sealed plan cannot be altered. To change any element of the preregistration
(researchers must:
1. Create a new experiment directory.
2. Record the parent experiment ID in the new plan's identity block.
3. Provide an amendment reason (a non-empty string explaining why the change
   is necessary).
4. Register a new plan (which may differ in hypothesis, gates, etc.) and
   seal it anew.

The new experiment receives a fresh experiment ID and a fresh seal. The
original experiment remains unchanged, preserving its audit trail and its
failed-run status (if applicable). This mechanism guarantees that any change
to the hypothesis or its evaluation criteria is transparent and does not
allow "hidden" amendments.

Multiplicity
------------
The plan specifies a multiple-testing correction method (NONE_SINGLE_PRIMARY,
BONFERRONI, or HOLM_BONFERRONI) and a fixed family size for the primary
hypotheses. The correction is applied *only* to the preregistered primary
family; secondary exploratory metrics are never folded into the family.
Because the family size is locked at plan time, the procedure cannot
advantageously add or remove hypotheses after seeing the data.

Dependence and Overlap
----------------------
The tribunal evaluates dependence between experiments in two ways:
* **Statistical dependence** - overlap in the used data (including holdout
  reuse) is detected via the holdout registry and the dataset binding's
  execution-data completeness flag.
* **Conceptual dependence** - similarity in hypothesis or mechanism is left
  to the investigator to disclose; the tribunal does not automate this check.

If an experiment reuses a holdout interval while claiming it is untouched,
the hard gate "hard_no_clean_holdout_reuse" fails, leading to an
INVALID_EXPERIMENT verdict. Overlap in non-holdout data is not a hard
failure but is recorded in the audit log and may affect the interpretation of
independence across studies.

Execution Realism as a Separate Level
-------------------------------------
Execution realism concerns whether the protocol used to generate the evidence
matches the conditions under which the hypothesis would be applied in practice.
The tribunal treats execution realism as a distinct research-reality level,
separate from statistical significance.

The research-reality ladder is:
  LEVEL_0_SYNTHETIC_VALIDATION_ONLY,
  LEVEL_1_HISTORICAL_BID_ONLY_RESEARCH,
  LEVEL_2_QUOTE_AND_COST_AWARE_HISTORICAL_RESEARCH,
  LEVEL_3_PREREGISTERED_PAPER_FORWARD_TEST,
  LEVEL_4_MICRO_LIVE_HUMAN_REVIEW,
  LEVEL_5_PRODUCTION_GOVERNANCE

A historical verdict based solely on BID-only data can never exceed
FORWARD_TEST_ELIGIBLE. That verdict recommends eligibility to begin a separately
preregistered LEVEL_3 paper-forward test; it does not claim the historical
evidence itself has reached LEVEL_3. Execution realism is evaluated independently:
a statistically passing historical experiment does *not* imply that the strategy
would work with live execution costs, latency, or slippage.

Verdict Reminder (required verbatim)
-------------------------------------
A passed Tribunal verdict does not prove future profitability.
A passed historical Tribunal verdict does not authorize trading.
A passed historical Tribunal verdict only permits the next preregistered stage.

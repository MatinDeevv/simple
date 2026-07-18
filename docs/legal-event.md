# Legal-Regulatory Causal Event Research Engine

`fxresearch/models/events/legal_event.py` is a data-contract-first implementation of the
legal-regulatory event architecture. It does not infer law from text, provide
legal advice, predict legal outcomes, construct trades, or execute orders.

Its job is narrower and testable: preserve what was known when, reject
impossible citation lineage, map a recorded scenario ledger into a disclosed FX
exposure vector, and measure only the later gap-safe response in canonical BID
data.

## Required corpus

The runner expects ignored `data/raw/events/legal_events.jsonl` records conforming
to tracked `fxresearch/config/legal-event-schema.json`. Each source record must include:

```text
event_id, source_document_id, jurisdiction, authority,
published_at, known_at, legal_stage,
scenario_probabilities, pair_exposures
```

`known_at` is the earliest timestamp at which the research process could have
used the source. It is not a publication-date guess. Scenario probabilities and
pair exposures are recorded inputs, not outputs fitted to later returns.

Each recorded assessment has an immutable hash, author, model version, creation
time, optional parent-assessment hash, and a seal declaring the latest market
timestamp it could have consumed. The validator rejects altered assessment
content, duplicate hashes, forward parent links, and seals that permit market
data later than the assessment itself. This establishes provenance, not a claim
that the scenario inputs are correct.

Every scenario in the fixed ledger is supplied, probabilities sum to one, pair
exposures are bounded in `[-1,1]`, and every citation must reference a source
whose `known_at` is no later than the citing event. Conflicting duplicate event
identifiers and unavailable citations fail validation.

The in-file hash chain proves internal consistency only. Each accepted ledger
version must be externally anchored (for example, by a signed Git tag or trusted
timestamp receipt). `write_ledger_anchor` writes the deterministic root and
requires the external reference; it does not describe the local artifact itself
as immutable.

## Causal study flow

```text
immutable primary-source event at known_at
  -> validate schema, source hash, timestamps, and citation DAG
  -> create and seal assessment at assessment_created_at >= known_at
  -> fixed scenario-weighted canonical-pair pressure vector
  -> first canonical bar strictly after decision_at = max(known_at, assessment_created_at)
  -> only a fully contiguous post-event horizon is eligible
  -> estimate each pair's expected post return from its pre-event drift
  -> report baseline-adjusted abnormal return
```

Using the first bar strictly after `decision_at` avoids assuming the document or
its assessment was available before the close of a bar with the same timestamp. A gap through the
pre-event baseline or post-event target invalidates the observation instead of
creating a session-crossing return.

The current expected-return model is deliberately minimal and explicit. For
pre-event return `R_pre` across `h_pre` observed minutes and post horizon
`h_post`, it reports:

```text
expected_post_return = (h_post / h_pre) * R_pre
abnormal_return = post_return - expected_post_return
```

This is a per-pair drift adjustment, not a factor model or causal estimate.
Currency-factor controls, rolling market models, synthetic controls, matched
non-event dates, intraday seasonality, overlapping-event exclusion, and
event-clustered inference remain required before interpreting an event result.

The scenario-impact multipliers are frozen disclosed assumptions used only to
turn recorded scenario probabilities and pair exposures into a diagnostic
pressure vector. They are not measured legal-outcome quantities. Any future
corpus must supply source provenance and sensitivity ranges, or estimate them
only on a declared training partition.

## Commands

```powershell
python pipeline\legal_event.py --self-check
python pipeline\legal_event.py --events data/raw/events\legal_events.jsonl --max-rows 50000
```

The self-check uses an in-memory synthetic document only to prove timestamp,
citation, schema, and window behavior. It is not an event-study result. The
normal command fails closed until a real timestamped primary-source corpus is
provided.

## Boundary

The current FX source is BID-only. Therefore this engine does not calculate
spread-aware returns, fill probability, market impact, execution cost, PnL,
capacity, or trade recommendations. It also has no NLP model, company-security
linker, corporate knowledge graph, court-outcome dataset, or macro-vintage
database. Those are separately gated data and model additions, not fields to
invent from price candles.

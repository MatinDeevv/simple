# Legal-Regulatory Causal Event Research Engine

`pipeline/legal_event.py` is a data-contract-first implementation of the
legal-regulatory event architecture. It does not infer law from text, provide
legal advice, predict legal outcomes, construct trades, or execute orders.

Its job is narrower and testable: preserve what was known when, reject
impossible citation lineage, map a recorded scenario ledger into a disclosed FX
exposure vector, and measure only the later gap-safe response in canonical BID
data.

## Required corpus

The runner expects ignored `data_events/legal_events.jsonl` records conforming
to tracked `config/legal-event-schema.json`. Each source record must include:

```text
event_id, source_document_id, jurisdiction, authority,
published_at, known_at, legal_stage,
scenario_probabilities, pair_exposures
```

`known_at` is the earliest timestamp at which the research process could have
used the source. It is not a publication-date guess. Scenario probabilities and
pair exposures are recorded inputs, not outputs fitted to later returns.

Every scenario in the fixed ledger is supplied, probabilities sum to one, pair
exposures are bounded in `[-1,1]`, and every citation must reference a source
whose `known_at` is no later than the citing event. Conflicting duplicate event
identifiers and unavailable citations fail validation.

## Causal study flow

```text
immutable primary-source event at known_at
  -> validate schema, source hash, timestamps, and citation DAG
  -> fixed scenario-weighted canonical-pair pressure vector
  -> first canonical bar strictly after known_at
  -> only a fully contiguous post-event horizon is eligible
  -> report pair return and cross-sectional abnormal-return diagnostic
```

Using the first bar strictly after `known_at` avoids assuming the document was
available before the close of a bar with the same timestamp. A gap through the
pre-event baseline or post-event target invalidates the observation instead of
creating a session-crossing return.

## Commands

```powershell
python pipeline\legal_event.py --self-check
python pipeline\legal_event.py --events data_events\legal_events.jsonl --max-rows 50000
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

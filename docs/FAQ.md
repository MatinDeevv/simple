# Frequently asked questions

## General

### Is Azar a trading bot?

No. Azar is a causality-first research simulator. It does not execute orders, manage capital, or make profitability claims.

### What data does Azar use?

It is designed for ten Dukascopy one-minute FX BID-bar series. Raw market data is not included in the repository.

### Can I run Azar without market data?

Yes. Every module has a `--self-check` mode that uses synthetic data and runs without external data sources.

## Classical dynamics

### What does "causality-first" mean?

Every state update depends only on information available at or before the current bar. Future bars, lookahead features, and fill-forward of parameters are forbidden.

### Why is there a 60-second contiguity rule?

FX markets trade continuously during the week. A missing minute indicates a gap (data outage, weekend, holiday). The first post-gap bar resets state because no valid return can be formed across the gap.

### What is directional coupling?

It is a model of how one currency pair influences another, constrained by economic identity (e.g., EUR/USD vs. GBP/USD share USD). The coupling field is diagnosed for stability and transient growth.

## Quantum archive

### Why is the quantum work frozen?

The quantum experiments produced negative or non-predictive results. They are kept as an audited archive so the findings are reproducible, but they do not affect the active classical pipeline.

### Will quantum modules ever become active?

Only if a predeclared experiment on untouched post-2024 data shows statistically significant, out-of-sample improvement over classical comparators.

## Contributing

### How do I report a bug?

Use the [bug report issue template](https://github.com/MatinDeevv/simple/issues/new?template=bug_report.yml).

### Can I add my own FX model?

Yes. Keep it causality-safe, add a self-check, and ensure it does not modify the canonical state schema unless the schema itself is updated with documentation.

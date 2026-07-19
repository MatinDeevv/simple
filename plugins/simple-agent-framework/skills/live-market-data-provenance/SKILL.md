---
name: live-market-data-provenance
description: Safely inspect live or historical market-data availability and provenance for MatinDeevv/simple. Use for Dukascopy, FX ticks/bars, feed metadata, timestamps, source continuity, or cross-provider comparisons.
---

Use `market_dukascopy_history_url` then `market_dukascopy_probe` before download. Record provider, symbol, bid/ask/mid side, timestamp timezone, interval, retrieval time, content hash, gaps, revisions, and licensing/terms.

Probe is metadata-only. Full downloads require explicit task authorization and must remain outside Git. Availability does not validate economics, causality, execution, or trading viability.

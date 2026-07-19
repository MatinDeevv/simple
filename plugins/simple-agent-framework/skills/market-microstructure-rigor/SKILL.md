---
name: market-microstructure-rigor
description: Review FX and statistical-arbitrage execution assumptions, market microstructure, liquidity, costs, and latency. Use for backtests, fills, bar data, spreads, order timing, or execution claims.
---

Require quote side, timestamp convention, signal-to-order delay, order type, spread/slippage/fees, liquidity/participation constraint, partial-fill policy, cancel policy, session/calendar behavior, and corporate/data revisions where applicable.

Bar close is not a fill. Midprice is not executable. Bid/ask asymmetry and same-bar knowledge need explicit treatment. Return `UNKNOWN` where execution evidence is absent; do not invent costs or claim tradability.

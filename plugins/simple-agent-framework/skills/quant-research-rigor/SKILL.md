---
name: quant-research-rigor
description: Design, review, and reproduce quantitative finance research for MatinDeevv/simple. Use for FX, statistical arbitrage, time-series forecasting, causal inference, backtests, portfolio constraints, or research claims.
---

Use `simple-research` MCP for sources and `simple-tester` for receipts.

Required gates: predeclared hypothesis; time-causal features; chronological train/validation/test; embargo where needed; neutral outcomes retained; transaction/execution assumptions explicit; external baseline; failure cases; exact seeds/dependencies/commit; holdout untouched.

Report effect size and uncertainty, not headline return alone. Treat costless fills, selection after test, target leakage, regime leakage, and post-hoc thresholds as blockers. Never call a result profitable, causal, deployable, or promotable without independent evidence.

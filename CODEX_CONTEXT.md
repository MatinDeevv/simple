# Codex Session Context — MatinDeevv/simple

## Current baseline

- Branch: `main`
- Repository: `MatinDeevv/simple`
- Python: 3.11
- Package: `engine`
- Purpose: causality-first FX research simulator. Not a trading system, broker, or profitability claim.
- Root stays compact: no `src/`, `fxresearch/`, or extra wrapper package.

## Safety boundaries

- Do not inspect/download market data or run promotable research unless user explicitly asks.
- Do not alter frozen stat-arb targets, probabilities, thresholds, horizons, optimizer, or quantum archive without a scoped task.
- Never infer execution quality, tradability, causality, profitability, or quantum advantage from a self-check, paper, chart, or simulation.
- Preserve other-agent work. Never reset, checkout, or switch a shared worktree to solve a conflict.
- Use explicit `git add <paths>`; run `git diff --check` before commits.

## Merged work

- `agent1/policy-sensitivity-and-publication-hardening`: policy-population/stat-arb publication hardening.
- `agent2/package-ci-and-provenance-hardening`: package metadata, wheel smoke, schema/manifest/provenance, CI hardening.
- `agent2/package-ci-reliability`: merged; overlapping implementation was superseded by newer package/provenance tree.
- `agent1/stat-evaluation-integrity` and `agent3/edge-tribunal`: already contained by main.
- `agent2/research-reliability-hardening`: already recorded in ancestry; do not revive its obsolete alternate layout.

## Agent framework

Project plugin: `plugins/simple-agent-framework`.

- Start with `simple-repo-orchestrator`.
- Roles: `simple-scout`, `simple-builder`, `simple-tester`, `simple-reviewer`.
- Loop: preflight → scoped worktree → focused change → test receipt → review → explicit commit → CI.
- Run `plugins/simple-agent-framework/scripts/preflight.ps1` before edits.
- Test receipts are local under `.agents/receipts/` and ignored by Git.
- Global Codex hooks must remain disabled unless user explicitly approves a new active hook.

## Specialist skills

- `quant-research-rigor`: causal timing, sealed OOS, baselines, costs, provenance.
- `market-microstructure-rigor`: executable side, latency, fills, liquidity, costs.
- `causal-econometrics`: estimands, treatment timing, identification, placebo/sensitivity.
- `quantum-physics-research`: physical-model and finance-analogy boundary.
- `quantum-simulation-verification`: density matrices, channels, convergence, classical baseline.
- `live-market-data-provenance`: provider/symbol/side/timezone/continuity contract.
- `tradingview-research-discipline`: chart research only; chart feed is not execution feed.

## MCPs

`simple-research` is read-only and includes:

- Paper discovery, DOI metadata, curated research library, claim evidence gate.
- Repo snapshot, diff-risk scan, test finder, receipt summary, tool health.
- Quant protocol gate and density-matrix checks.
- Dukascopy URL construction/metadata probe, TradingView chart-link construction, feed-contract checks.

Perplexity and Firecrawl are intentionally not enabled until valid environment-only
keys exist. Add them only through `codex mcp add` after setting
`PERPLEXITY_API_KEY` or `FIRECRAWL_API_KEY`; never add secrets to files.

Never place secrets in source, config, receipts, commits, or chat. External tools are research-only; no broker/order functionality exists.

## Package and CI contract

- Development: install `requirements-core.txt`, then `pip install -e .`.
- Reproducibility proof: build wheel, install in isolated venv outside checkout, execute `auractl` core self-checks.
- Required package resources: instrument/legal/frozen-archive JSON and schemas.
- Core runtime excludes Qiskit/Qiskit Aer/matplotlib. Quantum dependencies remain isolated.
- Formal repository audit uses `python -m engine.tools.verify_repository --tree head`.
- `--tree index` is for staged-index verification. Both ignore unstaged/untracked files by contract.

## Manifest contract

- Manifests self-hash, validate schema, and verify on normal read.
- Forensic reads require explicit unverified mode.
- Promotion requires successful structured test evidence bound to commit; bare Boolean is insufficient.
- Logical output paths may differ from physical staging paths; physical bytes are hashed, logical path is published.

## First-session commands

```powershell
git status --short --branch
git log --oneline -8
python -m compileall -q engine tests
python -m pytest tests -q
python -m engine.tools.verify_repository --tree head
```

If an external API key is needed, set it for current shell only. Do not paste it into chat.

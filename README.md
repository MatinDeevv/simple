# FX Dynamics Research Simulator

This repository is a causality-first simulator for ten Dukascopy one-minute FX
BID-bar series. It is not a trading system and makes no profitability,
execution, market-causation, or physical-quantum claims.

## Current status

The canonical work is the classical dynamics pipeline: causal parameter
estimation, identity-aware directional coupling, and a numerically safe
three-pair integrator diagnostic replay. The quantum files are a frozen
negative-results research archive: numerically audited but not predictive, and
unable to affect the classical state schema, integrator, controller, or trading
path.

The residual-level FX research arena is frozen at v0.2 before new data
evaluation. Existing data end in the burned 2024 holdout; v0.2 runs only
synthetic self-checks until post-2024 data and a fresh predeclared split exist.

## Setup

Use Python 3.11 for core research and tests:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-core.txt
.venv\Scripts\python.exe -m pip install -e .
```

Qiskit Aer noise calibration has its own environment:

```powershell
python -m venv .venv-quantum
.venv-quantum\Scripts\python.exe -m pip install -r requirements-quantum.txt
```

## Reproducible checks

```powershell
python -m pytest tests -q
python -m engine.models.classical.simulate_integrator --self-check
python -m engine.quantum.quantum_lindblad --self-check
python -m engine.quantum.quantum_reservoir --self-check
python -m engine.models.statistical.stat_arb --self-check
python -m engine.models.events.legal_event --self-check
.venv-quantum\Scripts\python.exe -m engine.quantum.quantum_aer_noise --self-check
```

CI runs the test suite and numerical self-checks on every push and pull request.

For a reproducible installed-package check, build a wheel and install it in a
fresh environment; editable installation above is for source development only:

```powershell
python -m pip wheel . --no-deps -w dist
python -m pip install dist\engine-*.whl
auractl --help
auractl stat-arb --self-check
```

`pyproject.toml` declares compatible core runtime ranges. `requirements-core.txt`
is the exact Python 3.11 reproducibility set used by CI. Qiskit/Aer remains an
optional isolated environment and is not required to import or run core checks.

## Canonical contracts

- `engine/config/instruments.json` is the tracked, load-bearing instrument order.
- Ingestion writes that order into `data/canonical/manifest.json`; generated
  manifests must validate against the tracked configuration.
- A state update requires an observed contiguous 60-second predecessor.
- The first bar after a gap resets state and EWMA; it may not form a return from
  the prior session.
- Parameters and coupling are zero-order-held only from timestamps at or before
  the current bar.
- Directional coupling needs more than spectral radius: the integrator logs
  largest singular value, transient power growth, eigenvector conditioning, and
  a unit-circle pseudospectral sensitivity estimate.

## Documents

- [`docs/state-schema.md`](docs/state-schema.md): source-of-truth state and open questions.
- [`docs/dynamics.md`](docs/dynamics.md): causal parameter definitions.
- [`docs/coupling.md`](docs/coupling.md): directional coupling and identity controls.
- [`docs/integrator.md`](docs/integrator.md): replay, checkpoints, gaps, and stability.
- [`docs/stat-arb.md`](docs/stat-arb.md): frozen residual-level research contract and OQ-14 gate.
- [`docs/legal-event.md`](docs/legal-event.md): legal-event lineage, scenario, and causal-study contract.
- [`docs/quantum-redteam.md`](docs/quantum-redteam.md): frozen experiment findings.

## Promotion boundary

No model is promoted by an in-sample score or numerical invariant. Promotion
requires a predeclared target, matched classical comparators, untouched
chronological folds, leakage/placebo checks, statistical confidence, and
execution-quality data.

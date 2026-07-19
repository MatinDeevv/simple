# Contributing to Azar

Thank you for your interest in the Azar FX Dynamics Research Simulator. This document explains how to make safe, reproducible contributions.

## Before you start

- Azar is a **research simulator**, not a trading system.
- All contributions must preserve the causality-first state schema and the separation between classical dynamics and the frozen quantum archive.
- No new quantum or residual-trading module should affect the classical state schema, integrator, controller, or trading path.

## Development setup

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-core.txt
.venv\Scripts\python.exe -m pip install -e .
```

## Running checks

Always run the core checks before committing:

```powershell
python -m pytest tests -q
python -m engine.tools.verify_repository --tree head
```

If you touched quantum modules, also run:

```powershell
.venv-quantum\Scripts\python.exe -m engine.quantum.quantum_aer_noise --self-check
```

## Commit guidelines

- Keep commits focused and atomic.
- Use clear commit messages that describe *what* changed and *why*.
- Do not commit raw market data, secrets, generated artifacts, or another agent's working files.

## Pull request process

1. Fork or branch from `main`.
2. Make your changes with tests.
3. Ensure CI passes (pytest, self-checks, repository audit, and installed-package smoke test).
4. Request review.

## Questions?

Open an issue with the `question` label or see the documentation in `docs/`.

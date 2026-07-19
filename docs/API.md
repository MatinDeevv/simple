# API overview

Azar is primarily a Python package. The public surface is intentionally small and stable.

## CLI

`auractl` is the command-line interface. Entry point: `engine.cli.main:main`.

```powershell
auractl --help
auractl stat-arb --self-check
auractl verify-repository --tree head
```

## Python package

Install as an editable package for development:

```powershell
python -m pip install -e .
```

Import the engine:

```python
import engine
from engine.core.state import StateSchema
from engine.data.ingestion import load_manifest
```

## State schema

The canonical state vector is defined in `engine/core/state.py` (or equivalent). Key invariants:

- Every bar requires a contiguous 60-second predecessor.
- Gaps reset state and EWMA.
- Parameters are zero-order-held from observed timestamps only.

## Self-check entry points

| Module | Self-check command |
|--------|--------------------|
| Classical integrator | `python -m engine.models.classical.simulate_integrator --self-check` |
| Statistical arbitrage | `python -m engine.models.statistical.stat_arb --self-check` |
| Legal events | `python -m engine.models.events.legal_event --self-check` |
| Quantum Lindblad | `python -m engine.quantum.quantum_lindblad --self-check` |
| Quantum reservoir | `python -m engine.quantum.quantum_reservoir --self-check` |
| Quantum Aer noise | `.venv-quantum\Scripts\python.exe -m engine.quantum.quantum_aer_noise --self-check` |

## Repository verification

```powershell
python -m engine.tools.verify_repository --tree head
```

Use `--tree head` to verify the committed HEAD, `--tree index` for the staged index.

## Configuration files

- `engine/config/instruments.json` — canonical instrument order.
- `engine/config/schemas/*.schema.json` — JSON schemas for manifests and state artifacts.

## Note on backward compatibility

Until v1.0, the state schema and public API may change. After v1.0, breaking changes will follow semantic versioning.

"""Unified command surface for reproducible research jobs.

Thin dispatch only: model implementation remains in its owned package.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence


COMMANDS = {
    "ingest": "engine.data.ingestion.ingest",
    "coupling": "engine.models.classical.estimate_coupling",
    "dynamics": "engine.models.classical.estimate_dynamics",
    "integrator": "engine.models.classical.simulate_integrator",
    "stat-arb": "engine.models.statistical.stat_arb",
    "legal-event": "engine.models.events.legal_event",
    "quantum-lindblad": "engine.quantum.quantum_lindblad",
    "quantum-trajectories": "engine.quantum.quantum_trajectories",
    "quantum-mps": "engine.quantum.quantum_mps",
    "quantum-kernel": "engine.quantum.quantum_kernel",
    "quantum-reservoir": "engine.quantum.quantum_reservoir",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="auractl")
    parser.add_argument("command", choices=sorted(COMMANDS))
    args, remainder = parser.parse_known_args(argv)
    module_name = COMMANDS[args.command]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        parser.error(f"command {args.command!r} could not import {module_name!r}: {exc}")
    return int(module.main(remainder))


if __name__ == "__main__":
    raise SystemExit(main())

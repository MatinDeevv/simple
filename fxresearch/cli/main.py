"""Unified command surface for reproducible research jobs.

Thin dispatch only: model implementation remains in its owned package.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence


COMMANDS = {
    "ingest": "fxresearch.data.ingestion.ingest",
    "coupling": "fxresearch.models.classical.estimate_coupling",
    "dynamics": "fxresearch.models.classical.estimate_dynamics",
    "integrator": "fxresearch.models.classical.simulate_integrator",
    "stat-arb": "fxresearch.models.statistical.stat_arb",
    "legal-event": "fxresearch.models.events.legal_event",
    "quantum-lindblad": "research.quantum.quantum_lindblad",
    "quantum-trajectories": "research.quantum.quantum_trajectories",
    "quantum-mps": "research.quantum.quantum_mps",
    "quantum-kernel": "research.quantum.quantum_kernel",
    "quantum-reservoir": "research.quantum.quantum_reservoir",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="auractl")
    parser.add_argument("command", choices=sorted(COMMANDS))
    args, remainder = parser.parse_known_args(argv)
    module = importlib.import_module(COMMANDS[args.command])
    return int(module.main(remainder))


if __name__ == "__main__":
    raise SystemExit(main())

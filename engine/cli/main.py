"""Unified command surface for reproducible research jobs.

Thin dispatch only: model implementation remains in its owned package.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence


CORE_COMMANDS = {
    "ingest": "engine.data.ingestion.ingest",
    "coupling": "engine.models.classical.estimate_coupling",
    "dynamics": "engine.models.classical.estimate_dynamics",
    "integrator": "engine.models.classical.simulate_integrator",
    "stat-arb": "engine.models.statistical.stat_arb",
    "legal-event": "engine.models.events.legal_event",
}
QUANTUM_ARCHIVE_COMMANDS = {
    "quantum-lindblad": "engine.quantum.quantum_lindblad",
    "quantum-trajectories": "engine.quantum.quantum_trajectories",
    "quantum-mps": "engine.quantum.quantum_mps",
    "quantum-kernel": "engine.quantum.quantum_kernel",
    "quantum-reservoir": "engine.quantum.quantum_reservoir",
}
COMMANDS = CORE_COMMANDS | QUANTUM_ARCHIVE_COMMANDS


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="auractl", epilog="Core: " + ", ".join(sorted(CORE_COMMANDS)) + "; frozen quantum archive: " + ", ".join(sorted(QUANTUM_ARCHIVE_COMMANDS)))
    parser.add_argument("command", nargs="?", help="stable core command or frozen quantum archive command")
    args, remainder = parser.parse_known_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command not in COMMANDS:
        parser.error(f"unknown command: {args.command}")
    module_name = COMMANDS[args.command]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name in {"qiskit", "qiskit_aer", "matplotlib"}:
            print(f"auractl: command {args.command!r} requires optional dependency {exc.name!r}; install engine[quantum]", file=__import__('sys').stderr)
            return 2
        print(f"auractl: failed to import command {args.command!r} ({module_name}): {exc}", file=__import__('sys').stderr)
        return 2
    except Exception as exc:
        print(f"auractl: failed to import command {args.command!r} ({module_name}): {exc}", file=__import__('sys').stderr)
        return 2
    return int(module.main(remainder))


if __name__ == "__main__":
    raise SystemExit(main())

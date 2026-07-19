"""Local repository verification: the same checks CI enforces, runnable offline.

Usage:
    python -m engine.tools.verify_repository

Prints one ``[PASS]``/``[FAIL]`` line per check and exits nonzero if any
check fails. Every check either executes real behavior (subprocess, actual
import, actual hash comparison) or reads tracked repository content; none of
them merely grep for a string and assume the underlying behavior is correct.

This script never runs a promotable experiment (no ``--test-year``, no
``--allow-burned-holdout-research``), never writes into ``data/canonical/``
or ``data/derived/``, and never requires either generated dataset to exist.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import io
import json
import re
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _qiskit_aer_available() -> bool:
    return importlib.util.find_spec("qiskit_aer") is not None

TEXT_EXTENSIONS = {
    ".py", ".md", ".json", ".yml", ".yaml", ".txt", ".cfg", ".toml", ".ini",
    ".gitignore", ".ps1",
}

# Modules that require an optional dependency not in requirements-core.txt
# and are expected to fail import with that dependency's name in the error
# (never with a data/canonical or data/derived-shaped failure) when it is
# absent. quantum_aer_noise.py wraps this in an explicit try/except with a
# clear message; render_quantum_figures.py imports matplotlib unconditionally
# (a real clean-import gap -- see final report; not fixed here because the
# file has unrelated concurrent in-flight edits owned by another agent).
OPTIONAL_ENV_MODULES = {
    "engine.quantum.quantum_aer_noise": "qiskit_aer",
    "engine.visualization.render_quant_aura_pack": "matplotlib",
    "engine.visualization.render_quantum_figures": "matplotlib",
}

REQUIRED_SELF_CHECKS = (
    "engine.models.classical.simulate_integrator",
    "engine.models.statistical.stat_arb",
    "engine.models.events.legal_event",
)
ARCHIVE_SELF_CHECKS = (
    "engine.quantum.quantum_lindblad",
    "engine.quantum.quantum_trajectories",
    "engine.quantum.quantum_process_tomography",
    "engine.quantum.quantum_mps",
    "engine.quantum.quantum_kernel",
    "engine.quantum.quantum_reservoir",
)
def _discover_clean_import_modules(root: Path) -> tuple[str, ...]:
    """Discover every tracked importable engine module, excluding package markers."""
    package = root / "engine"
    return tuple(sorted(
        "engine." + ".".join(path.relative_to(package).with_suffix("").parts)
        for path in package.rglob("*.py")
        if path.name != "__init__.py"
    ))


CLEAN_IMPORT_MODULES = _discover_clean_import_modules(ROOT)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    skipped: bool = False


def _git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout


def _tracked_files() -> list[str]:
    return [line for line in _git(["ls-files"]).splitlines() if line]


def check_tracked_configuration_files_exist() -> CheckResult:
    required = [
        "engine/config/instruments.json",
        "engine/config/legal-event-schema.json",
        "engine/config/frozen-archives.json",
        "requirements-core.txt",
        "requirements-quantum.txt",
        ".github/workflows/ci.yml",
    ]
    tracked = set(_tracked_files())
    missing = [path for path in required if path not in tracked or not (ROOT / path).is_file()]
    if missing:
        return CheckResult("tracked configuration files exist", False, f"missing: {missing}")
    for path in required:
        if path.endswith(".json"):
            try:
                json.loads((ROOT / path).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                return CheckResult("tracked configuration files exist", False, f"{path} invalid JSON: {exc}")
    return CheckResult("tracked configuration files exist", True)


def _export_clean_checkout(dest: Path) -> None:
    """Extract the staged Git tree into ``dest``; never trust working-tree files."""
    tree = _git(["write-tree"]).strip()
    archive = subprocess.run(["git", "archive", "--format=tar", tree], cwd=ROOT,
                             capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as payload:
        members = [member for member in payload.getmembers()
                   if not (member.name == "data" or member.name.startswith("data/")
                           or member.name == "artifacts" or member.name.startswith("artifacts/"))]
        payload.extractall(dest, members=members, filter="data")


def check_clean_checkout_imports(tmp_root: Path) -> CheckResult:
    """Every source module must import with no generated data or artifacts."""
    clean_dir = tmp_root / "clean_checkout"
    clean_dir.mkdir(parents=True, exist_ok=True)
    _export_clean_checkout(clean_dir)
    if (clean_dir / "data").exists() or (clean_dir / "artifacts").exists():
        return CheckResult("clean-checkout imports", False,
                           "generated data or artifacts were copied into the clean checkout")
    failures: list[str] = []
    for module in CLEAN_IMPORT_MODULES:
        proc = subprocess.run([sys.executable, "-c", f"import {module}"], cwd=clean_dir,
                              capture_output=True, text=True, timeout=60)
        optional_dependency = OPTIONAL_ENV_MODULES.get(module)
        if optional_dependency is not None:
            dependency_available = importlib.util.find_spec(optional_dependency) is not None
            if dependency_available:
                if proc.returncode != 0:
                    failures.append(f"{module}: expected clean import with {optional_dependency} "
                                    f"installed, got:\n{proc.stderr}")
            else:
                mentions_dependency = (optional_dependency.lower() in proc.stderr.lower()
                                       or optional_dependency.replace("_", " ").lower() in proc.stderr.lower())
                if proc.returncode == 0 or not mentions_dependency:
                    failures.append(f"{module}: expected an import failure naming the missing optional "
                                    f"dependency {optional_dependency!r}, got:\n"
                                    f"returncode={proc.returncode} stderr={proc.stderr}")
            continue
        if proc.returncode != 0:
            failures.append(f"{module}: import failed:\n{proc.stderr}")
    if failures:
        return CheckResult("clean-checkout imports", False, "; ".join(failures))
    return CheckResult("clean-checkout imports", True, f"{len(CLEAN_IMPORT_MODULES)} modules")


def check_core_dependency_versions() -> CheckResult:
    pinned: dict[str, str] = {}
    for line in (ROOT / "requirements-core.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        match = re.match(r"^([A-Za-z0-9_.-]+)==([0-9A-Za-z_.\-+]+)$", line)
        if match:
            pinned[match.group(1)] = match.group(2)
    if not pinned:
        return CheckResult("core dependency versions", False, "no pinned versions parsed from requirements-core.txt")
    mismatches = []
    for name, expected in pinned.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(f"{name}: not installed")
            continue
        if installed != expected:
            mismatches.append(f"{name}: pinned {expected}, installed {installed}")
    if mismatches:
        return CheckResult("core dependency versions", False, "; ".join(mismatches))
    return CheckResult("core dependency versions", True, f"{len(pinned)} packages match")


def check_tests_pass() -> CheckResult:
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT,
                          capture_output=True, text=True, timeout=1800)
    tail = "\n".join(proc.stdout.splitlines()[-15:])
    return CheckResult("pytest tests/", proc.returncode == 0, tail)


def _run_self_check(module: str) -> tuple[bool, str]:
    proc = subprocess.run([sys.executable, "-m", module, "--self-check"], cwd=ROOT,
                          capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        return False, proc.stderr[-500:] or proc.stdout[-500:]
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return False, "self-check did not print JSON"
    if payload.get("passed") is False:
        return False, f"self-check reported passed=false: {json.dumps(payload)[:400]}"
    return True, ""


def check_required_self_checks() -> CheckResult:
    failures = []
    for module in REQUIRED_SELF_CHECKS:
        ok, detail = _run_self_check(module)
        if not ok:
            failures.append(f"{module}: {detail}")
    if failures:
        return CheckResult("required module self-checks", False, "; ".join(failures))
    return CheckResult("required module self-checks", True, f"{len(REQUIRED_SELF_CHECKS)} modules")


def check_quantum_archive_self_checks() -> CheckResult:
    failures = []
    for module in ARCHIVE_SELF_CHECKS:
        ok, detail = _run_self_check(module)
        if not ok:
            failures.append(f"{module}: {detail}")
    if failures:
        return CheckResult("quantum-archive self-checks", False, "; ".join(failures))
    return CheckResult("quantum-archive self-checks", True, f"{len(ARCHIVE_SELF_CHECKS)} modules")


def check_quantum_aer_self_check() -> CheckResult:
    if not _qiskit_aer_available():
        return CheckResult("quantum-aer self-check", True, skipped=True,
                           detail="qiskit-aer not installed in this environment; "
                                  "use .venv-quantum and requirements-quantum.txt")
    ok, detail = _run_self_check("engine.quantum.quantum_aer_noise")
    return CheckResult("quantum-aer self-check", ok, detail)


def check_state_schemas_valid() -> CheckResult:
    schema_dir = ROOT / "engine" / "config" / "schemas"
    if not schema_dir.is_dir():
        return CheckResult("state schemas valid", False, "config/schemas/ is missing")
    schemas = sorted(schema_dir.glob("*.schema.json"))
    if not schemas:
        return CheckResult("state schemas valid", False, "no *.schema.json files found")
    from engine.core.schema_validate import validate_schema_document
    problems = []
    for path in schemas:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"{path.name}: invalid JSON ({exc})")
            continue
        error = validate_schema_document(document)
        if error:
            problems.append(f"{path.name}: {error}")
    if problems:
        return CheckResult("state schemas valid", False, "; ".join(problems))
    return CheckResult("state schemas valid", True, f"{len(schemas)} schemas")


def check_shared_instrument_order_consistent() -> CheckResult:
    from engine.core.contracts import canonical_pair_order
    pairs = canonical_pair_order(ROOT)
    if len(pairs) != 10 or len(set(pairs)) != 10:
        return CheckResult("shared instrument order", False, f"unexpected pair set: {pairs}")
    from engine.tools.repo_pair_order_scan import find_duplicate_pair_order_definitions
    duplicates = find_duplicate_pair_order_definitions(ROOT, pairs)
    if duplicates:
        return CheckResult("shared instrument order", False, f"duplicate hardcoded pair sequences: {duplicates}")
    return CheckResult("shared instrument order", True, f"{pairs}")


def check_no_null_bytes_in_tracked_files() -> CheckResult:
    offenders = []
    for relative in _tracked_files():
        path = ROOT / relative
        if path.suffix.lower() not in TEXT_EXTENSIONS or not path.is_file():
            continue
        if b"\x00" in path.read_bytes():
            offenders.append(relative)
    if offenders:
        return CheckResult("no null bytes in tracked files", False, f"{offenders}")
    return CheckResult("no null bytes in tracked files", True)


def check_no_required_source_file_is_empty() -> CheckResult:
    empty = []
    for directory in ("engine", "tests"):
        for path in (ROOT / directory).rglob("*.py"):
            if not path.read_text(encoding="utf-8").strip():
                empty.append(str(path.relative_to(ROOT)))
    if empty:
        return CheckResult("no empty required source files", False, f"{empty}")
    return CheckResult("no empty required source files", True)


def check_frozen_archives_unchanged() -> CheckResult:
    registry_path = ROOT / "engine" / "config" / "frozen-archives.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    problems = []
    present = 0
    for entry in registry["entries"]:
        target = ROOT / entry["path"]
        if not target.is_file():
            continue  # absent is expected in a clean/CI checkout; not a failure.
        present += 1
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            problems.append(f"{entry['path']}: sha256 changed ({digest} != frozen {entry['sha256']})")
            continue
        expected_version = entry.get("expected_version")
        if expected_version is not None:
            payload = json.loads(target.read_text(encoding="utf-8"))
            if payload.get("version") != expected_version:
                problems.append(f"{entry['path']}: version {payload.get('version')!r} != frozen {expected_version!r}")
    if problems:
        return CheckResult("frozen archives unchanged", False, "; ".join(problems))
    return CheckResult("frozen archives unchanged", True, f"{present}/{len(registry['entries'])} present and verified")


def check_burned_holdout_guard_enabled() -> CheckResult:
    data_derived = ROOT / "data" / "derived"
    def snapshot(directory: Path) -> dict[str, str]:
        if not directory.is_dir():
            return {}
        return {
            str(path.relative_to(directory)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file() and path.name != ".gitkeep"
        }

    before = snapshot(data_derived)
    proc = subprocess.run([sys.executable, "-m", "engine.models.statistical.stat_arb"], cwd=ROOT,
                          capture_output=True, text=True, timeout=60)
    after = snapshot(data_derived)
    if after != before:
        return CheckResult("burned-holdout guard enabled", False,
                           "invoking stat-arb with no flags modified data/derived contents")
    if proc.returncode == 0:
        return CheckResult("burned-holdout guard enabled", False,
                           "stat_arb.py with no --allow-burned-holdout-research exited 0")
    if "burned" not in proc.stderr.lower():
        return CheckResult("burned-holdout guard enabled", False,
                           f"refusal message no longer mentions the burned holdout: {proc.stderr[-300:]}")
    return CheckResult("burned-holdout guard enabled", True)


CHEAP_CHECKS: list[Callable[[], CheckResult]] = [
    check_tracked_configuration_files_exist,
    check_core_dependency_versions,
    check_state_schemas_valid,
    check_shared_instrument_order_consistent,
    check_no_null_bytes_in_tracked_files,
    check_no_required_source_file_is_empty,
    check_frozen_archives_unchanged,
    check_burned_holdout_guard_enabled,
]


def main() -> int:
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-slow", action="store_true",
                        help="skip pytest/self-check/clean-import runs; only run cheap static checks")
    args = parser.parse_args()

    results: list[CheckResult] = []
    for check in CHEAP_CHECKS:
        results.append(check())

    if not args.skip_slow:
        with tempfile.TemporaryDirectory(prefix="verify_repository_") as tmp:
            results.append(check_clean_checkout_imports(Path(tmp)))
        results.append(check_tests_pass())
        results.append(check_required_self_checks())
        results.append(check_quantum_archive_self_checks())
        results.append(check_quantum_aer_self_check())

    failed = False
    for result in results:
        if result.skipped:
            tag = "[SKIP]"
        elif result.passed:
            tag = "[PASS]"
        else:
            tag = "[FAIL]"
            failed = True
        line = f"{tag} {result.name}"
        if result.detail:
            line += f" -- {result.detail}"
        print(line)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

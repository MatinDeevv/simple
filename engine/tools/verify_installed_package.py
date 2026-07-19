"""Build one fresh wheel and prove its installed console entry point works."""
from __future__ import annotations
import hashlib, json, os, subprocess, sys, tempfile, venv, zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ("data/", "artifacts/", ".git/", ".venv", "__pycache__", ".pytest_cache", "dist/", "tests/")

def _run(command: list[str], cwd: Path, env: dict[str, str]) -> dict[str, object]:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, env=env)
    return {"command": command, "exit_code": result.returncode, "sha256": hashlib.sha256((result.stdout + result.stderr).encode()).hexdigest(), "stdout": result.stdout, "stderr": result.stderr}

def main() -> int:
    started = datetime.now(timezone.utc).isoformat()
    required = {str(path.relative_to(ROOT)).replace("\\", "/") for folder in (ROOT / "engine" / "config", ROOT / "engine" / "config" / "schemas") for path in folder.rglob("*.json")}
    with tempfile.TemporaryDirectory(prefix="auractl_build_") as tmp:
        build_dir = Path(tmp) / "wheel"; build_dir.mkdir()
        build = subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(build_dir)], cwd=ROOT, capture_output=True, text=True)
        wheels = list(build_dir.glob("*.whl"))
        if build.returncode or len(wheels) != 1:
            print(build.stdout + build.stderr, file=sys.stderr); return build.returncode or 1
        wheel = wheels[0]; digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist()); missing = required - names; forbidden = [name for name in names if name.startswith(FORBIDDEN)]
        if missing or forbidden:
            print(json.dumps({"missing": sorted(missing), "forbidden": forbidden}), file=sys.stderr); return 1
        base = Path(tmp) / "isolated"; env_dir = base / "venv"; outside = base / "outside"; outside.mkdir(parents=True); venv.EnvBuilder(with_pip=True).create(env_dir)
        python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        auractl = env_dir / ("Scripts/auractl.exe" if os.name == "nt" else "bin/auractl")
        install = subprocess.run([str(python), "-m", "pip", "install", str(wheel)], capture_output=True, text=True)
        if install.returncode or not auractl.is_file(): print(install.stdout + install.stderr, file=sys.stderr); return install.returncode or 1
        environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        commands = [[str(python), "-c", "import engine, pathlib, sys; assert pathlib.Path(engine.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve())"], [str(auractl), "--help"], [str(auractl), "stat-arb", "--self-check"], [str(auractl), "integrator", "--self-check"], [str(auractl), "legal-event", "--self-check"], [str(auractl), "unknown-command"]]
        receipts = [_run(command, outside, environment) for command in commands]
        if any(item["exit_code"] != (2 if item["command"][-1] == "unknown-command" else 0) for item in receipts):
            print(json.dumps(receipts, sort_keys=True), file=sys.stderr); return 1
        pip = _run([str(python), "-m", "pip", "list", "--format=json"], outside, environment)
        if pip["exit_code"]: return 1
        dist = ROOT / "dist"; dist.mkdir(exist_ok=True); final = dist / wheel.name; final.write_bytes(wheel.read_bytes())
        member_hash = hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()
        pip_snapshot = json.loads(str(pip["stdout"]))
        payload = {
            "receipt_version": "simple-installed-wheel-receipt-v1", "wheel": wheel.name,
            "wheel_sha256": digest, "wheel_member_list_sha256": member_hash,
            "build_command": [sys.executable, "-m", "build", "--wheel"],
            "build_python_version": sys.version.split()[0], "started_at_utc": started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(), "resources": sorted(required),
            "forbidden_members": sorted(forbidden), "venv_path": str(env_dir),
            "auractl_path": str(auractl), "pip_snapshot_sha256": hashlib.sha256(json.dumps(pip_snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "receipts": [{key:value for key,value in item.items() if key not in {"stdout","stderr"}} for item in receipts],
        }
        payload["receipt_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0

if __name__ == "__main__": raise SystemExit(main())

"""Build and smoke-test a wheel from an unrelated working directory."""
from __future__ import annotations
import hashlib, json, os, subprocess, sys, tempfile, venv, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {"engine/config/instruments.json", "engine/config/legal-event-schema.json", "engine/config/frozen-archives.json"}
FORBIDDEN = ("data/", "artifacts/", ".git/", ".venv", "__pycache__", ".pytest_cache", "dist/", "tests/")

def main() -> int:
    dist = ROOT / "dist"; dist.mkdir(exist_ok=True)
    build = subprocess.run([sys.executable,"-m","build","--wheel","--outdir",str(dist)],cwd=ROOT,capture_output=True,text=True)
    if build.returncode: print(build.stdout + build.stderr, file=sys.stderr); return build.returncode
    wheel = max(dist.glob("*.whl"), key=lambda p:p.stat().st_mtime)
    with zipfile.ZipFile(wheel) as z:
        names=set(z.namelist()); missing=REQUIRED-names; forbidden=[n for n in names if n.startswith(FORBIDDEN)]
    if missing or forbidden: print(json.dumps({"missing":sorted(missing),"forbidden":forbidden}),file=sys.stderr); return 1
    with tempfile.TemporaryDirectory(prefix="auractl_isolated_") as tmp:
        base=Path(tmp); env=base/"venv"; work=base/"outside"; work.mkdir(); venv.EnvBuilder(with_pip=True).create(env)
        py=env/("Scripts/python.exe" if os.name=="nt" else "bin/python")
        install=subprocess.run([str(py),"-m","pip","install",str(wheel)],capture_output=True,text=True)
        if install.returncode: print(install.stdout+install.stderr,file=sys.stderr); return install.returncode
        commands=([str(py),"-c","import engine; print(engine.__file__)"],[str(py),"-m","engine.cli.main","--help"],[str(py),"-m","engine.cli.main","stat-arb","--self-check"],[str(py),"-m","engine.cli.main","integrator","--self-check"],[str(py),"-m","engine.cli.main","legal-event","--self-check"])
        receipts=[]
        for command in commands:
            result=subprocess.run(command,cwd=work,capture_output=True,text=True,env={k:v for k,v in os.environ.items() if k!="PYTHONPATH"})
            receipts.append({"command":command,"exit_code":result.returncode,"sha256":hashlib.sha256((result.stdout+result.stderr).encode()).hexdigest()})
            if result.returncode: print(result.stdout+result.stderr,file=sys.stderr); return result.returncode
        print(json.dumps({"wheel":wheel.name,"wheel_sha256":hashlib.sha256(wheel.read_bytes()).hexdigest(),"python":subprocess.check_output([str(py),"--version"],text=True).strip(),"receipts":receipts},sort_keys=True))
    return 0

if __name__ == "__main__": raise SystemExit(main())

from pathlib import Path
import yaml

def test_required_ci_jobs_exist():
    workflow=yaml.safe_load((Path(__file__).parents[1]/".github/workflows/ci.yml").read_text())
    assert {"core-contracts","core-tests","research-self-checks","edge-tribunal-tests","quantum-archive-checks","quantum-aer-checks","repository-audit","installed-package-smoke","agent-framework-tests"} <= set(workflow["jobs"])

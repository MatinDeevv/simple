from pathlib import Path
import yaml

ROOT = Path(__file__).parents[1]

def test_all_github_yaml_is_valid():
    paths = list((ROOT / ".github").rglob("*.yml")) + list((ROOT / ".github").rglob("*.yaml"))
    assert paths
    for path in paths:
        assert yaml.safe_load(path.read_text(encoding="utf-8")) is not None, path

def test_issue_forms_have_required_shape():
    for path in (ROOT / ".github/ISSUE_TEMPLATE").glob("*.yml"):
        if path.name == "config.yml": continue
        form = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert {"name", "description", "body"} <= set(form), path
        assert isinstance(form["body"], list) and form["body"], path

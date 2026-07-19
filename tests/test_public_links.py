from pathlib import Path

ROOT = Path(__file__).parents[1]

def test_public_files_do_not_link_to_wrong_repository():
    paths = [ROOT / "README.md", ROOT / ".github/SECURITY.md", *list((ROOT / ".github/ISSUE_TEMPLATE").glob("*"))]
    for path in paths:
        if path.is_file(): assert "Aphelion-Research/Azar" not in path.read_text(encoding="utf-8"), path

def test_readme_has_live_badge_no_bom_and_experiments_layout():
    raw=(ROOT/"README.md").read_bytes(); text=raw.decode()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert "actions/workflows/ci.yml/badge.svg" in text
    assert "experiments/" in text

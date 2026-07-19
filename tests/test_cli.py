from __future__ import annotations
from engine.cli.main import main
def test_help_and_unknown_command_are_clean(capsys):
    try: main(["--help"])
    except SystemExit as exc: assert exc.code == 0
    assert "stat-arb" in capsys.readouterr().out
    try: main(["unknown"])
    except SystemExit as exc: assert exc.code == 2

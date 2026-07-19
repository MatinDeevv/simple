from __future__ import annotations
from engine.tools import verify_installed_package
def test_real_wheel_installs_outside_checkout():
    assert verify_installed_package.main() == 0

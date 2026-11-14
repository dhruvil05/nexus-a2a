"""
tests/test_packaging.py

The package ships a py.typed marker and the Typing :: Typed classifier, so
downstream type checkers use nexus-a2a's annotations instead of treating
every name as Any.
"""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestPackaging:
    def test_py_typed_marker_in_package(self) -> None:
        assert resources.files("nexus_a2a").joinpath("py.typed").is_file()

    def test_typed_classifier(self) -> None:
        meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        assert "Typing :: Typed" in meta["project"]["classifiers"]

    def test_wheel_ships_whole_package_dir(self) -> None:
        """py.typed reaches the wheel because the wheel target is the package
        directory itself; an explicit include list would have to name it."""
        meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        wheel = meta["tool"]["hatch"]["build"]["targets"]["wheel"]
        assert wheel["packages"] == ["nexus_a2a"]
        assert "include" not in wheel and "exclude" not in wheel

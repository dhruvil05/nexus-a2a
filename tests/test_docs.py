"""
tests/test_docs.py

README/MIGRATING code examples must not use APIs removed in 2.0 (README
showed AuthManager(allow_unregistered=True), which raises TypeError).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from nexus_a2a import AgentCredentialConfig, AuthManager, AuthScheme

ROOT = Path(__file__).resolve().parent.parent
DOCS = ["README.md", "MIGRATING.md"]

# ```python fences, also inside "> " block quotes.
_FENCE = re.compile(
    r"^(?P<indent>(?:> ?)?)```python\n(?P<body>.*?)^(?P=indent)```",
    re.MULTILINE | re.DOTALL,
)


def python_blocks(name: str) -> list[str]:
    text = (ROOT / name).read_text(encoding="utf-8")
    return [m.group("body") for m in _FENCE.finditer(text)]


class TestDocs:
    @pytest.mark.parametrize("name", DOCS)
    def test_has_python_examples(self, name: str) -> None:
        if name == "README.md":
            assert len(python_blocks(name)) > 10

    @pytest.mark.parametrize("name", DOCS)
    def test_no_removed_allow_unregistered(self, name: str) -> None:
        for block in python_blocks(name):
            assert "allow_unregistered" not in block, block

    def test_readme_shows_the_replacement(self) -> None:
        blocks = "\n".join(python_blocks("README.md"))
        assert "AuthManager(default=AgentCredentialConfig(scheme=AuthScheme.NONE))" in (
            blocks
        )

    def test_documented_replacement_works(self) -> None:
        auth = AuthManager(default=AgentCredentialConfig(scheme=AuthScheme.NONE))
        assert auth is not None

    def test_removed_form_still_raises(self) -> None:
        with pytest.raises(TypeError, match="removed"):
            AuthManager(allow_unregistered=True)

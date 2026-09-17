"""
tests/test_cli.py

CLI entry points and the import graph behind them.

Regression cover for the main <-> commands import cycle fixed in v1.6.0.
main.py registers every sub-command at import time, and every sub-command
needs NexusContext/pass_ctx. While those lived in main.py the graph was
circular, which broke 'python -m nexus_a2a.cli.main' outright:

    ImportError: cannot import name 'inspect' from partially initialized
    module 'nexus_a2a.cli.commands.inspect'

The static check below is the real guard — it fails the moment a command
module imports from cli.main again, before anyone hits the ImportError.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from nexus_a2a.cli.context import NexusContext, pass_ctx
from nexus_a2a.cli.main import cli

COMMANDS = ["dev", "inspect", "ping", "replay", "run", "status", "trace", "verify"]
COMMANDS_DIR = Path(__file__).resolve().parents[1] / "nexus_a2a" / "cli" / "commands"


def run_module(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Invoke the CLI in a fresh interpreter."""
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


# ── The import cycle must not come back ───────────────────────────────────────


class TestNoImportCycle:
    @pytest.mark.parametrize("name", COMMANDS)
    def test_command_module_does_not_import_cli_main(self, name: str):
        """
        A command importing from cli.main recreates the cycle. Shared CLI
        state belongs in cli.context, which imports nothing from main.
        """
        source = (COMMANDS_DIR / f"{name}.py").read_text(encoding="utf-8")
        assert "from nexus_a2a.cli.main import" not in source
        assert "import nexus_a2a.cli.main" not in source

    def test_context_module_does_not_import_main(self):
        path = COMMANDS_DIR.parent / "context.py"
        source = path.read_text(encoding="utf-8")
        assert "from nexus_a2a.cli.main import" not in source
        assert "import nexus_a2a.cli.main" not in source

    @pytest.mark.parametrize("name", COMMANDS)
    def test_command_imports_standalone(self, name: str):
        """Each command must import cleanly without main being loaded first."""
        result = run_module(
            ["-c", f"import nexus_a2a.cli.commands.{name} as m; print(m.__name__)"]
        )
        assert result.returncode == 0, result.stderr
        assert f"nexus_a2a.cli.commands.{name}" in result.stdout


# ── Entry points ──────────────────────────────────────────────────────────────


class TestEntryPoints:
    @pytest.mark.parametrize(
        "args",
        [
            ["-m", "nexus_a2a.cli.main", "--help"],
            ["-m", "nexus_a2a.cli", "--help"],
            ["-c", "from nexus_a2a.cli.main import cli; cli()", "--help"],
        ],
        ids=["module_main", "module_package", "console_script"],
    )
    def test_invocation_succeeds(self, args: list[str]):
        result = run_module(args)
        assert result.returncode == 0, result.stderr
        assert "Commands:" in result.stdout

    @pytest.mark.parametrize(
        "args",
        [
            ["-m", "nexus_a2a.cli.main", "--help"],
            ["-m", "nexus_a2a.cli", "--help"],
        ],
        ids=["module_main", "module_package"],
    )
    def test_all_commands_registered(self, args: list[str]):
        result = run_module(args)
        for name in COMMANDS:
            assert name in result.stdout

    def test_no_circular_import_error_text(self):
        """The original failure mode, asserted directly."""
        result = run_module(["-m", "nexus_a2a.cli.main", "--help"])
        assert "circular import" not in result.stderr
        assert "partially initialized" not in result.stderr


# ── Shared context identity ───────────────────────────────────────────────────


class TestContextIdentity:
    def test_main_reexports_the_same_class(self):
        """
        Back-compat: 'from nexus_a2a.cli.main import NexusContext' still works
        and must be the SAME class object — click.make_pass_decorator matches
        on class identity, so a duplicate would silently fail to find the
        context the root group stored.
        """
        from nexus_a2a.cli.main import NexusContext as FromMain

        assert FromMain is NexusContext

    def test_main_reexports_the_same_pass_ctx(self):
        from nexus_a2a.cli.main import pass_ctx as from_main

        assert from_main is pass_ctx

    @pytest.mark.parametrize("name", COMMANDS)
    def test_commands_share_one_context_class(self, name: str):
        module = __import__(
            f"nexus_a2a.cli.commands.{name}", fromlist=["NexusContext"]
        )
        assert module.NexusContext is NexusContext


# ── Group wiring ──────────────────────────────────────────────────────────────


class TestGroupWiring:
    def test_all_commands_attached_to_group(self):
        assert set(cli.commands) == set(COMMANDS)

    def test_help_lists_commands(self):
        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        for name in COMMANDS:
            assert name in result.output

    @pytest.mark.parametrize("name", COMMANDS)
    def test_each_command_has_help(self, name: str):
        result = CliRunner().invoke(cli, [name, "--help"])
        assert result.exit_code == 0


# ── Context behaviour ─────────────────────────────────────────────────────────


class TestNexusContext:
    def test_missing_config_returns_empty_dict(self, tmp_path: Path):
        ctx = NexusContext(tmp_path / "nope.toml", verbose=False, fmt="table")
        assert ctx.load_config() == {}

    def test_loads_toml(self, tmp_path: Path):
        cfg = tmp_path / "nexus.toml"
        cfg.write_text('[agent]\nurl = "http://localhost:8001"\n', encoding="utf-8")
        ctx = NexusContext(cfg, verbose=False, fmt="table")
        assert ctx.load_config()["agent"]["url"] == "http://localhost:8001"

    def test_fields_are_stored(self, tmp_path: Path):
        ctx = NexusContext(tmp_path / "n.toml", verbose=True, fmt="json")
        assert ctx.verbose is True
        assert ctx.fmt == "json"

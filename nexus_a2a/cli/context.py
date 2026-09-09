"""
nexus_a2a.cli.context
~~~~~~~~~~~~~~~~~~~~~
The context object shared by the root CLI group and every sub-command.

This lives in its own module to break an import cycle: main.py registers the
sub-commands at import time, and every sub-command needs NexusContext and
pass_ctx. When those lived in main.py the graph was

    main -> commands.* -> main

which failed outright under 'python -m nexus_a2a.cli.main', where main.py is
executed once as '__main__' and then loaded a SECOND time as
'nexus_a2a.cli.main' when a command module imported back from it. The second
pass re-entered _register_commands() while the first command module was still
half-initialised:

    ImportError: cannot import name 'inspect' from partially initialized
    module 'nexus_a2a.cli.commands.inspect' (most likely due to a circular
    import)

The duplicate load was also a latent runtime bug beyond the import error: two
module objects meant two distinct NexusContext classes, and
click.make_pass_decorator matches on class identity, so the decorator would
have failed to find the context the root group had stored.

Nothing here imports main, so the cycle is gone rather than worked around.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from nexus_a2a.cli.output import print_error


class NexusContext:
    """
    Global CLI state, built once by the root group and handed to every
    sub-command through pass_ctx.

    Attributes:
        config_path: Path to nexus.toml (may not exist).
        verbose:     True when -v/--verbose was passed.
        fmt:         Output format, "table" or "json".
    """

    def __init__(self, config_path: Path, verbose: bool, fmt: str) -> None:
        self.config_path = config_path
        self.verbose = verbose
        self.fmt = fmt  # "table" | "json"

    def load_config(self) -> dict[str, Any]:
        """Load nexus.toml if it exists; return empty dict otherwise."""
        if self.config_path.exists():
            try:
                import tomllib  # Python 3.11+
            except ImportError:
                try:
                    import tomli as tomllib  # type: ignore[no-redef]
                except ImportError:
                    print_error(
                        "tomllib not available. Install tomli for Python <3.11."
                    )
                    return {}
            with open(self.config_path, "rb") as f:
                return tomllib.load(f)
        return {}


pass_ctx = click.make_pass_decorator(NexusContext, ensure=True)


__all__ = ["NexusContext", "pass_ctx"]

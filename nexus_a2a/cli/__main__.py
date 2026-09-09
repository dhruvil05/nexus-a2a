"""
nexus_a2a.cli.__main__
~~~~~~~~~~~~~~~~~~~~~~
Lets the CLI run as a module:

    python -m nexus_a2a.cli [ARGS]

The 'nexus' console script (pyproject [project.scripts]) is the primary entry
point; this exists so the CLI is reachable without the script on PATH — inside
a container, a CI step, or a virtualenv that was not installed with pip.
"""

from __future__ import annotations

from nexus_a2a.cli.main import cli

if __name__ == "__main__":
    cli()

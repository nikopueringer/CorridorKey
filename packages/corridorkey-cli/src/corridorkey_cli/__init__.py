"""CorridorKey CLI - command-line interface for the keying pipeline."""

from __future__ import annotations

import sys

import typer

from corridorkey_cli.commands.config import app as config_app
from corridorkey_cli.commands.doctor import doctor
from corridorkey_cli.commands.init import init
from corridorkey_cli.commands.process import process
from corridorkey_cli.commands.scan import scan
from corridorkey_cli.commands.wizard import wizard

app = typer.Typer(
    name="corridorkey",
    help="AI green screen keyer - process clips from the command line.",
    add_completion=False,
    no_args_is_help=True,
)

app.command("init")(init)
app.command("doctor")(doctor)
app.command("wizard")(wizard)
app.command("process")(process)
app.command("scan")(scan)
app.add_typer(config_app, name="config")


def main() -> None:
    """Entry point for the corridorkey console script."""
    try:
        app()
    except KeyboardInterrupt:
        from corridorkey_cli._helpers import console

        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()

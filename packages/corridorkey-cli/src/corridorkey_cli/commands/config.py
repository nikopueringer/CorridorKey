"""``corridorkey config`` subcommands."""

from __future__ import annotations

import typer
from corridorkey import export_config, load_config
from rich.table import Table

from corridorkey_cli._helpers import console

app = typer.Typer(help="Manage CorridorKey configuration.")


@app.command("show")
def config_show() -> None:
    """Print the resolved configuration (all sources merged)."""
    config = load_config()

    table = Table(title="CorridorKey Config", show_header=True, header_style="bold")
    table.add_column("Field")
    table.add_column("Value")

    for field_name in config.model_fields:
        value = getattr(config, field_name)
        table.add_row(field_name, str(value))

    console.print(table)
    console.print(
        "\n[dim]Sources (lowest to highest priority): defaults → "
        "~/.config/corridorkey/corridorkey.toml → ./corridorkey.toml → "
        "CORRIDORKEY_* env vars[/dim]"
    )


@app.command("init")
def config_init() -> None:
    """Write a starter config file to ~/.config/corridorkey/corridorkey.toml."""
    config = load_config()
    dest = export_config(config)
    console.print(f"[green]Config written:[/green] {dest}")

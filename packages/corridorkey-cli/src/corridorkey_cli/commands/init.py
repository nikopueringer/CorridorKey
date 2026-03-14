"""``corridorkey init`` - one-time environment setup."""

from __future__ import annotations

import contextlib

import typer
from corridorkey import CorridorKeyConfig, export_config, load_config
from corridorkey.model_manager import MODEL_DOWNLOAD_URL, MODEL_FILENAME, download_model, is_model_present
from rich.prompt import Confirm

from corridorkey_cli._helpers import console, err_console, make_progress

app = typer.Typer(help="One-time environment setup.")


@app.callback(invoke_without_command=True)
def init() -> None:
    """Set up CorridorKey for first use.

    - Runs the environment health check (doctor)
    - Creates the config file if missing
    - Offers to download the inference model if missing
    """
    console.print("[bold cyan]CorridorKey — Init[/bold cyan]\n")

    # 1. Run doctor inline (import here to avoid circular at module level)
    from corridorkey_cli.commands.doctor import doctor

    with contextlib.suppress(SystemExit, typer.Exit):
        doctor()

    console.print()

    # 2. Config file
    config = load_config()
    config_file = config.app_dir / "corridorkey.toml"

    if config_file.exists():
        console.print(f"[green]Config file already exists:[/green] {config_file}")
    else:
        export_config(config, path=config_file)
        console.print(f"[green]Config file created:[/green] {config_file}")

    console.print()

    # 3. Inference model
    if is_model_present(config):
        console.print("[green]Inference model found.[/green]")
        console.print("\n[bold green]Init complete. Run `corridorkey wizard` to get started.[/bold green]")
        return

    console.print(f"[yellow]Inference model not found[/yellow] in: {config.checkpoint_dir}")
    resolved_url = config.model_download_url or MODEL_DOWNLOAD_URL
    console.print(f"URL: [dim]{resolved_url}[/dim]")
    console.print()

    if not Confirm.ask("Download inference model now?", default=True):
        console.print(
            f"\nTo download manually, place [bold]{MODEL_FILENAME}[/bold] in:\n"
            f"  {config.checkpoint_dir}\n"
            "Then run [bold]corridorkey doctor[/bold] to verify."
        )
        return

    _download_with_progress(config)

    console.print("\n[bold green]Init complete. Run `corridorkey wizard` to get started.[/bold green]")


def _download_with_progress(config: CorridorKeyConfig) -> None:
    """Download the inference model with a Rich progress bar."""
    progress = make_progress()

    with progress:
        task = progress.add_task("Downloading inference model...", total=None)

        def on_progress(downloaded: int, total: int) -> None:
            progress.update(task, completed=downloaded, total=total or None)

        try:
            dest = download_model(config, on_progress=on_progress)
            progress.update(task, completed=1, total=1)

        except RuntimeError as e:
            err_console.print(f"\n[red]Download failed:[/red] {e}")
            raise typer.Exit(1) from e

    console.print(f"[green]Model saved:[/green] {dest}")

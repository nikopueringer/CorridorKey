"""``corridorkey scan`` - show clip states without processing."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from corridorkey import ClipState, CorridorKeyService
from rich.table import Table

from corridorkey_cli._helpers import console, err_console

app = typer.Typer(help="Scan a directory and show clip states.")

# Maps ClipState to a Rich colour name for the state column.
_STATE_COLOURS: dict[str, str] = {
    "EXTRACTING": "orange3",
    "RAW": "white",
    "MASKED": "blue",
    "READY": "yellow",
    "COMPLETE": "green",
    "ERROR": "red",
}


@app.callback(invoke_without_command=True)
def scan(
    clips_dir: Annotated[Path, typer.Argument(help="Directory to scan for clips.")],
) -> None:
    """Scan CLIPS_DIR and print a state table. No processing is performed."""
    if not clips_dir.exists():
        err_console.print(f"[red]Error:[/red] Path does not exist: {clips_dir}")
        raise typer.Exit(1)

    service = CorridorKeyService()
    clips = service.scan_clips(str(clips_dir))

    if not clips:
        console.print("[yellow]No clips found.[/yellow]")
        return

    table = Table(title=f"Clips in {clips_dir}", show_header=True, header_style="bold")
    table.add_column("Clip")
    table.add_column("State")
    table.add_column("Input Frames", justify="right")
    table.add_column("Alpha Frames", justify="right")
    table.add_column("Error")

    counts: dict[ClipState, int] = {}
    for clip in clips:
        colour = _STATE_COLOURS.get(clip.state.value, "white")
        input_frames = str(clip.input_asset.frame_count) if clip.input_asset else "-"
        alpha_frames = str(clip.alpha_asset.frame_count) if clip.alpha_asset else "-"
        error = clip.error_message or ""
        table.add_row(
            clip.name,
            f"[{colour}]{clip.state.value}[/{colour}]",
            input_frames,
            alpha_frames,
            f"[red]{error}[/red]" if error else "",
        )
        counts[clip.state] = counts.get(clip.state, 0) + 1

    console.print(table)

    summary_parts = [f"[bold]{len(clips)}[/bold] clip(s)"]
    for state, count in sorted(counts.items(), key=lambda x: x[0].value):
        colour = _STATE_COLOURS.get(state.value, "white")
        summary_parts.append(f"[{colour}]{count} {state.value}[/{colour}]")
    console.print("  ".join(summary_parts))

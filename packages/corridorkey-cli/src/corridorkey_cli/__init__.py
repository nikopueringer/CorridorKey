"""CorridorKey CLI - command-line interface for the keying pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from corridorkey.pipeline import PipelineResult, process_directory
from corridorkey.service import CorridorKeyService, InferenceParams, OutputConfig
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

app = typer.Typer(
    name="corridorkey",
    help="AI green screen keyer - process clips from the command line.",
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=err_console, show_path=False, markup=True)],
    )


def _print_result(result: PipelineResult) -> None:
    table = Table(title="Pipeline Results", show_header=True, header_style="bold")
    table.add_column("Clip")
    table.add_column("State")
    table.add_column("Frames")
    table.add_column("Status")

    for clip in result.clips:
        if clip.error:
            status = f"[red]FAILED: {clip.error}[/red]"
        elif clip.skipped:
            status = "[yellow]SKIPPED[/yellow]"
        else:
            status = "[green]OK[/green]"

        frames = f"{clip.frames_processed}/{clip.frames_total}" if clip.frames_total else "-"
        table.add_row(clip.name, clip.state, frames, status)

    console.print(table)
    console.print(
        f"[bold]Done:[/bold] {len(result.succeeded)} succeeded, "
        f"{len(result.failed)} failed, {len(result.skipped)} skipped"
    )


@app.command()
def process(
    clips_dir: Annotated[Path, typer.Argument(help="Directory containing clips to process.")],
    device: Annotated[str, typer.Option("--device", "-d", help="Compute device: auto, cuda, mps, cpu.")] = "auto",
    despill: Annotated[float, typer.Option("--despill", help="Green spill removal strength (0.0-1.0).")] = 1.0,
    despeckle: Annotated[bool, typer.Option("--despeckle/--no-despeckle", help="Remove small matte artifacts.")] = True,
    despeckle_size: Annotated[int, typer.Option("--despeckle-size", help="Min artifact size in pixels.")] = 400,
    refiner: Annotated[float, typer.Option("--refiner", help="Edge refiner scale (0.0 = disabled).")] = 1.0,
    linear: Annotated[bool, typer.Option("--linear", help="Treat input as linear light (not sRGB).")] = False,
    fg_format: Annotated[str, typer.Option("--fg-format", help="FG output format: exr or png.")] = "exr",
    matte_format: Annotated[str, typer.Option("--matte-format", help="Matte output format: exr or png.")] = "exr",
    comp_format: Annotated[str, typer.Option("--comp-format", help="Comp output format: exr or png.")] = "png",
    no_comp: Annotated[bool, typer.Option("--no-comp", help="Skip comp output.")] = False,
    no_processed: Annotated[bool, typer.Option("--no-processed", help="Skip processed RGBA output.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
) -> None:
    """Process all READY clips in CLIPS_DIR through the keying pipeline."""
    _setup_logging(verbose)

    if not clips_dir.exists():
        err_console.print(f"[red]Error:[/red] Path does not exist: {clips_dir}")
        raise typer.Exit(1)

    params = InferenceParams(
        input_is_linear=linear,
        despill_strength=despill,
        auto_despeckle=despeckle,
        despeckle_size=despeckle_size,
        refiner_scale=refiner,
    )

    output_config = OutputConfig(
        fg_format=fg_format,
        matte_format=matte_format,
        comp_enabled=not no_comp,
        comp_format=comp_format,
        processed_enabled=not no_processed,
    )

    current_clip: list[str] = [""]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Processing...", total=None)

        def on_progress(clip_name: str, current: int, total: int) -> None:
            if clip_name != current_clip[0]:
                current_clip[0] = clip_name
                progress.update(task, description=f"[cyan]{clip_name}[/cyan]", total=total)
            progress.update(task, completed=current)

        def on_warning(msg: str) -> None:
            progress.console.print(f"[yellow]Warning:[/yellow] {msg}")

        def on_clip_start(clip_name: str, state: str) -> None:
            progress.update(task, description=f"[cyan]{clip_name}[/cyan] ({state})", completed=0, total=None)

        result = process_directory(
            clips_dir=str(clips_dir),
            params=params,
            output_config=output_config,
            device=device,
            on_progress=on_progress,
            on_warning=on_warning,
            on_clip_start=on_clip_start,
        )

    _print_result(result)

    if result.failed:
        raise typer.Exit(1)


@app.command()
def scan(
    clips_dir: Annotated[Path, typer.Argument(help="Directory to scan for clips.")],
) -> None:
    """Scan a directory and show clip states without processing."""
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
    table.add_column("Input Frames")
    table.add_column("Alpha Frames")

    state_colors = {
        "EXTRACTING": "orange3",
        "RAW": "white",
        "MASKED": "blue",
        "READY": "yellow",
        "COMPLETE": "green",
        "ERROR": "red",
    }

    for clip in clips:
        color = state_colors.get(clip.state.value, "white")
        input_frames = str(clip.input_asset.frame_count) if clip.input_asset else "-"
        alpha_frames = str(clip.alpha_asset.frame_count) if clip.alpha_asset else "-"
        table.add_row(clip.name, f"[{color}]{clip.state.value}[/{color}]", input_frames, alpha_frames)

    console.print(table)
    console.print(f"[bold]{len(clips)}[/bold] clip(s) found")


def main() -> None:
    """Entry point for the corridorkey CLI."""
    app()


if __name__ == "__main__":
    main()

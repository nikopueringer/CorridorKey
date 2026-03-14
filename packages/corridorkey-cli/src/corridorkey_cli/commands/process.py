"""``corridorkey process`` - non-interactive batch processing."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from corridorkey import InferenceParams, OutputConfig, PipelineResult, process_directory
from rich.table import Table

from corridorkey_cli._helpers import ProgressContext, console, err_console, setup_logging

app = typer.Typer(help="Process all READY clips in a directory.")


@app.callback(invoke_without_command=True)
def process(
    clips_dir: Annotated[Path, typer.Argument(help="Directory containing clips to process.")],
    device: Annotated[str, typer.Option("--device", "-d", help="Compute device: auto, cuda, mps, cpu.")] = "auto",
    despill: Annotated[float, typer.Option("--despill", help="Green spill removal strength (0.0-1.0).")] = 1.0,
    despeckle: Annotated[bool, typer.Option("--despeckle/--no-despeckle", help="Remove small matte artifacts.")] = True,
    despeckle_size: Annotated[int, typer.Option("--despeckle-size", help="Min artifact area in pixels.")] = 400,
    refiner: Annotated[float, typer.Option("--refiner", help="Edge refiner scale (0.0 = disabled).")] = 1.0,
    linear: Annotated[bool, typer.Option("--linear", help="Treat input as linear light (not sRGB).")] = False,
    fg_format: Annotated[str, typer.Option("--fg-format", help="FG output format: exr or png.")] = "exr",
    matte_format: Annotated[str, typer.Option("--matte-format", help="Matte output format: exr or png.")] = "exr",
    comp_format: Annotated[str, typer.Option("--comp-format", help="Comp output format: exr or png.")] = "png",
    no_comp: Annotated[bool, typer.Option("--no-comp", help="Skip comp output.")] = False,
    no_processed: Annotated[bool, typer.Option("--no-processed", help="Skip processed RGBA output.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
) -> None:
    """Process all READY clips in CLIPS_DIR through the keying pipeline.

    Clips in RAW or MASKED state are skipped - they need an alpha generator
    first. Install an alpha generator package (e.g. corridorkey-gbm) to
    process those clips.
    """
    setup_logging(verbose)

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

    with ProgressContext() as prog:
        result = process_directory(
            clips_dir=str(clips_dir),
            params=params,
            output_config=output_config,
            device=device,
            on_progress=prog.on_progress,
            on_warning=prog.on_warning,
            on_clip_start=prog.on_clip_start,
        )

    _print_result(result)

    if result.failed:
        raise typer.Exit(1)


def _print_result(result: PipelineResult) -> None:
    table = Table(title="Pipeline Results", show_header=True, header_style="bold")
    table.add_column("Clip")
    table.add_column("State")
    table.add_column("Frames", justify="right")
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
